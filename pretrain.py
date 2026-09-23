"""Pretrain a dense ELECTRA encoder with the paper's RTD recipe."""
import argparse
import random
from pathlib import Path
import torch
from apr.artifacts import fresh_dir, tokenizer_from, save_bundle, write_json, sha256
from apr.config import Config, FAMILIES, TOKENIZER_ID
from apr.pretraining import RTD, AdamWNoBias, CorpusSampler
from apr.training import seed_all

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default=TOKENIZER_ID)
    p.add_argument("--family", choices=FAMILIES, default="Small")
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--steps", type=int, default=1000000)
    p.add_argument("--warmup", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--save-every", type=int, default=50000)
    p.add_argument("--stop-after", type=int, default=0, help="Pause after this absolute update; schedule still uses --steps")
    p.add_argument("--resume", type=Path, help="Resume a training.pt file into a NEW output directory")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    a = p.parse_args()
    if not 0 <= a.warmup < a.steps or min(a.batch_size, a.threads, a.save_every) < 1 or a.stop_after < 0:
        p.error("Invalid training schedule or batch/thread/checkpoint limit")
    if a.precision == "bf16" and (not a.device.startswith("cuda") or not torch.cuda.is_bf16_supported()):
        p.error("BF16 pretraining requires a compatible CUDA GPU")
    torch.set_num_threads(a.threads)
    seed_all(a.seed)
    tok = tokenizer_from(a.tokenizer)
    config = Config.family(a.family, a.layers, vocab_size=len(tok), pad_token_id=tok.pad_token_id)
    system = RTD(config, tok).to(a.device)
    decay, other = [], []
    for name, param in system.named_parameters():  # PyTorch deduplicates tied embeddings.
        (other if "LayerNorm" in name or "bias" in name else decay).append(param)
    optimizer = AdamWNoBias([{"params": decay, "weight_decay": 0.01}, {"params": other, "weight_decay": 0.0}])
    def factor(step):
        return step/a.warmup if a.warmup and step < a.warmup else max(0.0, (a.steps-step)/(a.steps-a.warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    sampler = CorpusSampler(a.data_dir, a.batch_size, a.seed)
    data_hashes = {f.name: sha256(f) for f in sampler.files}
    identity = {"architecture": config.to_dict(), "steps": a.steps, "warmup": a.warmup, "batch_size": a.batch_size,
                "seed": a.seed, "precision": a.precision, "data_sha256": data_hashes,
                "tokenizer": tok.backend_tokenizer.to_str()}
    start = 0
    if a.resume:
        state = torch.load(a.resume, map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise ValueError("Resume configuration, corpus or tokenizer differs")
        system.load_state_dict(state["system"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        sampler.load_state_dict(state["sampler"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if a.device.startswith("cuda") and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start = state["step"]
    end = min(a.stop_after, a.steps) if a.stop_after else a.steps
    if end <= start:
        raise ValueError("Requested end must be after the checkpoint step")
    out = fresh_dir(a.out)
    write_json(out / "run.json", {"arguments": vars(a) | {"resume": str(a.resume) if a.resume else None}, "data_sha256": data_hashes})
    system.train()
    for step in range(start, end):
        batch = tok(sampler.batch(), return_tensors="pt", padding=True, truncation=True, max_length=128)
        ids = batch["input_ids"].to(a.device)
        mask = batch["attention_mask"].to(a.device).bool()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=torch.device(a.device).type, dtype=torch.bfloat16, enabled=a.precision == "bf16"):
            loss, gen, disc = system(ids, mask)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at update {step+1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(system.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == start or (step+1) % 100 == 0:
            print(f"step={step+1} generator={float(gen):.6f} discriminator={float(disc):.6f}", flush=True)
        if (step+1) % a.save_every == 0 or step+1 == end:
            state = {"identity": identity, "step": step+1, "system": system.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "sampler": sampler.state_dict(), "python_rng": random.getstate(),
                     "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if a.device.startswith("cuda") else None}
            temp = out / "training.pt.tmp"
            torch.save(state, temp)
            temp.replace(out / "training.pt")
    save_bundle(out / "encoder", system.encoder, tok, {"recipe": "RTD", "seed": a.seed, "updates": end,
                "planned_updates": a.steps, "completed_training": end == a.steps, "sequence_length": 128,
                "batch_size": a.batch_size, "precision": a.precision, "data_sha256": data_hashes})
    print(f"Saved encoder: {out / 'encoder'}")

if __name__ == "__main__":
    main()
