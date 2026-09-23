import argparse
import math
import random
from pathlib import Path
import numpy as np
import torch
from .artifacts import fresh_dir, load_bundle, save_bundle, write_json, sha256
from .legacy_data import generate_random_lenghts, connect_sentences
from .data import make_batch, load_finetuning_data, filter_lexical_blocks
from .config import CLASS_WEIGHTS
from .metrics import metrics
from .model import Punctuator

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def masked_ce(logits, labels, mask, weights):
    if not mask.any():
        raise ValueError("No supervised positions")
    return torch.nn.functional.cross_entropy(logits[mask], labels[mask], weight=weights)

@torch.inference_mode()
def validate(model, blocks, tokenizer, device, batch_size=8, loss_mode="original_subwords"):
    model.eval()
    gold = []
    predicted = {str(depth): [] for depth in model.exits}
    for offset in range(0, len(blocks), batch_size):
        ids, mask, labels, _, ends = make_batch(blocks[offset:offset+batch_size], tokenizer, loss_mode)
        outputs = model(ids.to(device), mask.to(device))
        gold.extend(labels[ends].tolist())
        for depth, logits in zip(model.exits, outputs):
            predicted[str(depth)].extend(logits.cpu().argmax(-1)[ends].tolist())
    return {depth: metrics(gold, labels) for depth, labels in predicted.items()}

def main(exit_training=False):
    p = argparse.ArgumentParser(description="Train frozen intermediate exit heads" if exit_training else "Fine-tune punctuation ELECTRA")
    p.add_argument("--model" if exit_training else "--encoder", required=True)
    p.add_argument("--data-dir", required=True, help="UTF-8 .utf8 files; exclude the temporal test before this step")
    p.add_argument("--out", required=True)
    p.add_argument("--loss-mode", default="word_final", choices=["original_subwords", "word_final", "archived_word_final"], help="word_final: manuscript recipe (default); other modes import the historical training/ablation recipes")
    p.add_argument("--epochs", type=int, default=2 if exit_training else 4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=0, help="Smoke test only; zero means the complete run")
    p.add_argument("--max-validation-blocks", type=int, default=3000)
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1 or a.threads < 1 or a.max_steps < 0 or a.max_validation_blocks < 1:
        p.error("Invalid epochs, batch size, threads or step/block limits")
    torch.set_num_threads(a.threads)
    seed_all(a.seed)
    source = a.model if exit_training else a.encoder
    loaded, tok, doc = load_bundle(source, a.device, "punctuation" if exit_training else "encoder")
    if exit_training:
        if len(loaded.exits) != 1:
            raise ValueError("Supply a fine-tuned model with only its final head")
        if doc["metadata"].get("loss_mode") != a.loss_mode:
            raise ValueError("Exit heads must use the base model's loss mode")
        model = loaded
        model.add_exits()
    else:
        model = Punctuator(loaded.config).to(a.device)
        model.backbone.load_state_dict(loaded.state_dict(), strict=True)
    if model.config.max_position_embeddings < 512:
        raise ValueError("Original fine-tuning requires 512 position embeddings")
    train, dev, data_statistics = load_finetuning_data(a.data_dir, a.loss_mode)
    train_lengths = generate_random_lenghts(train)
    dev_lengths = generate_random_lenghts(dev)
    dev_blocks = connect_sentences(dev, dev_lengths)
    if a.loss_mode == "word_final":
        dev_blocks, dropped = filter_lexical_blocks(dev_blocks, "validation")
        data_statistics["discarded_validation_blocks"] = dropped
    dev_blocks = dev_blocks[:a.max_validation_blocks]
    print(f"Data preparation: {data_statistics}", flush=True)
    out = fresh_dir(a.out)
    metadata = {"recipe": "exit" if exit_training else "fine-tuning", "loss_mode": a.loss_mode,
                "seed": a.seed, "source_weights_sha256": doc["weights_sha256"],
                "class_weights": CLASS_WEIGHTS, "validation_fraction": 0.05, "split_seed": 0,
                "data_sha256": {f.name: sha256(f) for f in sorted(Path(a.data_dir).glob("*.utf8"))},
                "data_statistics": data_statistics, "arguments": vars(a)}
    write_json(out / "run.json", metadata)
    weights = torch.tensor(CLASS_WEIGHTS, device=a.device)
    groups = [{"params": [v for v in model.heads.parameters() if v.requires_grad], "lr": 2e-4}]
    if not exit_training:
        groups.insert(0, {"params": model.backbone.parameters(), "lr": 5e-5})
    optimizer = torch.optim.AdamW(groups, lr=2e-4)
    global_step = 0
    history = []
    stop = False
    for epoch in range(1, a.epochs+1):
        blocks = connect_sentences(train, train_lengths)
        dropped = 0
        if a.loss_mode == "word_final":
            blocks, dropped = filter_lexical_blocks(blocks, "training")
        random.shuffle(blocks)
        model.train()  # Archived frozen-head training retains backbone dropout during training.
        for step, start in enumerate(range(0, len(blocks), a.batch_size)):
            if not exit_training and epoch > 1 and step and step % 10000 == 0:
                for group in optimizer.param_groups:
                    group["lr"] *= 0.95
            batch = [x.to(a.device) for x in make_batch(blocks[start:start+a.batch_size], tok, a.loss_mode)]
            ids, mask, labels, supervised, _ = batch
            optimizer.zero_grad(set_to_none=True)
            outputs = model(ids, mask)
            active = outputs[:-1] if exit_training else outputs
            loss = sum(masked_ce(logits, labels, supervised, weights) for logits in active) / len(active)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            if step == 0 or global_step % 2000 == 0:
                print(f"epoch={epoch} step={global_step} loss={float(loss.detach()):.6f}", flush=True)
            if a.max_steps and global_step >= a.max_steps:
                stop = True
                break
        quality = validate(model, dev_blocks, tok, a.device, a.batch_size, a.loss_mode)
        history.append({"epoch": epoch, "steps": global_step,
                        "discarded_training_blocks": dropped,
                        "validation_word_final": quality[str(model.exits[-1])],
                        "validation_by_depth": quality})
        save_bundle(out / f"epoch{epoch}", model, tok, {**metadata, "completed_epoch": not stop, "epoch": epoch, "steps": global_step})
        if stop:
            break
    save_bundle(out / "model", model, tok, {**metadata, "completed_training": not stop, "epochs_completed": epoch if not stop else epoch-1, "steps": global_step})
    write_json(out / "training_history.json", history)
    print(f"Saved {'smoke-test' if stop else 'trained'} model: {out / 'model'}")
