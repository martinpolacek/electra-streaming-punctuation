"""Import a pretrained HF encoder or an original tensor-only checkpoint."""
import argparse
from pathlib import Path
import torch
from transformers import AutoModel
from apr.artifacts import tokenizer_from, save_bundle, convert_encoder_state, sha256
from apr.config import Config, FAMILIES, TOKENIZER_ID
from apr.model import Encoder, Punctuator

def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--hf", help="HF encoder ID or local native Transformers directory")
    source.add_argument("--legacy", type=Path, help="Original tensor-only checkpoint; loaded with weights_only=True")
    p.add_argument("--revision")
    p.add_argument("--loss-mode", choices=["word_final", "original_subwords", "archived_word_final"], help="Required for legacy task checkpoints: training loss provenance cannot be inferred from weights")
    p.add_argument("--tokenizer", default=TOKENIZER_ID)
    p.add_argument("--kind", choices=["encoder", "punctuation", "exit"], default="encoder")
    p.add_argument("--family", choices=FAMILIES, default="Small")
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    if a.legacy and a.kind != "encoder" and a.loss_mode is None:
        p.error("Legacy punctuation/exit import requires --loss-mode from the checkpoint's training record")
    if (a.hf or a.kind == "encoder") and a.loss_mode is not None:
        p.error("Pretrained encoder import has no punctuation loss mode")
    if a.hf:
        if a.kind != "encoder":
            p.error("HF import here accepts a pretrained encoder, not a punctuation classifier")
        original = AutoModel.from_pretrained(a.hf, revision=a.revision, token=False, trust_remote_code=False).eval()
        hc = original.config
        if hc.model_type != "electra" or hc.hidden_act != "gelu" or hc.type_vocab_size != 2 or hc.layer_norm_eps != 1e-12 or hc.attention_probs_dropout_prob != hc.hidden_dropout_prob:
            raise ValueError("Unsupported HF encoder geometry or activation")
        c = Config(hc.embedding_size, hc.hidden_size, hc.intermediate_size, hc.num_attention_heads, hc.num_hidden_layers,
                   hc.vocab_size, hc.pad_token_id, hc.max_position_embeddings, hc.hidden_dropout_prob)
        model = Encoder(c)
        state = convert_encoder_state(original.state_dict(), "hf")
        if "embeddings_project.weight" not in state and c.embedding_size == c.hidden_size:
            state["embeddings_project.weight"] = torch.eye(c.hidden_size)
            state["embeddings_project.bias"] = torch.zeros(c.hidden_size)
        model.load_state_dict(state, strict=True)
        tok = tokenizer_from(a.hf, a.revision)
        metadata = {"source": a.hf, "revision": a.revision, "fine_tuned": False}
    else:
        tok = tokenizer_from(a.tokenizer)
        c = Config.family(a.family, a.layers, vocab_size=len(tok), pad_token_id=tok.pad_token_id)
        raw = torch.load(a.legacy, map_location="cpu", weights_only=True)
        if not isinstance(raw, dict) or not all(isinstance(v, torch.Tensor) for v in raw.values()):
            raise ValueError("Expected a flat tensor state dict")
        raw = {k.removeprefix("module."): v for k, v in raw.items()}
        if a.kind == "encoder":
            model = Encoder(c)
            enc = {k.removeprefix("electra."): v for k, v in raw.items() if k.startswith("electra.")}
            model.load_state_dict(convert_encoder_state(enc or raw), strict=True)
        else:
            if a.kind == "exit" and (a.family != "Small" or a.layers != 6):
                raise ValueError("Archived exit importer supports Small-L6 only")
            exits = [2, 3, 4, 5, 6] if a.kind == "exit" else [a.layers]
            model = Punctuator(c, exits)
            enc = {k.removeprefix("bert_layer."): v for k, v in raw.items() if k.startswith("bert_layer.")}
            state = {"backbone."+k: v for k, v in convert_encoder_state(enc).items()}
            for key, value in raw.items():
                if key.startswith("bert_layer."):
                    continue
                if a.kind == "exit" and key.startswith("heads."):
                    parts = key.split(".")
                    key = ".".join(["heads", str(exits[int(parts[1])]), *parts[2:]])
                elif key.startswith("lin."):
                    key = f"heads.{a.layers}.0." + key[4:]
                elif key.startswith("lin2."):
                    key = f"heads.{a.layers}.2." + key[5:]
                state[key] = value
            model.load_state_dict(state, strict=True)
        metadata = {"source": a.legacy.name, "source_sha256": sha256(a.legacy), "loss_mode": a.loss_mode if a.kind != "encoder" else None}
    if len(tok) != c.vocab_size or tok.pad_token_id != c.pad_token_id:
        raise ValueError("Tokenizer does not match the encoder")
    save_bundle(a.out, model, tok, metadata)
    print(f"Saved {a.kind} bundle: {a.out}")

if __name__ == "__main__":
    main()
