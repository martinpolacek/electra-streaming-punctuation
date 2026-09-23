"""Self-contained, strict, safetensors model bundles; no executable checkpoints."""
import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from .config import Config, LABELS, TOKENIZER_ID, TOKENIZER_REVISION
from .model import Encoder, Punctuator

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

def tokenizer_hash(directory):
    directory = Path(directory)
    return object_hash({p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file()})

def architecture_hash(document):
    return object_hash({key: document[key] for key in ("architecture", "exits", "labels", "stage")})

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def fresh_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a nonempty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path

def tokenizer_from(path=TOKENIZER_ID, revision=None):
    kwargs = {"token": False, "use_fast": True, "trust_remote_code": False}
    if not Path(path).exists():
        if revision is None and path == TOKENIZER_ID:
            revision = TOKENIZER_REVISION
        if revision:
            kwargs["revision"] = revision
    tok = AutoTokenizer.from_pretrained(path, **kwargs)
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "[MASK]"})
    if tok.pad_token_id is None or not tok.is_fast:
        raise ValueError("A fast tokenizer with a padding token and word alignment is required")
    return tok

def save_bundle(path, model, tokenizer, metadata=None):
    path = fresh_dir(path)
    stage = "punctuation" if isinstance(model, Punctuator) else "encoder"
    document = {"format_version": 1, "stage": stage, "architecture": model.config.to_dict(),
                "labels": LABELS, "exits": list(model.exits) if stage == "punctuation" else [],
                "metadata": metadata or {}}
    save_file({k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}, str(path / "model.safetensors"))
    tokenizer.save_pretrained(path / "tokenizer")
    document["weights_sha256"] = sha256(path / "model.safetensors")
    document["tokenizer_sha256"] = tokenizer_hash(path / "tokenizer")
    document["architecture_sha256"] = architecture_hash(document)
    write_json(path / "config.json", document)
    return document

def load_bundle(path, device="cpu", stage=None):
    path = Path(path)
    doc = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if doc.get("format_version") != 1 or doc.get("labels") != LABELS or doc.get("stage") not in ("encoder", "punctuation"):
        raise ValueError("Unsupported bundle or label mapping")
    if stage and stage != doc["stage"]:
        raise ValueError(f"Expected a {stage} bundle; this is a {doc['stage']} bundle")
    for name, key in [("model.safetensors", "weights_sha256")]:
        if sha256(path / name) != doc[key]:
            raise ValueError(f"Bundle integrity check failed: {name}")
    if tokenizer_hash(path / "tokenizer") != doc["tokenizer_sha256"] or architecture_hash(doc) != doc["architecture_sha256"]:
        raise ValueError("Bundle integrity check failed: tokenizer files or architecture")
    config = Config(**doc["architecture"])
    model = Punctuator(config, doc["exits"]) if doc["stage"] == "punctuation" else Encoder(config)
    model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
    tok = tokenizer_from(str(path / "tokenizer"))
    if len(tok) != config.vocab_size or tok.pad_token_id != config.pad_token_id:
        raise ValueError("Tokenizer/model vocabulary mismatch")
    return model.to(device).eval(), tok, doc

def convert_encoder_state(state, source="legacy"):
    result = {}
    for key, value in state.items():
        if key == "embeddings.position_ids" or ".moe.router.gate." in key:
            continue
        if source == "legacy":
            key = key.replace(".moe.experts.0.dense_up.", ".ffn.dense_up.").replace(".moe.experts.0.dense_down.", ".ffn.dense_down.").replace(".moe.LayerNorm.", ".ffn.LayerNorm.")
        elif source == "hf":
            key = key.replace(".intermediate.dense.", ".ffn.dense_up.").replace(".output.dense.", ".ffn.dense_down.") if ".attention." not in key else key
            if ".attention." not in key:
                key = key.replace(".output.LayerNorm.", ".ffn.LayerNorm.")
        else:
            raise ValueError(source)
        if key in result:
            raise ValueError(f"Duplicate converted tensor: {key}")
        result[key] = value
    return result
