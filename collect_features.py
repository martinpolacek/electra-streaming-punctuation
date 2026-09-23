"""Collect fast/full predictions, gate features and optional exit confidences."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from apr.artifacts import load_bundle, sha256
from apr.cache import KVCache
from apr.data import read_records, windows
from apr.model import target_logits

@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=["fit", "calibrate", "validate"], required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if a.threads < 1:
        p.error("threads must be positive")
    torch.set_num_threads(a.threads)
    if Path(a.out).exists():
        raise FileExistsError(a.out)
    model, tok, doc = load_bundle(a.model, a.device, "punctuation")
    records = read_records(a.data)
    output = {key: [] for key in ["features", "gold", "fast_pred", "full_pred", "recording", "exit_pred", "confidence"]}
    fingerprints = set()
    for index, record in enumerate(records):
        if record.get("split") != a.split or not record.get("sentence_hashes"):
            raise ValueError("Use the matching output of prepare_gate_data.py; split identity and sentence hashes are required")
        if "loss_mode" in record and record["loss_mode"] != doc["metadata"].get("loss_mode"):
            raise ValueError("Gate data loss mode does not match the punctuation model")
        fingerprints.update(record["sentence_hashes"])
        cache = KVCache(model)
        for label, item in zip(record["labels"], windows(record["words"], tok, device=a.device)):
            ids, start, end, _, target, _ = item
            fast, features = cache.step(*item)
            mask = torch.ones_like(ids, dtype=torch.bool)
            targets = torch.tensor([target-start], device=a.device)
            full, _ = target_logits(model, ids, mask, targets)
            output["features"].append(features.cpu().numpy())
            output["gold"].append(label)
            output["fast_pred"].append(int(fast.argmax()))
            output["full_pred"].append(int(full.argmax()))
            output["recording"].append(index)
            if len(model.exits) > 1:
                all_logits = target_logits(model, ids, mask, targets, all_exits=True)[0]
                output["exit_pred"].append(all_logits.argmax(-1).cpu().numpy())
                output["confidence"].append(all_logits.float().softmax(-1).amax(-1).cpu().numpy())
        print(f"{a.split}: {index+1}/{len(records)} streams", flush=True)
    arrays = {k: np.asarray(v, dtype=np.float32 if k in ("features", "confidence") else np.int64) for k, v in output.items() if v}
    metadata = {"split": a.split, "model_sha256": doc["weights_sha256"], "tokenizer_sha256": doc["tokenizer_sha256"],
                "architecture_sha256": doc["architecture_sha256"], "data_sha256": sha256(a.data), "sentence_hashes": sorted(fingerprints), "exits": list(model.exits),
                "window": 64, "lookahead": 4, "anchor_limit": 128, "loss_mode": doc["metadata"].get("loss_mode")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "xb") as f:
        np.savez_compressed(f, **arrays, metadata=np.asarray(json.dumps(metadata)))
    print(f"Saved {len(arrays['gold'])} word decisions: {a.out}")

if __name__ == "__main__":
    main()
