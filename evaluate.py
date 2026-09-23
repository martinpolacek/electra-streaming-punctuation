"""Evaluate already aligned word labels; raw ASR alignment is outside this CLI."""
import argparse
import json
import torch
from apr.data import read_records
from apr.inference import Predictor
from apr.metrics import metrics

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="JSONL words/labels, optional deleted_reference_labels")
    p.add_argument("--mode", choices=["full", "exit", "fast", "gate", "margin", "entropy"], default="full")
    p.add_argument("--policy")
    p.add_argument("--gate")
    p.add_argument("--tau", type=float)
    p.add_argument("--window", type=int, choices=[32, 64, 128], default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if a.threads < 1:
        p.error("threads must be positive")
    torch.set_num_threads(a.threads)
    predictor = Predictor(a.model, a.mode, a.policy, a.gate, a.tau, window=a.window, device=a.device)
    gold, pred, deleted = [], [], []
    repairs, depths = 0, 0
    for record in read_records(a.data):
        output = list(predictor.predict_words(record["words"]))
        gold.extend(record["labels"])
        pred.extend(row["label"] for row in output)
        deleted.extend(record.get("deleted_reference_labels", []))
        repairs += sum(row["repair"] for row in output)
        depths += sum(row["depth"] for row in output)
    result = metrics(gold, pred, deleted)
    count = len(gold)
    repair_fraction = None if a.mode == "exit" else (1.0 if a.mode == "full" and count else repairs/count if count else 0.0)
    result.update(words=count, repair_fraction=repair_fraction, mean_depth=depths/count if count else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
