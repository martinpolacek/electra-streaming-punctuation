"""Restore punctuation using full L6, real early exit or selective KV reuse."""
import argparse
import json
from pathlib import Path
import torch
from apr.inference import Predictor
from apr.data import normalize_words


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Fine-tuned model bundle; the public pretrained HF encoder needs fine-tuning first")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--input", type=Path, help="One recording as UTF-8 text; line breaks are whitespace")
    p.add_argument("--mode", choices=["full", "exit", "fast", "gate", "margin", "entropy"], default="full")
    p.add_argument("--policy")
    p.add_argument("--gate")
    p.add_argument("--tau", type=float)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--format", choices=["text", "jsonl"], default="text")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if a.threads < 1:
        p.error("threads must be positive")
    torch.set_num_threads(a.threads)
    raw = a.text if a.text is not None else a.input.read_text(encoding="utf-8")
    words = normalize_words(raw)
    predictor = Predictor(a.model, a.mode, a.policy, a.gate, a.tau, a.window, a.device)
    if a.format == "jsonl":
        for row in predictor.predict_words(words):
            print(json.dumps(row, ensure_ascii=False))
    else:
        punctuation = ["", "?", ".", ","]
        print(" ".join(row["word"] + punctuation[row["label"]] for row in predictor.predict_words(words)))

if __name__ == "__main__":
    main()
