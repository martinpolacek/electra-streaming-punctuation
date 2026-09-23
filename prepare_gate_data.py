"""Make the paper's disjoint artificial fit/calibration/validation streams."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from apr.artifacts import fresh_dir, write_json, sha256
from apr.data import words_labels, normalized_words_labels, write_records, load_finetuning_data

def split_pools(train, dev, loss_mode="word_final"):
    def unique(lines):
        result = {}
        for line in lines:
            words, labels = (normalized_words_labels if loss_mode == "word_final" else words_labels)(line)
            if words:
                key = hashlib.sha256(" ".join(words).encode()).hexdigest()
                result.setdefault(key, (words, labels))
        return result
    train, dev = unique(train), unique(dev)
    overlap = set(train) & set(dev)
    for key in overlap:
        del train[key]
        del dev[key]
    pools = {"fit": train, "calibrate": {k: v for k, v in dev.items() if int(k[:8], 16) % 2 == 0},
             "validate": {k: v for k, v in dev.items() if int(k[:8], 16) % 2 == 1}}
    return pools, len(overlap)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, help="The same training .utf8 files used for punctuation fine-tuning")
    p.add_argument("--out", required=True)
    p.add_argument("--loss-mode", choices=["word_final", "original_subwords", "archived_word_final"],
                   default="word_final", help="Match the punctuation model's training mode")
    p.add_argument("--fit-streams", type=int, default=128)
    p.add_argument("--calibration-streams", type=int, default=48)
    p.add_argument("--validation-streams", type=int, default=48)
    p.add_argument("--words-per-stream", type=int, default=256)
    a = p.parse_args()
    counts = dict(fit=a.fit_streams, calibrate=a.calibration_streams, validate=a.validation_streams)
    if min(*counts.values(), a.words_per_stream) < 1:
        p.error("All stream counts and lengths must be positive")
    train, dev, statistics = load_finetuning_data(a.data_dir, a.loss_mode)
    pools, dropped = split_pools(train, dev, a.loss_mode)
    rng = np.random.default_rng(20260909)
    prepared = {}
    for split, pool in pools.items():
        keys = sorted(pool)
        rng.shuffle(keys)
        cursor = 0
        streams = []
        for index in range(counts[split]):
            words, labels, families = [], [], []
            while len(words) < a.words_per_stream:
                if cursor == len(keys):
                    raise ValueError(f"Not enough unique sentences for {split}; supply the full corpus or explicitly reduce stream counts for a smoke test")
                key = keys[cursor]
                cursor += 1
                w, y = pool[key]
                words.extend(w)
                labels.extend(y)
                families.append(key)
            streams.append({"id": f"{split}_{index:03d}", "split": split, "words": words[:a.words_per_stream],
                            "labels": labels[:a.words_per_stream], "sentence_hashes": families, "loss_mode": a.loss_mode})
        prepared[split] = streams
    out = fresh_dir(a.out)
    manifest = {"source_files": {p.name: sha256(p) for p in sorted(Path(a.data_dir).glob("*.utf8"))},
                "loss_mode": a.loss_mode, "data_statistics": statistics,
                "removed_train_dev_families": dropped, "seed": 20260909, "words_per_stream": a.words_per_stream,
                "pool_sizes": {k: len(v) for k, v in pools.items()}, "splits": {}}
    for split, rows in prepared.items():
        write_records(out / f"{split}.jsonl", rows)
        manifest["splits"][split] = {"streams": len(rows), "words": sum(len(r["words"]) for r in rows),
                                     "sha256": sha256(out / f"{split}.jsonl")}
    write_json(out / "manifest.json", manifest)
    print(f"Saved disjoint streams: {out}")

if __name__ == "__main__":
    main()
