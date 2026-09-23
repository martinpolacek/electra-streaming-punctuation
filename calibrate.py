"""Select matched-quality policies on calibration data, then freeze them."""
import argparse
from pathlib import Path
import numpy as np
import torch
from apr.artifacts import write_json, sha256
from apr.cache import confidence_score
from apr.calibration import load_features, load_gate, gate_scores, check_identity, calibrate_repair, calibrate_exit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True)
    p.add_argument("--gate", help="Optional learned gate directory")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if Path(a.out).exists():
        raise FileExistsError(a.out)
    data, meta = load_features(a.features, "calibrate")
    support = np.bincount(data["gold"], minlength=4)
    if np.any(support[1:] == 0):
        raise ValueError("Calibration must include all three punctuation classes")
    policies, curves = {}, {}
    for kind in ("margin", "entropy"):
        score = confidence_score(torch.from_numpy(data["features"]), kind).numpy()
        policies[kind], curves[kind] = calibrate_repair(data, score)
    gate_sha = None
    if a.gate:
        gate, mean, scale, gate_meta = load_gate(a.gate)
        check_identity(meta, gate_meta)
        if set(meta["sentence_hashes"]) & set(gate_meta["sentence_hashes"]):
            raise ValueError("Gate fit and calibration share normalized sentences")
        score = gate_scores(data["features"], gate, mean, scale).numpy()
        policies["gate"], curves["gate"] = calibrate_repair(data, score)
        gate_sha = gate_meta["weights_sha256"]
    if "confidence" in data:
        policies["exit"], curves["exit"] = calibrate_exit(data, meta["exits"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(a.out, {"format_version": 1, "metadata": meta, "gate_sha256": gate_sha,
                "calibration_features_sha256": sha256(a.features), "policies": policies, "curves": curves,
                "constraint": "W-F1 >= full; question/period/comma F1 >= full minus 0.5 percentage points"})
    for name, row in policies.items():
        print(name, {key: value for key, value in row.items() if key != "quality"})

if __name__ == "__main__":
    main()
