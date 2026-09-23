"""Calibration uses reference labels only; frozen policies are reused at test time."""
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from .artifacts import sha256
from .cache import Gate, confidence_score
from .metrics import metrics, feasible

def load_features(path, split):
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    metadata = json.loads(str(data.pop("metadata")))
    if metadata.get("split") != split:
        raise ValueError(f"Expected {split} features, got {metadata.get('split')}")
    if metadata.get("window") != 64 or metadata.get("lookahead") != 4 or metadata.get("anchor_limit") != 128:
        raise ValueError("Feature protocol mismatch")
    if not metadata.get("sentence_hashes"):
        raise ValueError("Sentence provenance is required")
    n = len(data["gold"])
    if not n or any(data[key].shape != (n,) for key in ("gold", "fast_pred", "full_pred")):
        raise ValueError("Invalid prediction shapes")
    if any(not np.isin(data[key], range(4)).all() for key in ("gold", "fast_pred", "full_pred")):
        raise ValueError("Labels must be in [0,3]")
    x = data["features"]
    if x.ndim != 2 or x.shape[0] != n or x.shape[1] < 13 or not np.isfinite(x).all():
        raise ValueError("Invalid or non-finite gate features")
    if not np.allclose(x[:, :4].sum(1), 1, atol=1e-5) or np.any(x[:, :4] < 0):
        raise ValueError("First four features must be class probabilities")
    if "confidence" in data:
        shape = (n, len(metadata["exits"]))
        if data["confidence"].shape != shape or data["exit_pred"].shape != shape or not np.isfinite(data["confidence"]).all():
            raise ValueError("Invalid early-exit arrays")
        if np.any((data["confidence"] < 0) | (data["confidence"] > 1)) or not np.isin(data["exit_pred"], range(4)).all():
            raise ValueError("Invalid confidence or exit labels")
        if not np.array_equal(data["exit_pred"][:, -1], data["full_pred"]):
            raise ValueError("Final exit differs from full model")
    return data, metadata

def load_gate(path, device="cpu"):
    path = Path(path)
    doc = json.loads((path / "gate.json").read_text(encoding="utf-8"))
    if sha256(path / "gate.safetensors") != doc["weights_sha256"]:
        raise ValueError("Gate integrity check failed")
    state = load_file(str(path / "gate.safetensors"), device=device)
    mean, scale = state.pop("normalization.mean"), state.pop("normalization.scale")
    if mean.shape != (doc["dimension"],) or scale.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid gate normalization")
    model = Gate(doc["dimension"]).to(device)
    model.load_state_dict(state, strict=True)
    return model.eval(), mean, scale, doc

@torch.inference_mode()
def gate_scores(features, gate, mean, scale):
    x = torch.as_tensor(features, dtype=torch.float32, device=mean.device)
    return gate(((x-mean)/scale).clamp(-8, 8)).sigmoid()

def check_identity(left, right):
    for key in ("model_sha256", "architecture_sha256", "tokenizer_sha256", "window", "lookahead", "anchor_limit"):
        if key not in left or key not in right or left[key] != right[key]:
            raise ValueError(f"Mismatched {key}; recalibrate for this exact model/tokenizer/protocol")

def midpoint_grid(scores, extra):
    ordered = np.unique(scores)
    values = list(extra)
    if len(ordered) > 1:
        indices = np.unique(np.linspace(0, len(ordered)-2, 1025).astype(int))
        values.extend(float((float(ordered[i])+float(ordered[i+1]))/2) for i in indices)
    # Store the comparison threshold in the same precision used at runtime.
    return sorted(set(float(np.float32(value)) for value in values))

def calibrate_repair(data, scores):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape != data["gold"].shape or not np.isfinite(scores).all():
        raise ValueError("Invalid policy scores")
    reference = metrics(data["gold"], data["full_pred"])
    curve = []
    for threshold in midpoint_grid(scores, [-1e9, 1e9]):
        repair = scores >= np.float32(threshold)
        q = metrics(data["gold"], np.where(repair, data["full_pred"], data["fast_pred"]))
        curve.append({"threshold": threshold, "repair_fraction": float(repair.mean()), "quality": q, "feasible": feasible(q, reference)})
    accepted = [row for row in curve if row["feasible"]]
    return min(accepted, key=lambda row: (row["repair_fraction"], -row["quality"]["weighted_f1"])), curve

def choose_exit(data, exits, threshold):
    if threshold is None:
        return data["full_pred"], np.full(len(data["gold"]), exits[-1])
    passed = data["confidence"] >= np.float32(threshold)
    passed[:, -1] = True
    first = passed.argmax(1)
    return data["exit_pred"][np.arange(len(first)), first], np.asarray(exits)[first]

def calibrate_exit(data, exits):
    reference = metrics(data["gold"], data["full_pred"])
    curve = []
    for threshold in midpoint_grid(data["confidence"][:, :-1], [0, 1, 0.99]) + [None]:
        pred, depths = choose_exit(data, exits, threshold)
        q = metrics(data["gold"], pred)
        curve.append({"threshold": threshold, "kind": "full" if threshold is None else "exit", "mean_depth": float(depths.mean()), "quality": q, "feasible": feasible(q, reference)})
    selected = min((row for row in curve if row["feasible"]), key=lambda row: (row["mean_depth"], -row["quality"]["weighted_f1"], row["kind"] != "full"))
    return selected, curve
