"""Train the 32-unit helpful-repair gate on fit data only."""
import argparse
import numpy as np
import torch
from safetensors.torch import save_file
from apr.artifacts import fresh_dir, write_json, sha256
from apr.cache import Gate
from apr.calibration import load_features
from apr.config import CLASS_WEIGHTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.threads) < 1:
        p.error("Epochs, batch size and thread count must be positive")
    torch.set_num_threads(a.threads)
    data, metadata = load_features(a.features, "fit")
    x = data["features"].astype(np.float32)
    helpful = (data["full_pred"] == data["gold"]) & (data["fast_pred"] != data["gold"])
    if not helpful.any() or helpful.all():
        raise ValueError("Gate fit needs both helpful and non-helpful examples; supply representative fit streams")
    y = torch.from_numpy(helpful.astype(np.float32))
    weights = torch.tensor(CLASS_WEIGHTS)[torch.from_numpy(data["gold"].astype(np.int64))]
    positive_weight = weights[~torch.from_numpy(helpful)].sum() / weights[torch.from_numpy(helpful)].sum()
    mean = torch.from_numpy(x.mean(0))
    scale = torch.from_numpy(np.maximum(x.std(0), 1e-4))
    features = ((torch.from_numpy(x)-mean)/scale).clamp(-8, 8)
    torch.manual_seed(a.seed)
    gate = Gate(x.shape[1])
    optimizer = torch.optim.AdamW(gate.parameters(), lr=0.001, weight_decay=0.001)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight, reduction="none")
    out = fresh_dir(a.out)
    for epoch in range(a.epochs):
        for index in torch.randperm(len(y)).split(a.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = (criterion(gate(features[index]), y[index]) * weights[index]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite gate loss")
            loss.backward()
            optimizer.step()
        print(f"epoch={epoch+1} gate_loss={float(loss.detach()):.6f}", flush=True)
    tensors = dict(gate.state_dict(), **{"normalization.mean": mean, "normalization.scale": scale})
    save_file(tensors, str(out / "gate.safetensors"))
    write_json(out / "gate.json", {**metadata, "dimension": x.shape[1], "parameters": sum(v.numel() for v in gate.parameters()),
               "weights_sha256": sha256(out / "gate.safetensors"), "fit_features_sha256": sha256(a.features),
               "seed": a.seed, "epochs": a.epochs, "batch_size": a.batch_size, "positive_weight": float(positive_weight),
               "objective": "fast_wrong_and_full_correct", "class_weights": CLASS_WEIGHTS})
    print(f"Saved gate: {out}")

if __name__ == "__main__":
    main()
