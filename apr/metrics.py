"""Punctuation-support-weighted F1 in percentage points, excluding NONE."""
import numpy as np

def metrics(gold, pred, deleted_labels=()):
    gold, pred = np.asarray(gold), np.asarray(pred)
    if gold.shape != pred.shape or gold.ndim != 1 or not np.isin(gold, range(4)).all() or not np.isin(pred, range(4)).all():
        raise ValueError("Expected aligned labels in [0,3]")
    if not np.isin(deleted_labels, range(4)).all():
        raise ValueError("Invalid deleted labels")
    confusion = np.bincount((gold.astype(np.int64) * 4 + pred.astype(np.int64)), minlength=16).reshape(4, 4)
    confusion[:, 0] += np.bincount(np.asarray(deleted_labels, dtype=np.int64), minlength=4)
    support = confusion.sum(axis=1)
    denom = support + confusion.sum(axis=0)
    f1 = np.divide(200.0 * np.diag(confusion), denom, out=np.zeros(4), where=denom != 0)
    total = int(support[1:].sum())
    weighted = float(f1[1:] @ support[1:] / total) if total else 0.0
    return {"weighted_f1": weighted, "question_f1": float(f1[1]), "period_f1": float(f1[2]), "comma_f1": float(f1[3]),
            "punctuation_support": total, "confusion": confusion.tolist()}

def feasible(candidate, reference):
    return candidate["weighted_f1"] >= reference["weighted_f1"] - 1e-9 and all(candidate[key] >= reference[key] - 0.5 - 1e-9 for key in ("question_f1", "period_f1", "comma_f1"))
