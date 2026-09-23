import numpy as np
import pytest
from apr.calibration import calibrate_repair, calibrate_exit, check_identity
from apr.metrics import metrics, feasible
from prepare_gate_data import split_pools


def test_repair_minimizes_rate_under_strict_constraints():
    gold = np.array([1,2,3,0]*10)
    full = gold.copy()
    fast = gold.copy()
    fast[0] = 0
    data = dict(gold=gold, full_pred=full, fast_pred=fast)
    scores = np.zeros(len(gold), dtype=np.float32)
    scores[0] = 1
    selected, curve = calibrate_repair(data, scores)
    assert selected["repair_fraction"] == pytest.approx(1/40)
    assert selected["quality"]["weighted_f1"] == 100
    # Equal scores are a valid degenerate case; always repair remains feasible.
    selected, _ = calibrate_repair(data, np.zeros(len(gold), dtype=np.float32))
    assert selected["repair_fraction"] == 1


def test_exit_uses_depth_and_can_fall_back_to_static():
    gold = np.array([1,2,3,0])
    data = dict(gold=gold, full_pred=gold.copy(), exit_pred=np.column_stack([np.zeros(4, dtype=int), gold]),
                confidence=np.ones((4,2), dtype=np.float32))
    selected, _ = calibrate_exit(data, [2,6])
    assert selected["kind"] == "full" and selected["threshold"] is None
    data["exit_pred"][:,0] = gold
    selected, _ = calibrate_exit(data, [2,6])
    assert selected["mean_depth"] == 2


def test_gate_split_deduplicates_and_removes_train_dev_overlap():
    pools, removed = split_pools(["hello world .", "hello world ?", "this is a test ."],
                                ["hello world .", "how are you ?", "yes today .", "no today ."])
    assert removed == 1
    keys = [set(pool) for pool in pools.values()]
    assert not keys[0] & keys[1] and not keys[0] & keys[2] and not keys[1] & keys[2]
    assert len(pools["fit"]) == 1


def test_policy_rejects_foreign_model():
    first = dict(model_sha256="one", architecture_sha256="arch", tokenizer_sha256="tok", window=64, lookahead=4, anchor_limit=128)
    second = {**first, "model_sha256": "two"}
    with pytest.raises(ValueError, match="model_sha256"):
        check_identity(first, second)
