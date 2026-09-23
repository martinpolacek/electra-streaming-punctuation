"""Offline end-to-end tests use synthetic text and randomly initialized tiny models."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from apr.artifacts import save_bundle, load_bundle, write_json, sha256
from apr.model import Encoder

ROOT = Path(__file__).resolve().parents[1]

def run(*args):
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-B", str(ROOT / args[0]), *map(str, args[1:])], cwd=ROOT,
                            env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_training_exit_calibration_and_inference_cli(tmp_path, model, tokenizer):
    encoder_path = tmp_path / "encoder"
    save_bundle(encoder_path, model.backbone, tokenizer)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    lines = [f"hello entry{i}, this is a test today" + ("?" if i % 2 else ".") for i in range(1000)]
    (corpus / "train.utf8").write_text("\n".join(lines), encoding="utf-8")
    common = ["--data-dir", corpus, "--device", "cpu", "--threads", "1", "--epochs", "1", "--max-steps", "2", "--batch-size", "2", "--max-validation-blocks", "2"]
    ft = tmp_path / "finetune"
    run("finetune.py", "--encoder", encoder_path, "--out", ft, *common)
    full, _, full_doc = load_bundle(ft / "model")
    assert full_doc["metadata"]["loss_mode"] == "word_final"
    assert not full_doc["metadata"]["completed_training"]
    ex = tmp_path / "exit"
    run("train_exit.py", "--model", ft / "model", "--out", ex, *common)
    exited, _, exit_doc = load_bundle(ex / "model")
    assert exited.exits == (2,3,4,5,6)
    for key, value in full.state_dict().items():
        assert torch.equal(exited.state_dict()[key], value)
    for mode in ["full", "exit", "fast"]:
        output = run("infer.py", "--model", ex / "model", "--text", "hello playing world how are you today", "--mode", mode, "--format", "jsonl")
        rows = [json.loads(line) for line in output.splitlines()]
        assert len(rows) == 7
        assert all(row["label"] in range(4) for row in rows)
    splits = tmp_path / "splits"
    run("prepare_gate_data.py", "--data-dir", corpus, "--out", splits, "--fit-streams", "1", "--calibration-streams", "1", "--validation-streams", "1", "--words-per-stream", "32")
    features_path = tmp_path / "real_features.npz"
    run("collect_features.py", "--model", ex / "model", "--data", splits / "fit.jsonl", "--out", features_path, "--split", "fit", "--threads", "1")
    with np.load(features_path, allow_pickle=False) as f:
        assert f["features"].shape == (32, model.config.hidden_size+12)
        assert f["confidence"].shape == (32, 5)
    # Synthetic helpful-repair examples exercise gate fitting even when the
    # untrained smoke model happens to have no disagreements on the tiny corpus.
    rng = np.random.default_rng(42)
    base = {"model_sha256": exit_doc["weights_sha256"], "architecture_sha256": exit_doc["architecture_sha256"],
            "tokenizer_sha256": exit_doc["tokenizer_sha256"], "window": 64, "lookahead": 4,
            "anchor_limit": 128, "exits": [2,3,4,5,6]}
    for split in ("fit", "calibrate"):
        gold = np.tile(np.arange(4), 20)
        full_pred = gold.copy()
        fast_pred = gold.copy()
        fast_pred[::3] = (fast_pred[::3] + 1) % 4
        x = rng.normal(size=(80, model.config.hidden_size+12)).astype(np.float32)
        x[:, :4] = 0.1
        x[np.arange(80), fast_pred] = 0.7
        exit_pred = np.tile(full_pred[:, None], (1,5))
        metadata = {**base, "split": split, "sentence_hashes": [split+"_disjoint"], "data_sha256": split}
        np.savez_compressed(tmp_path / (split+".npz"), features=x, gold=gold, full_pred=full_pred,
                            fast_pred=fast_pred, confidence=np.full((80,5), 0.99, dtype=np.float32), exit_pred=exit_pred,
                            metadata=np.asarray(json.dumps(metadata)))
    gate = tmp_path / "gate"
    run("train_gate.py", "--features", tmp_path / "fit.npz", "--out", gate, "--epochs", "1", "--threads", "1")
    policy = tmp_path / "policies.json"
    run("calibrate.py", "--features", tmp_path / "calibrate.npz", "--gate", gate, "--out", policy)
    for mode in ("gate", "margin", "entropy", "exit"):
        args = ["infer.py", "--model", ex / "model", "--text", "hello world today", "--mode", mode, "--policy", policy]
        if mode == "gate":
            args += ["--gate", gate]
        assert run(*args).strip()
    output = run("evaluate.py", "--model", ex / "model", "--data", splits / "validate.jsonl")
    assert json.loads(output)["words"] == 32


def test_pretraining_cli_resumes_exactly(tmp_path, tokenizer):
    tok_dir = tmp_path / "tok"
    tokenizer.save_pretrained(tok_dir)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "text.utf8").write_text("hello world playing this is a test " * 100, encoding="utf-8")
    args = ["pretrain.py", "--data-dir", corpus, "--tokenizer", tok_dir, "--family", "Tiny", "--layers", "1",
            "--steps", "2", "--warmup", "0", "--batch-size", "1", "--device", "cpu", "--threads", "1"]
    run(*args, "--out", tmp_path / "whole")
    run(*args, "--out", tmp_path / "first", "--stop-after", "1")
    run(*args, "--out", tmp_path / "resumed", "--resume", tmp_path / "first/training.pt")
    whole, _, _ = load_bundle(tmp_path / "whole/encoder")
    resumed, _, _ = load_bundle(tmp_path / "resumed/encoder")
    for key, value in whole.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key]), key


def test_legacy_import_preserves_explicit_training_provenance(tmp_path, model, tokenizer, monkeypatch):
    import import_model
    from apr.model import Punctuator
    original = Punctuator(model.config)
    # Use a paper geometry here, so the public importer can reconstruct it.
    from apr.config import Config
    original = Punctuator(Config.family("Tiny", 4, vocab_size=len(tokenizer)))
    legacy = {}
    for key, value in original.state_dict().items():
        if key.startswith("backbone."):
            key = "bert_layer." + key.removeprefix("backbone.")
            key = key.replace(".ffn.dense_up.", ".moe.experts.0.dense_up.").replace(".ffn.dense_down.", ".moe.experts.0.dense_down.").replace(".ffn.LayerNorm.", ".moe.LayerNorm.")
        else:
            key = key.replace("heads.4.0.", "lin.").replace("heads.4.2.", "lin2.")
        legacy[key] = value
    checkpoint = tmp_path / "legacy.data"
    torch.save(legacy, checkpoint)
    tokenizer.save_pretrained(tmp_path / "tok")
    common = ["import_model.py", "--legacy", str(checkpoint), "--kind", "punctuation", "--family", "Tiny", "--layers", "4", "--tokenizer", str(tmp_path / "tok"), "--out", str(tmp_path / "imported")]
    import pytest
    monkeypatch.setattr(sys, "argv", common)
    with pytest.raises(SystemExit):
        import_model.main()
    monkeypatch.setattr(sys, "argv", common + ["--loss-mode", "archived_word_final"])
    import_model.main()
    actual, _, document = load_bundle(tmp_path / "imported")
    assert document["metadata"]["loss_mode"] == "archived_word_final"
    for key, value in original.state_dict().items():
        assert torch.equal(value, actual.state_dict()[key])


def test_evaluation_retains_empty_asr_deletions(tmp_path, model, tokenizer, monkeypatch, capsys):
    import evaluate
    path = tmp_path / "model"
    save_bundle(path, model, tokenizer)
    data = tmp_path / "empty_asr.jsonl"
    data.write_text(json.dumps({"words": [], "labels": [], "deleted_reference_labels": [1,2,3]}) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--model", str(path), "--data", str(data)])
    evaluate.main()
    result = json.loads(capsys.readouterr().out)
    assert result["words"] == 0 and result["punctuation_support"] == 3
    assert result["weighted_f1"] == 0
    assert result["mean_depth"] is None
