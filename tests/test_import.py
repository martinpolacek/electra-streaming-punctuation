"""Offline import contracts and numerical parity for native and legacy models."""
import sys
import pytest
import torch
from transformers import ElectraConfig, ElectraModel
import import_model
from apr.artifacts import load_bundle
from apr.config import Config
from apr.model import Punctuator


@pytest.mark.parametrize("embedding_size", [8, 12])
def test_hf_import_preserves_hidden_states(tmp_path, tokenizer, monkeypatch, embedding_size):
    config = ElectraConfig(vocab_size=len(tokenizer), embedding_size=embedding_size,
                           hidden_size=12, num_hidden_layers=2, num_attention_heads=3,
                           intermediate_size=24, hidden_dropout_prob=.1,
                           attention_probs_dropout_prob=.1)
    original = ElectraModel(config).eval()
    source, destination = tmp_path / "hf", tmp_path / "bundle"
    original.save_pretrained(source)
    tokenizer.save_pretrained(source)
    monkeypatch.setattr(sys, "argv", ["import_model.py", "--hf", str(source), "--out", str(destination)])
    import_model.main()
    actual, _, doc = load_bundle(destination, stage="encoder")
    for length in (1, 7, 32):
        ids = torch.randint(5, len(tokenizer), (2, length))
        mask = torch.ones_like(ids, dtype=torch.bool)
        if length > 1:
            ids[1, -2:] = tokenizer.pad_token_id
            mask[1, -2:] = False
        with torch.inference_mode():
            expected = original(ids, attention_mask=mask).last_hidden_state
            torch.testing.assert_close(actual(ids, mask), expected, atol=1e-5, rtol=1e-5)
    assert doc["metadata"]["revision"] is None
    if embedding_size == config.hidden_size:
        assert torch.equal(actual.embeddings_project.weight, torch.eye(12))
        assert not actual.embeddings_project.bias.any()


@pytest.mark.parametrize("args", [
    ["--hf", "unused", "--family", "Small"],
    ["--hf", "unused", "--layers", "6"],
    ["--hf", "unused", "--tokenizer", "unused"],
    ["--legacy", "unused", "--revision", "main"],
])
def test_import_rejects_ignored_options_before_loading(args, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["import_model.py", *args, "--out", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as caught:
        import_model.main()
    assert caught.value.code == 2
    assert "appl" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_hf_import_records_resolved_commit_and_pins_tokenizer(tmp_path, tokenizer, monkeypatch):
    original = ElectraModel(ElectraConfig(vocab_size=len(tokenizer), embedding_size=8,
                           hidden_size=12, num_hidden_layers=1, num_attention_heads=3,
                           intermediate_size=24)).eval()
    revision = "a" * 40
    original.config._commit_hash = revision
    monkeypatch.setattr(import_model.AutoModel, "from_pretrained", lambda *a, **kw: original)
    seen = []
    def tokenizer_from(path, rev):
        seen.append((path, rev))
        return tokenizer
    monkeypatch.setattr(import_model, "tokenizer_from", tokenizer_from)
    monkeypatch.setattr(sys, "argv", ["import_model.py", "--hf", "test/encoder", "--revision", "main",
                                    "--out", str(tmp_path / "out")])
    import_model.main()
    _, _, doc = load_bundle(tmp_path / "out")
    assert doc["metadata"]["revision"] == revision
    assert doc["metadata"]["requested_revision"] == "main"
    assert seen == [("test/encoder", revision)]


def test_legacy_exit_import_maps_every_head(tmp_path, tokenizer, monkeypatch):
    original = Punctuator(Config.family("Small", 6, vocab_size=len(tokenizer)), [2, 3, 4, 5, 6]).eval()
    state = {}
    for key, value in original.state_dict().items():
        if key.startswith("backbone."):
            key = "bert_layer." + key.removeprefix("backbone.")
            key = key.replace(".ffn.dense_up.", ".moe.experts.0.dense_up.").replace(".ffn.dense_down.", ".moe.experts.0.dense_down.").replace(".ffn.LayerNorm.", ".moe.LayerNorm.")
        else:
            parts = key.split(".")
            parts[1] = str(original.exits.index(int(parts[1])))
            key = ".".join(parts)
        state["module." + key] = value
    source = tmp_path / "exit.data"
    torch.save(state, source)
    tokenizer.save_pretrained(tmp_path / "tok")
    monkeypatch.setattr(sys, "argv", ["import_model.py", "--legacy", str(source), "--kind", "exit",
                                    "--loss-mode", "original_subwords", "--tokenizer", str(tmp_path / "tok"),
                                    "--out", str(tmp_path / "out")])
    import_model.main()
    actual, _, doc = load_bundle(tmp_path / "out")
    assert actual.exits == original.exits
    assert doc["metadata"]["loss_mode"] == "original_subwords"
    ids = torch.tensor([[8, 9, 10, 11]])
    with torch.inference_mode():
        expected = original(ids, torch.ones_like(ids, dtype=torch.bool))
        imported = actual(ids, torch.ones_like(ids, dtype=torch.bool))
    for before, after in zip(expected, imported):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
