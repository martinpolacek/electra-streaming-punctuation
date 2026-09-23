import copy
import numpy as np
import pytest
import torch
from apr.artifacts import save_bundle, load_bundle
from apr.cache import KVCache, Gate
from apr.data import windows, make_batch
from apr.model import Punctuator, target_logits
from apr.metrics import metrics
from apr.training import masked_ce


def test_real_exit_matches_all_heads_and_skips_layers(model):
    ids = torch.randint(5, model.config.vocab_size, (3, 15))
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[1, 10:] = False
    targets = torch.tensor([3, 8, 12])
    calls = [0] * 6
    handles = [layer.register_forward_hook(lambda _m, _i, _o, j=j: calls.__setitem__(j, calls[j]+1)) for j, layer in enumerate(model.backbone.encoder.layer)]
    with torch.inference_mode():
        all_logits = target_logits(model, ids, mask, targets, all_exits=True)
        full, depth = target_logits(model, ids, mask, targets)
        torch.testing.assert_close(full, all_logits[:, -1])
        for tau in [0.0, 0.25, 0.3, 0.5, 0.99, 1.0]:
            logits, depth = target_logits(model, ids, mask, targets, tau)
            passed = all_logits.softmax(-1).amax(-1) >= tau
            passed[:, -1] = True
            selected = passed.int().argmax(-1)
            torch.testing.assert_close(logits, all_logits[torch.arange(3), selected])
            assert depth.tolist() == [model.exits[i] for i in selected]
        calls[:] = [0] * 6
        target_logits(model, ids[:1], mask[:1], targets[:1], 0.0)
        assert calls == [1, 1, 0, 0, 0, 0]
    for handle in handles:
        handle.remove()


def test_cache_matches_fresh_when_refreshed_and_does_not_change_on_repair(model, tokenizer):
    words = ("hello playing world how are you today yes no ".split()) * 25
    cache = KVCache(model)
    seen_refresh = 0
    with torch.inference_mode():
        for item in windows(words, tokenizer):
            logits, features = cache.step(*item)
            ids, a, b, start, target, ends = item
            assert features.shape == (model.config.hidden_size+12,)
            assert 0 <= cache.last_stats["origin"] <= a
            assert b-cache.last_stats["origin"] <= 128
            if cache.last_stats["full_refresh"]:
                expected, _ = target_logits(model, ids, torch.ones_like(ids, dtype=torch.bool), torch.tensor([target-a]))
                torch.testing.assert_close(logits, expected[0])
                seen_refresh += 1
            before = [(k.clone(), v.clone()) for k, v in cache.cache]
            target_logits(model, ids, torch.ones_like(ids, dtype=torch.bool), torch.tensor([target-a]))
            for old, new in zip(before, cache.cache):
                assert all(torch.equal(a, b) for a, b in zip(old, new))
        assert seen_refresh >= 3
        assert cache.last_stats["active_tokens"] < 64
        cache.reset()
        assert cache.bounds is None and cache.cache is None


def test_window_targets_four_future_words_and_short_tail(tokenizer):
    words = ["playing", "hello", "world", "yes", "no", "today", "test"]
    items = list(windows(words, tokenizer))
    assert len(items) == len(words)
    assert [len(item[-1]) for item in items] == [5,5,5,4,3,2,1]
    for ids, a, b, start, target, ends in items:
        assert ids.shape == (1,b-a)
        assert int(ends[0]) == target and int(ends[-1]) == b-1
    assert list(windows([], tokenizer)) == []
    with pytest.raises(ValueError, match="exceeds"):
        list(windows(["playing"]*6, tokenizer, window=2))


def test_loss_masks_only_select_completed_word_ends(tokenizer):
    text = ["hello playing , world ?"]
    batch = make_batch(text, tokenizer, "word_final")
    ids, mask, labels, supervised, ends = batch
    selected = tokenizer.convert_ids_to_tokens(ids[0, ends[0]].tolist())
    assert selected == ["hello", "##ing", "world"]
    assert labels[ends].tolist() == [0,3,1]
    logits = torch.randn(*ids.shape, 4, requires_grad=True)
    masked_ce(logits, labels, supervised, torch.ones(4)).backward()
    assert not logits.grad[~ends].any()
    assert logits.grad[ends].abs().sum() > 0
    old = make_batch(text, tokenizer, "original_subwords")
    ablation = make_batch(text, tokenizer, "archived_word_final")
    for i in [0,1,2]:
        assert torch.equal(old[i], ablation[i])
    assert old[3].sum() > ablation[3].sum()
    boundary = " ".join(["hello"]*511 + ["playing", "?"])
    ids, mask, labels, supervised, _ = make_batch([boundary], tokenizer, "word_final")
    assert ids.shape[1] == 512 and supervised[0, -1].item() is False


def test_exit_training_keeps_base_and_final_head_fixed(model):
    base = Punctuator(model.config)
    original = {k: v.clone() for k, v in base.state_dict().items()}
    base.add_exits()
    optimizer = torch.optim.AdamW([p for p in base.parameters() if p.requires_grad], lr=1e-3)
    ids = torch.randint(5, model.config.vocab_size, (2, 12))
    outputs = base(ids, torch.ones_like(ids, dtype=torch.bool))
    sum(x.square().mean() for x in outputs[:-1]).backward()
    optimizer.step()
    for name, value in original.items():
        assert torch.equal(base.state_dict()[name], value), name
    assert any(p.grad is not None for p in base.heads["2"].parameters())


def test_bundle_roundtrip_and_integrity(tmp_path, model, tokenizer):
    path = tmp_path / "model"
    save_bundle(path, model, tokenizer)
    loaded, _, _ = load_bundle(path, stage="punctuation")
    for k, v in model.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k])
    with pytest.raises(FileExistsError):
        save_bundle(path, model, tokenizer)
    with (path / "model.safetensors").open("ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="integrity"):
        load_bundle(path)


def test_metrics_support_weighting_and_asr_deletions():
    q = metrics([1,2,2,3,0], [1,2,0,3,3], [1])
    assert q["confusion"][1][0] == 1
    assert q["question_f1"] == pytest.approx(200/3)
    assert q["weighted_f1"] == pytest.approx(200/3)
    assert metrics([0], [0])["weighted_f1"] == 0


def test_paper_gate_parameter_count():
    assert sum(p.numel() for p in Gate(268).parameters()) == 8641


def test_word_final_handles_attached_punctuation(tokenizer):
    for text in ["hello,world?", "hello ,world ?", "hello , world ?"]:
        _, _, labels, supervised, _ = make_batch([text], tokenizer, "word_final")
        assert labels[supervised].tolist() == [3,1]


def test_bundle_rejects_architecture_and_tokenizer_wrapper_changes(tmp_path, model, tokenizer):
    import json
    path = tmp_path / "model"
    save_bundle(path, model, tokenizer)
    config_path = path / "config.json"
    original = config_path.read_text(encoding="utf-8")
    doc = json.loads(original)
    doc["architecture"]["num_heads"] = 2
    config_path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        load_bundle(path)
    config_path.write_text(original, encoding="utf-8")
    config = path / "tokenizer/tokenizer_config.json"
    config.write_text(config.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        load_bundle(path)


def test_cache_reuses_exact_prefix_with_eviction(model, tokenizer):
    words = "hello playing world how are you today yes no".split() * 25
    cache = KVCache(model)
    reused, evicted = 0, 0
    for item in windows(words, tokenizer):
        previous_bounds = cache.bounds
        previous = None if cache.cache is None else [(k.clone(), v.clone()) for k, v in cache.cache]
        cache.step(*item)
        _, a, _, active, _, _ = item
        if not cache.last_stats["full_refresh"]:
            pa, _ = previous_bounds
            prefix = active - a
            for old, new in zip(previous, cache.cache):
                for before, after in zip(old, new):
                    assert torch.equal(before[:, :, a-pa:active-pa], after[:, :, :prefix])
            reused += 1
            evicted += int(a > pa)
    assert reused > 100 and evicted > 100


def test_validation_reports_each_exit_head(model, tokenizer):
    from apr.training import validate
    with torch.no_grad():
        for depth, head in model.heads.items():
            head[-1].weight.zero_()
            head[-1].bias.zero_()
            head[-1].bias[1 if depth == "2" else 0] = 10
    result = validate(model, ["hello ? world ?"], tokenizer, "cpu", loss_mode="word_final")
    assert set(result) == {"2", "3", "4", "5", "6"}
    assert result["2"]["question_f1"] == 100
    assert result["6"]["question_f1"] == 0
    assert result["2"]["punctuation_support"] == result["6"]["punctuation_support"] == 2
