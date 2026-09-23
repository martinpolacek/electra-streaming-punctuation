"""Input regressions: aligned labels must never be present in model input."""
import json
import pytest
from apr.data import (normalize_words, normalized_words_labels, read_records,
                      make_batch, load_finetuning_data, filter_lexical_blocks)
from apr.legacy_data import split_into_sentences, generate_dataset
from prepare_gate_data import split_pools


@pytest.mark.parametrize("text,words,labels", [
    ("hello,world?", ["hello", "world"], [3, 1]),
    ("hello ,world ?", ["hello", "world"], [3, 1]),
    ("3,5", ["3", "5"], [3, 0]),
    ("Hello! world", ["hello", "world"], [2, 0]),
    ("________ .", [], []),
])
def test_shared_lexical_normalization(text, words, labels, tokenizer):
    assert normalize_words(text) == words
    assert normalized_words_labels(text) == (words, labels)
    if words:
        _, _, gold, selected, _ = make_batch([text], tokenizer, "word_final")
        assert gold[selected].tolist() == labels
        pools, _ = split_pools([text], [])
        assert list(pools["fit"].values()) == [(words, labels)]


@pytest.mark.parametrize("word", ["hello?", "hello,world", ".", "____", "e-mail"])
def test_aligned_input_rejects_noncanonical_words(tmp_path, word):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"words": ["valid", word], "labels": [0, 1]}), encoding="utf-8")
    with pytest.raises(ValueError, match="record 1, word 2"):
        read_records(path)


def test_aligned_input_keeps_word_alignment(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"words": ["HELLO", "world"], "labels": [3, 1]}) + "\n" +
                    json.dumps({"words": [], "labels": [], "deleted_reference_labels": [2]}), encoding="utf-8")
    first, empty = read_records(path)
    assert first == {"words": ["hello", "world"], "labels": [3, 1]}
    assert empty["deleted_reference_labels"] == [2]


@pytest.mark.parametrize("text,expected", [
    ("hello world. trailing text", ["hello world.", "trailing text"]),
    ("hello world", ["hello world"]),
    ("hello number 2026.", ["hello number 2026."]),
    ("number 2026. another sentence?", ["number 2026.", "another sentence?"]),
    ("number 3.5 today! another sentence", ["number 3.5 today!", "another sentence"]),
])
def test_default_sentence_split_keeps_tail_and_numeric_end(text, expected):
    assert split_into_sentences(text, retain_tail=True) == expected


def test_legacy_sentence_parsing_is_explicitly_preserved():
    assert split_into_sentences("hello world. trailing text") == ["hello world."]
    assert split_into_sentences("hello number 2026.") == []
    pools, _ = split_pools(["hello ,world ?"], [], "original_subwords")
    assert list(pools["fit"].values()) == [(["hello", "world"], [0, 1])]


def test_default_dataset_filters_empty_sentences_and_preserves_legacy_split(tmp_path):
    lines = [f"hello entry{i}, this is a test." for i in range(100)] + ["________ ."]
    (tmp_path / "train.utf8").write_text("\n".join(lines), encoding="utf-8")
    train, dev, stats = load_finetuning_data(tmp_path)
    assert len(train) == 95 and len(dev) == 5
    assert stats == {"usable_sentences": 100, "discarded_nonlexical_sentences": 1}
    assert all(normalize_words(line) for line in train + dev)
    old_train, old_dev = generate_dataset(tmp_path, .05)
    for mode in ("original_subwords", "archived_word_final"):
        actual_train, actual_dev, _ = load_finetuning_data(tmp_path, mode)
        assert actual_train == old_train and actual_dev == old_dev


def test_empty_corpus_and_empty_blocks_have_actionable_errors(tmp_path):
    with pytest.raises(ValueError, match="No .utf8 files"):
        load_finetuning_data(tmp_path)
    (tmp_path / "train.utf8").write_text("________ .\n________ .", encoding="utf-8")
    with pytest.raises(ValueError, match="At least two usable sentences"):
        load_finetuning_data(tmp_path)
    assert filter_lexical_blocks(["________ .", "hello world ."], "training") == (["hello world ."], 1)
    with pytest.raises(ValueError, match="No lexical words remain in the validation blocks"):
        filter_lexical_blocks(["________ ."], "validation")
