"""Original punctuation-first tokenization plus explicit loss-position selection."""
import json
import re
from pathlib import Path
import torch

def _process_batch(texts, tokenizer, dot_id, comma_id, quest_id, cls_id, sep_id, pad_id):
    """Process a single batch: tokenize + extract punct labels. Runs in worker process."""
    texts_compact = [x.replace(' ,', ',').replace(' .', '.').replace(' ?', '?') for x in texts]
    encoded = tokenizer(
        texts_compact, return_tensors='pt', padding=True, truncation=True,
        max_length=512, add_special_tokens=False)
    all_ids = encoded['input_ids']
    all_mask = encoded['attention_mask'].bool()
    B, L = all_ids.shape

    out_ids = torch.full((B, L), pad_id, dtype=torch.long)
    out_mask = torch.zeros((B, L), dtype=torch.bool)
    output = torch.zeros((B, L), dtype=torch.long)

    punctuation = {quest_id: 1, dot_id: 2, comma_id: 3}
    for idx in range(B):
        kept_ids = []
        kept_labels = []
        for token_id in all_ids[idx][all_mask[idx]].tolist():
            label = punctuation.get(token_id)
            if label is not None:
                if kept_labels:
                    kept_labels[-1] = label
                continue
            if token_id in (cls_id, sep_id, pad_id):
                continue
            kept_ids.append(token_id)
            kept_labels.append(0)

        length = len(kept_ids)
        if length:
            out_ids[idx, :length] = torch.tensor(kept_ids, dtype=torch.long)
            output[idx, :length] = torch.tensor(kept_labels, dtype=torch.long)
            out_mask[idx, :length] = True

    output = torch.nn.functional.one_hot(output, 4)
    return [out_ids, out_mask], output


def word_end_mask(texts, tokenizer, input_ids, attention_mask,
                  punctuation_ids, excluded_ids, max_length=512):
    compact = [x.replace(' ,', ',').replace(' .', '.').replace(' ?', '?') for x in texts]
    full = tokenizer(compact, add_special_tokens=False, truncation=False,
                     return_attention_mask=False, verbose=False)
    # A standalone SentencePiece whitespace marker is not a lexical word end.
    whitespace_id = tokenizer.get_vocab().get('\u2581')
    ends = torch.zeros_like(attention_mask, dtype=torch.bool)
    for row, all_ids in enumerate(full['input_ids']):
        word_ids = full.word_ids(batch_index=row)
        if len(word_ids) != len(all_ids):
            raise ValueError('Tokenizer word alignment length mismatch')
        last_raw, last_lexical = {}, {}
        for pos, (token, word) in enumerate(zip(all_ids, word_ids)):
            if word is None:
                continue
            last_raw[word] = pos
            if token not in punctuation_ids:
                last_lexical[word] = pos
        kept = []
        selected = []
        for pos, (token, word) in enumerate(zip(all_ids[:max_length], word_ids[:max_length])):
            if token in punctuation_ids or token in excluded_ids:
                continue
            kept.append(token)
            selected.append(token != whitespace_id and word is not None
                            and last_raw[word] < max_length
                            and last_lexical.get(word) == pos)
        actual = input_ids[row][attention_mask[row]].tolist()
        if kept != actual:
            raise ValueError('Word mask would change the original lexical input')
        if kept:
            ends[row, :len(kept)] = torch.tensor(selected, dtype=torch.bool)
    if torch.any(ends & ~attention_mask.bool()):
        raise AssertionError('Padding selected for word-final loss')
    return ends


def make_batch(texts, tokenizer, loss_mode, max_length=512):
    if loss_mode not in ("original_subwords", "word_final", "archived_word_final"):
        raise ValueError("Choose original_subwords, word_final or archived_word_final explicitly")
    if max_length != 512:
        raise ValueError("Archived fine-tuning truncates the punctuated input at 512 subwords")
    if loss_mode == "word_final":
        return lexical_batch(texts, tokenizer, max_length)
    punctuation = [tokenizer.encode(x, add_special_tokens=False)[-1] for x in (".", ",", "?")]
    if len(set(punctuation)) != 3 or any(p in tokenizer.all_special_ids for p in punctuation):
        raise ValueError("Tokenizer must have distinct punctuation tokens")
    cls = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else -1
    sep = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else -1
    inputs, labels = _process_batch(texts, tokenizer, *punctuation, cls, sep, tokenizer.pad_token_id)
    ids, mask = inputs
    ends = word_end_mask(texts, tokenizer, ids, mask, set(punctuation), {cls, sep, tokenizer.pad_token_id})
    supervision = mask if loss_mode == "original_subwords" else ends
    if not supervision.any():
        raise ValueError("Batch has no supervised positions")
    return ids, mask, labels.argmax(-1), supervision, ends

def words_labels(text):
    words, labels = [], []
    for item in text.split():
        if item in (".", ",", "?"):
            if labels:
                labels[-1] = {"?": 1, ".": 2, ",": 3}[item]
        else:
            word = "".join(re.findall(r"\w+", item.lower()))
            if word and any(c.isalnum() for c in word):
                words.append(word)
                labels.append(0)
    return words, labels

def read_records(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            words, labels = row.get("words"), row.get("labels")
            if not isinstance(words, list) or not all(isinstance(w, str) and w and not any(c.isspace() for c in w) for w in words):
                raise ValueError(f"Invalid words in record {number}")
            if not isinstance(labels, list) or len(labels) != len(words) or any(type(v) is not int or v not in range(4) for v in labels):
                raise ValueError(f"Invalid labels in record {number}")
            deleted = row.get("deleted_reference_labels", [])
            if not isinstance(deleted, list) or any(type(v) is not int or v not in range(4) for v in deleted):
                raise ValueError(f"Invalid deleted reference labels in record {number}")
            row["words"] = [w.lower() for w in words]
            rows.append(row)
    if not rows:
        raise ValueError("No labeled records")
    return rows

def write_records(path, rows):
    with open(path, "x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def windows(words, tokenizer, window=64, lookahead=4, device="cpu"):
    if not 1 <= window <= 128 or lookahead != 4:
        raise ValueError("Use a window in [1,128] and four-word look-ahead")
    if not words:
        return
    encoding = tokenizer([w.lower() for w in words], is_split_into_words=True, add_special_tokens=False, truncation=False, verbose=False)
    last = {word: index for index, word in enumerate(encoding.word_ids()) if word is not None}
    if set(last) != set(range(len(words))):
        raise ValueError("Every word must produce at least one subword")
    tokens = torch.tensor(encoding["input_ids"], dtype=torch.long, device=device)
    for i in range(len(words)):
        visible = list(range(i, min(i + lookahead + 1, len(words))))
        target, b = last[i], last[visible[-1]] + 1
        a = max(0, b - window)
        if target < a:
            raise ValueError(f"Four-word right context exceeds the window at word {i}; shorten the input words or increase the window")
        start = max(a, 0 if i == 0 else last[i-1] + 1)
        yield tokens[a:b][None, :], a, b, start, target, torch.tensor([last[j] for j in visible], device=device)


def lexical_batch(texts, tokenizer, max_length=512):
    """Current manuscript: remove punctuation first, supervise completed word ends."""
    parsed = [words_labels(re.sub(r"([.,?!])", r" \1 ", text).replace("!", ".")) for text in texts]
    if any(not words for words, _ in parsed):
        raise ValueError("A fine-tuning block contains no lexical words")
    sequences = [words for words, _ in parsed]
    full = tokenizer(sequences, is_split_into_words=True, add_special_tokens=False, truncation=False, verbose=False)
    encoded = tokenizer(sequences, is_split_into_words=True, add_special_tokens=False,
                        truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    ids = encoded["input_ids"]
    mask = encoded["attention_mask"].bool()
    labels = torch.zeros_like(ids)
    ends = torch.zeros_like(mask)
    for row, (words, gold) in enumerate(parsed):
        all_word_ids = full.word_ids(batch_index=row)
        last = {word: position for position, word in enumerate(all_word_ids) if word is not None}
        if set(last) != set(range(len(words))):
            raise ValueError("Every training word must produce a subword")
        for word, position in last.items():
            if position < max_length:
                labels[row, position] = gold[word]
                ends[row, position] = True
    if not ends.any():
        raise ValueError("Batch has no completed word-final targets")
    return ids, mask, labels, ends, ends
