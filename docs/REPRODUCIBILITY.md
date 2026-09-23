# Reproducibility and checkpoint compatibility

## What this release provides

The scripts provide word-final fine-tuning and early-exit/KV-reuse inference. They can train new models
and import archived checkpoints. A new run is not a bit-identical regeneration
of an earlier experiment: original corpus files, model checkpoints, tokenizer,
training configuration and evaluation protocol determine the result. Training
corpora and fine-tuned task weights are not included.

The public Hugging Face Small-L6 model is the one-million-update **pretrained**
encoder, without punctuation fine-tuning, exit heads or a gate. It must be
fine-tuned before the inference commands are useful.

## Training modes

`finetune.py` and `train_exit.py` store the selected mode in every saved bundle:

| Mode | Input tokenization | Direct loss positions |
|---|---|---|
| `word_final` (default) | Remove punctuation before tokenization | Completed word ends |
| `original_subwords` | Tokenize punctuated text, then remove punctuation tokens | All nonpadding retained subwords |
| `archived_word_final` | Same token IDs/labels as `original_subwords` | Completed word ends only |

Use the same training mode when fitting exit heads for an existing model.
Fit new gates and exit heads after changing the fine-tuned encoder.

## Original checkpoints

A tensor-only original pretrained/fine-tuned checkpoint can be imported without
the original server paths or training package:

```bash
python import_model.py --legacy discriminator_1000000.pth --kind encoder --family Small --layers 6 --out runs/imported_encoder
python import_model.py --legacy epoch4_final.data --kind punctuation --loss-mode original_subwords --family Small --layers 6 --out runs/imported_punctuation
python import_model.py --legacy epoch2_final.data --kind exit --loss-mode original_subwords --family Small --layers 6 --out runs/imported_exit
```

Supply `--tokenizer` when using a local copy. Legacy task checkpoint import
requires an explicit `--loss-mode` because that provenance cannot be inferred
from tensor values. Select the mode matching the checkpoint's training
configuration from the table above. `XXS` in the old filenames maps to
`Mini`. The importer strictly checks tensor shapes/keys. It discards redundant
single-expert router parameters and the deterministic position-ID buffer; the
dense forward computation remains the same. Mini retains its learned embedding
projection even though embedding and hidden widths are both128. A stock HF
ELECTRA with equal widths otherwise omits this layer.

## Deliberate engineering changes

The release removes hardcoded cluster paths, remote experiment tracking and
asynchronous input workers. Pretraining uses a deterministic bounded-memory
sampler, with RNG and queued segments saved for resume. It randomizes loaded
segments before taking the bounded subset, so the tail of a small file remains
eligible. These changes affect sample/RNG order; full retraining is not claimed
to reproduce the old weight hashes. Mathematical objectives and configuration
values are documented in [METHOD.md](METHOD.md).

Model outputs are safetensors bundles with strict weight, architecture and
complete-tokenizer hashes. Gate/policy files are bound to those identities.
Original PyTorch checkpoints and pretraining resume files are loaded with
`weights_only=True`. Output directories are never silently overwritten.

## Evaluation boundaries

The provided `evaluate.py` takes already aligned labels. News/Talks evaluation
in the paper aligns recognized/reference words within annotated segments,
assigns NONE to inserted words and counts punctuation on deleted words as missed.
The plain transcript download alone does not encode all original segment
alignment metadata. Do not interpret whole-recording realignment as an exact
reproduction of those ASR scores.

CPU timings in the article used four pinned physical AMD EPYC9354 cores,
float32 inference and PyTorch2.13.0+cu130. This release's tests run with the
versions pinned in `requirements.txt`; it does not assert that inference times
on another environment equal the published measurements. It includes functional
inference, not a replacement for the article's controlled timing protocol.

The supplied tests verify algorithmic behavior and short training runs. They do
not repeat the one-million-step pretraining or three complete fine-tuning runs.
