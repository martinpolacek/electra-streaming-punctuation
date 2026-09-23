# Data and checkpoints

## Training modes

`--loss-mode` selects tokenization and loss positions:

| Mode | Tokenization | Loss positions |
|---|---|---|
| `word_final` (default) | Remove punctuation before tokenization | Completed word ends |
| `original_subwords` | Tokenize punctuated text, then remove punctuation tokens | All retained nonpadding subwords |
| `archived_word_final` | Same as `original_subwords` | Completed word ends |

Use the same mode for fine-tuning, exit-head training and `prepare_gate_data.py`.
It is stored in model metadata and generated gate streams. The two archived modes
retain the original sentence parser; `word_final` keeps final text fragments and
filters nonlexical inputs.

## Hugging Face import

`import_model.py --hf` reads the architecture and tokenizer from the same Hub
repository or local Transformers directory. Pin `--revision` to a commit for
repeatable downloads. Metadata stores the requested revision and resolved Hub
commit; the tokenizer is loaded at that commit. Local directories have no Hub revision.

`--family`, `--layers` and `--tokenizer` apply to `--legacy` import.

## Original checkpoints

```bash
python import_model.py --legacy discriminator_1000000.pth --kind encoder --family Small --layers 6 --out runs/imported_encoder
python import_model.py --legacy epoch4_final.data --kind punctuation --loss-mode original_subwords --family Small --layers 6 --out runs/imported_punctuation
python import_model.py --legacy epoch2_final.data --kind exit --loss-mode original_subwords --family Small --layers 6 --out runs/imported_exit
```

For punctuation and exit checkpoints, supply the loss mode used during training.
Use `--tokenizer` for a local tokenizer. The archived exit importer supports Small-L6.
Old filenames use `XXS` for the family now called `Mini`.

Import checks tensor names and shapes. It drops redundant single-expert router
weights and the position-ID buffer. Mini keeps its learned 128-to-128 embedding
projection; importing a stock HF model with equal widths inserts an identity projection.

## Saved files and resume

A model directory contains `model.safetensors`, `config.json` and `tokenizer/`.
Loading verifies hashes of weights, architecture and tokenizer files. Gates and
policies store the same model identifiers. Refit the gate and recalibrate after
changing its punctuation checkpoint.

Pretraining also saves `training.pt`: generator, discriminator, optimizer,
scheduler, random-number state and sampler state. Resume with the same corpus
and configuration into a new output directory:

```bash
python pretrain.py --data-dir data/pretraining --family Small --layers 6 --out runs/resumed --resume runs/pretrain_small_l6/training.pt --precision bf16
```

The sampler saves its queued segments and sampling state. PyTorch checkpoints
are loaded with `weights_only=True`. Existing nonempty output directories are rejected.

## ASR evaluation

The [test transcript download](https://owncloud.cesnet.cz/index.php/s/q0TbViTq8GG7sLI)
contains reference and ASR text. `evaluate.py` requires word-aligned JSONL, as
shown in the [README](../README.md#evaluate).

The paper aligns reference and recognized words within annotated segments,
assigns NONE to insertions and counts punctuation on deleted words as missed.
The download does not contain all segment-alignment metadata, so whole-recording
alignment can give different scores.

CPU timings in the paper use four pinned physical AMD EPYC 9354 cores, float32
and PyTorch 2.13.0+cu130. The test environment is recorded in
[VALIDATION.md](VALIDATION.md).
