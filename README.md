# Efficient Streaming Punctuation Restoration via Selective KV Reuse

Training and inference code for compact Czech ELECTRA punctuation models.
By default, punctuation is removed before tokenization and cross-entropy is
applied only to complete word-final subwords. Predictions use four labels: `NONE`, `QUESTION`, `PERIOD`,
and `COMMA` (IDs 0-3).

Included:

- ELECTRA replaced-token-detection pretraining with configurable family and depth;
- punctuation fine-tuning, with **word-final loss by default**;
- optional early-exit heads with the encoder and original final head frozen;
- selective KV reuse, helpful-repair gate training and matched calibration;
- plain-text inference and aligned-label evaluation.

See [the method settings](docs/METHOD.md) and
[the reproducibility notes](docs/REPRODUCIBILITY.md) for training modes,
checkpoint formats and the evaluation protocol.

## Install

Python 3.11-3.13. The release is tested with Python 3.13 and the versions below.
Use a virtual environment; commands are run from this directory.

```bash
python -m pip install -r requirements.txt
```

For CUDA, install the PyTorch 2.8.0 build for your machine first, then install
`requirements.txt`. CPU inference is the default. Training selects CUDA when
available, or accepts `--device cpu`. There is no experiment-tracking service,
private server dependency or remote-code execution requirement.

## Quick smoke test

Run these commands after installation to check import, fine-tuning and inference:

```bash
python import_model.py --hf AILabTUL/electra-small-l6-czech --revision b0bb960a057dc7891a861a0a830220529eb88b5a --out runs/smoke_encoder
python finetune.py --encoder runs/smoke_encoder --data-dir examples --out runs/smoke_l6 --device cpu --threads 1 --epochs 1 --max-steps 2
python infer.py --model runs/smoke_l6/model --text "povazujete to za problem" --mode full
```

The import downloads the public encoder and tokenizer; the next two commands run
locally. The example corpus only checks that the pipeline runs. Its model is not
trained enough to produce useful punctuation. Use fresh output directories when
repeating the commands. Follow the steps below for a complete training run.

## 1. Start with the released pretrained encoder

```bash
python import_model.py --hf AILabTUL/electra-small-l6-czech --revision b0bb960a057dc7891a861a0a830220529eb88b5a --out runs/pretrained
```

This Hugging Face model is the **pretrained encoder only**. It cannot restore
punctuation until a classification head has been trained.

Prepare UTF-8 text files with the `.utf8` extension in `data/train/`, containing
reference transcripts with punctuation. Exclude all test recordings before
creating this directory. Sentence extraction and the fixed 5% validation split
are performed by the script; train and validation blocks contain 1-15 sentences.
The default preprocessing keeps an unfinished final sentence and sentences
ending in a number. It filters sentences and constructed blocks without lexical
words; counts are saved in `run.json` and `training_history.json`. If too little
usable text remains for training and validation, the script stops with an error.
The 320 News and 64 Talks/interviews test recordings must not be used for training.
An example of the input format is in [examples/train.utf8](examples/train.utf8).

```bash
python finetune.py --encoder runs/pretrained --data-dir data/train --out runs/l6 --seed 42
```

The final bundle is `runs/l6/model/`. Repeat with seeds **13 and 100**, using
separate output directories, to obtain the three fine-tuning runs. Tiny example
data demonstrate the format; they cannot reproduce the paper's quality scores.

## 2. Run full-model inference

```bash
python infer.py --model runs/l6/model --text "povazujete to za problem" --mode full
python infer.py --model runs/l6/model --input recording.txt --mode full --format jsonl
```

Text uses the same lexical normalization as word-final training: it is lowercased,
`.,?!` separate words, and other non-word characters are removed within each word.
For example, `hello,world?` becomes two input words, `hello` and `world`.
One input file is one continuous recording. The model receives at most 64 subwords per decision and up
to four following words; the final four words use only available right context.
Use `--window 32` or `--window 128` for the other full/exit windows in the paper.
A word whose right context cannot fit causes an explicit error instead of a
silently different look-ahead. Each invocation resets the cache.

## 3. Optional early exit

```bash
python train_exit.py --model runs/l6/model --data-dir data/train --out runs/l6_exit --seed 42
python infer.py --model runs/l6_exit/model --input recording.txt --mode exit --tau 0.99
```

For Small-L6 this adds heads at layers 2-5 and trains them for two epochs. The
encoder and layer-6 head remain unchanged. At inference, every context token is
updated until the target word reaches the selected exit; later layers are
actually skipped. `--tau 0.99` is the fixed reference setting. Matched exit is
calibrated separately below. `training_history.json` reports word-final validation
metrics for every exit under `validation_by_depth`; `validation_word_final`
retains the final-head metrics.

## 4. Optional KV reuse and gate

Prepare the fixed gate streams from the **same training directory**:

```bash
python prepare_gate_data.py --data-dir data/train --out data/gate
```

This deduplicates normalized sentences, removes train/dev overlaps and produces
128 fit streams plus 48 calibration and 48 validation streams, each 256 words.
Calibration and validation come from disjoint halves of the original 5%
validation pool. These are artificial reference-text streams, not ASR recordings.
The default is `--loss-mode word_final`; use the model's training mode here too.
The selected mode is recorded in each stream and checked during feature collection.

Collect features using the exact model that will be deployed. Using the model
with exit heads also allows matched-exit calibration in the same command:

```bash
python collect_features.py --model runs/l6_exit/model --data data/gate/fit.jsonl --split fit --out runs/features/fit.npz
python collect_features.py --model runs/l6_exit/model --data data/gate/calibrate.jsonl --split calibrate --out runs/features/calibrate.npz
python train_gate.py --features runs/features/fit.npz --out runs/gate
python calibrate.py --features runs/features/calibrate.npz --gate runs/gate --out runs/policies.json
```

For cache-only deployment, use `runs/l6/model` in both feature commands instead.
No intermediate exit heads are needed by the cache or gate. A gate and policy are
bound to their exact model, architecture and tokenizer. Refit and recalibrate
when changing the encoder or fine-tuning checkpoint.

```bash
python infer.py --model runs/l6_exit/model --input recording.txt --mode gate --gate runs/gate --policy runs/policies.json
python infer.py --model runs/l6_exit/model --input recording.txt --mode margin --policy runs/policies.json
python infer.py --model runs/l6_exit/model --input recording.txt --mode entropy --policy runs/policies.json
python infer.py --model runs/l6_exit/model --input recording.txt --mode exit --policy runs/policies.json
```

`--mode fast` runs approximate KV reuse without repairs. A repair reruns the
complete current window with fresh positions and replaces only the current
word's label. It does not refresh the cache or revise earlier labels.

Thresholds minimize repairs (gate/margin/entropy), or average executed depth
(matched exit), subject to calibration W-F1 at least that of the full model and
at most a 0.5-point drop for each punctuation class. Test labels never select a
threshold. The constraints apply to calibration, not a guarantee for unseen data.

Evaluate the frozen policy on the separate validation streams:

```bash
python evaluate.py --model runs/l6_exit/model --data data/gate/validate.jsonl --mode gate --gate runs/gate --policy runs/policies.json
```

These labels assess the selected policy; they do not fit the gate or its threshold.

## 5. Pretrain another size from scratch

```bash
python pretrain.py --data-dir data/pretraining --family Small --layers 6 --out runs/pretrain_small_l6 --precision bf16
python pretrain.py --data-dir data/pretraining --family Mini --layers 12 --out runs/pretrain_mini_l12 --precision bf16
python pretrain.py --data-dir data/pretraining --family Tiny --layers 4 --out runs/pretrain_tiny_l4 --precision bf16
```

Use `--precision fp32` on CPU. The default is one million updates, batch 128,
sequence length 128, 10,000 warm-up updates and a peak learning rate of 5e-4.
Pretraining input is `.utf8` raw text; its corpus is separate from punctuation
fine-tuning. The exported encoder is `runs/pretrain_small_l6/encoder/` and can be
passed directly to `finetune.py --encoder`.

| Family | Embedding | Hidden | FFN | Heads | Paper depths |
|---|---:|---:|---:|---:|---|
| Small | 128 | 256 | 1024 | 4 | 12, 6, 4, 3 |
| Mini | 128 | 128 | 512 | 2 | 12, 6 |
| Tiny | 64 | 96 | 384 | 2 | 4 |

Other positive depths are supported as new experiments. Cache/exit quality claims
in the paper concern Small-L6; they do not automatically extend to other sizes.

A pretraining run saves `training.pt` with generator, discriminator, optimizer,
scheduler, RNG and sampler states. Resume into a **new** directory with the same
configuration and corpus:

```bash
python pretrain.py --data-dir data/pretraining --family Small --layers 6 --out runs/resumed --resume runs/pretrain_small_l6/training.pt --precision bf16
```

For a short pretraining smoke test use `--steps 3 --warmup 0 --batch-size 1`.
For fine-tuning/exit smoke tests use `--max-steps 2 --epochs 1`; saved metadata
marks these models as incomplete. Nonempty output directories are not overwritten.

## Evaluation and tests

```bash
python evaluate.py --model runs/l6/model --data examples/aligned.jsonl --mode full
python -m pip install -r requirements-test.txt
python -m pytest -q
```

Evaluation input contains `words` and one aligned integer `label` per word in
`labels` (NONE=0, QUESTION=1, PERIOD=2, COMMA=3). Words must already be normalized
and contain no punctuation; case is lowered automatically. For example:

```json
{"words": ["hello", "world"], "labels": [3, 1]}
```

Punctuated words such as `"world?"` are rejected with their record and word index.
The evaluator never splits or deletes aligned words. Optional
`deleted_reference_labels` count punctuation on deleted reference words as missed;
empty ASR word lists are supported. This command does **not** infer ASR/reference
alignments from raw text. Use `--window 32`, `64` or `128` for full/exit evaluation;
cache policies require `64`, and calibrated policies must match their calibration window.
The test suite uses synthetic data and tiny models, works offline and exercises
training, checkpoint roundtrips, early exit, cache resets, calibration, inference
and pretraining resume.

Verification results and independent review are recorded in
[docs/VALIDATION.md](docs/VALIDATION.md).

## Data and citation

The reference and ASR test transcripts are available from the
[CESNET data share](https://owncloud.cesnet.cz/index.php/s/q0TbViTq8GG7sLI).
The training corpora and fine-tuned punctuation/gate weights are not bundled here.
The pretrained encoder is on
[Hugging Face](https://huggingface.co/AILabTUL/electra-small-l6-czech).

Paper: *Efficient Streaming Punctuation Restoration via Selective KV Reuse*,
Martin Polacek, Lukas Mateju and Petr Cerva, AILab@TUL, Technical University of
Liberec. See [CITATION.cff](CITATION.cff).
