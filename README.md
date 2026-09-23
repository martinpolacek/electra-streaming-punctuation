# electra-streaming-punctuation

Code for *Efficient Streaming Punctuation Restoration via Selective KV Reuse*.
Czech ELECTRA models with punctuation fine-tuning, early exit and selective KV-cache reuse.

[Pretrained Small-L6](https://huggingface.co/AILabTUL/electra-small-l6-czech) | [Test transcripts](https://owncloud.cesnet.cz/index.php/s/q0TbViTq8GG7sLI) | [Citation](CITATION.cff)

## Install

Python 3.11-3.13. Run commands from the repository root, preferably in a virtual environment.

```bash
python -m pip install -r requirements.txt
```

For GPU training, install the appropriate PyTorch 2.8.0 CUDA build first.
Training uses CUDA when available; `--device cpu` selects CPU. Inference defaults to CPU.

## Fine-tune and run

Put punctuated training transcripts in UTF-8 `.utf8` files under `data/train/`.
See [examples/train.utf8](examples/train.utf8) for the format. Keep the test recordings
out of this directory. Training corpora and fine-tuned weights are not included.

The Hugging Face model is a pretrained encoder. Train its punctuation head before inference:

```bash
python import_model.py --hf AILabTUL/electra-small-l6-czech --revision b0bb960a057dc7891a861a0a830220529eb88b5a --out runs/pretrained
python finetune.py --encoder runs/pretrained --data-dir data/train --out runs/l6 --seed 42
python infer.py --model runs/l6/model --text "povazujete to za problem" --mode full
```

Fine-tuning uses four epochs, batch size 8 and loss on word-final subwords.
The script holds out 5% of sentences for validation. The final model is saved to
`runs/l6/model/`. Use seeds 42, 13 and 100 for the three runs, each with its own output directory.

To try the commands with the example data, replace `--data-dir data/train` with
`--data-dir examples --epochs 1 --max-steps 2 --device cpu`. This produces a test model.
Output directories must be empty or new.

For a whole recording and per-word JSON output:

```bash
python infer.py --model runs/l6/model --input recording.txt --mode full --format jsonl
```

Input is lowercased and punctuation is removed; `hello,world?` becomes two words.
Each decision uses at most 64 subwords, including up to four following words.
Full and exit modes also accept `--window 32` and `--window 128`.

## Early exit

For Small-L6, train heads at layers 2-5 while keeping the encoder and layer-6 head frozen:

```bash
python train_exit.py --model runs/l6/model --data-dir data/train --out runs/l6_exit --seed 42
python infer.py --model runs/l6_exit/model --input recording.txt --mode exit --tau 0.99
```

Inference stops at the first head whose maximum class probability reaches 0.99;
otherwise it runs through layer 6. The matched exit setting uses a threshold
selected on calibration data, as shown below.

## KV reuse and gate

The fast branch reuses keys and values for earlier words. The gate decides when
to rerun the full window to repair the current prediction. A repair changes the
current label and leaves the cache intact.

Prepare gate data from the same training directory, fit the gate, then calibrate:

```bash
python prepare_gate_data.py --data-dir data/train --out data/gate
python collect_features.py --model runs/l6_exit/model --data data/gate/fit.jsonl --split fit --out runs/features/fit.npz
python collect_features.py --model runs/l6_exit/model --data data/gate/calibrate.jsonl --split calibrate --out runs/features/calibrate.npz
python train_gate.py --features runs/features/fit.npz --out runs/gate
python calibrate.py --features runs/features/calibrate.npz --gate runs/gate --out runs/policies.json
```

These commands use the model with exit heads so the same calibration run can
select a matched exit threshold. For KV reuse alone, use `runs/l6/model` in both
feature commands and in inference. A gate and its thresholds apply to the exact
checkpoint used to collect their features.

```bash
python infer.py --model runs/l6_exit/model --input recording.txt --mode gate --gate runs/gate --policy runs/policies.json
python infer.py --model runs/l6_exit/model --input recording.txt --mode margin --policy runs/policies.json
python infer.py --model runs/l6_exit/model --input recording.txt --mode entropy --policy runs/policies.json
python infer.py --model runs/l6_exit/model --input recording.txt --mode exit --policy runs/policies.json
```

Use `--mode fast` to run KV reuse without repairs. Cache modes use a 64-subword window.
See [METHOD.md](docs/METHOD.md) for the gate features and calibration constraints.

## Evaluate

`evaluate.py` reads JSONL with one normalized word per label:

```json
{"words": ["hello", "world"], "labels": [3, 1]}
```

Labels are `NONE=0`, `QUESTION=1`, `PERIOD=2`, `COMMA=3`. Words must contain no
punctuation; uppercase letters are lowercased automatically. ASR inputs require
word alignment prepared beforehand. Put labels for deleted reference words in
`deleted_reference_labels`. An empty ASR word list is allowed.

```bash
python evaluate.py --model runs/l6/model --data examples/aligned.jsonl --mode full
python evaluate.py --model runs/l6_exit/model --data data/gate/validate.jsonl --mode gate --gate runs/gate --policy runs/policies.json
```

The second command evaluates the frozen gate on separate validation streams.
Full/exit evaluation accepts `--window 32`, `64` or `128`; a calibrated policy
requires the window used during calibration. W-F1 weights the three punctuation
classes by their reference counts.

## Pretrain another size

Use raw `.utf8` text in `data/pretraining/`:

```bash
python pretrain.py --data-dir data/pretraining --family Small --layers 6 --out runs/pretrain_small_l6 --precision bf16
```

Use `--precision fp32` on CPU. The output encoder is
`runs/pretrain_small_l6/encoder/`; pass it to `finetune.py --encoder`.

| Family | Embedding | Hidden | FFN | Attention heads | Depths used in the paper |
|---|---:|---:|---:|---:|---|
| Small | 128 | 256 | 1024 | 4 | 12, 6, 4, 3 |
| Mini | 128 | 128 | 512 | 2 | 12, 6 |
| Tiny | 64 | 96 | 384 | 2 | 4 |

Choose a family and depth with `--family` and `--layers`. The cache and exit
experiments use Small-L6. Training settings are in [METHOD.md](docs/METHOD.md);
checkpoint import and resume are in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Tests

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

Tests run offline with synthetic data and small models.
See [VALIDATION.md](docs/VALIDATION.md) for results and checkpoint comparisons,
and [PROVENANCE.md](docs/PROVENANCE.md) for source files and tokenizer attribution.
