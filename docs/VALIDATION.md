# Tests and checkpoint comparisons

## Test run

23 September 2026: **51 tests passed in 94.44 seconds**.
Python 3.13.3, Windows CPU, PyTorch 2.8.0+cu126, Transformers 4.55.0.

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

The offline suite covers:

- short pretraining and fine-tuning runs, including exact pretraining resume;
- frozen exit-head training, per-head validation and layer skipping;
- gate-data preparation, feature collection, gate fitting and calibration;
- inference policies, evaluation windows and ASR deletions;
- text normalization, empty inputs and partial-word loss masks;
- HF/legacy import, model integrity, cache eviction and position resets.

A separate check used the released Small-L6 encoder and its tokenizer: import,
fine-tune on `examples/`, then run inference. All three commands completed.
These checks use short training runs; full pretraining and three-seed replication
were not repeated.

## Checkpoint comparisons

The original fine-tuned Small-L6 seed-42 checkpoint was compared with this
implementation on 225 consecutive word windows, including cache eviction and
position reset.

| Output | Maximum absolute difference |
|---|---:|
| Full-model logits | 0.0 |
| Cache logits | 0.0 |
| All 268 gate features | 0.0 |

Full and cached predictions had zero label mismatches against their respective
original implementations.

The native HF encoder and imported encoder were compared on padded batches of
lengths 1, 7, 32, 64 and 128. Maximum hidden-state difference: **9.06e-6**, within
absolute and relative tolerances of 1e-5.

Results and source checkpoint hash: [real_checkpoint_parity.json](real_checkpoint_parity.json).
