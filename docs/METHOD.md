# Method and training settings

## Pretraining

ELECTRA replaced-token detection with shared word embeddings. The generator has
the discriminator's depth, one attention head and one quarter of its hidden/FFN
widths. Position tables have 512 entries.

| Setting | Value |
|---|---|
| Updates | 1,000,000 |
| Batch / sequence length | 128 / 128 subwords |
| Optimizer | AdamW without bias correction |
| Beta / epsilon | (0.9, 0.999) / 1e-6 |
| Weight decay | 0.01; zero for biases and LayerNorm |
| Learning rate | 5e-4 peak; 10,000 warm-up updates, then linear decay |
| Gradient norm limit | 1.0 |
| Seed | 42 |

Loss: generator cross-entropy + 50 x discriminator binary cross-entropy.
Select 15% of eligible tokens; mask 85% of those and leave the rest unchanged
at the generator input. Sample replacements with Gumbel-max; an unchanged token
has replacement label zero. Special tokens are excluded from corruption.
Discriminator loss covers nonpadding tokens. Shared embeddings occur once in
the optimizer; the first warm-up update has learning rate zero.

## Punctuation fine-tuning

Head: `Linear(hidden,512) -> SELU -> Linear(512,4)`. Default `word_final` loss:
weighted cross-entropy at completed word ends, after removing punctuation before
tokenization. Exclude padding and partial words at the 512-subword limit.

| Setting | Value |
|---|---|
| Epochs / batch size | 4 / 8 |
| Backbone / head learning rate | 5e-5 / 2e-4 |
| Optimizer | PyTorch AdamW; beta (0.9, 0.999), epsilon 1e-8, weight decay 0.01 |
| Learning-rate decay | Multiply by 0.95 every 10,000 steps within each epoch, starting at epoch 2 |
| Class weights: NONE / QUESTION / PERIOD / COMMA | 1 / 5 / 2.5 / 1.5 |
| Gradient norm limit | 1.0 |
| Fine-tuning seeds | 42, 13, 100 |
| Validation split | 5% of sentences, split seed 0 |

Lowercase sentences and combine them into blocks of 1-15 sentences, with random
removal of trailing whitespace-separated tokens. The default parser retains
unfinished final sentences and numeric endings, separates `.,?!` and maps `!` to
PERIOD. It drops nonlexical sentences/blocks and records counts in `run.json` and
`training_history.json`. Gate preparation uses the same parser.

Use the final epoch; record word-final validation metrics after each epoch.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for other loss modes.

## Early exit

Small-L6 adds heads at layers 2-5. Train for two epochs, batch 8, learning rate
2e-4; average weighted cross-entropy over the added heads. Freeze the encoder and
layer-6 head, keeping backbone dropout enabled during training. Per-head validation
is saved under `validation_by_depth` in `training_history.json`.

At inference, stop at the first head whose maximum softmax probability reaches
the threshold, or at layer 6. All context tokens advance together. Use evaluation
mode and float32. Fixed exit uses `tau=0.99`; matched exit calibrates the threshold.

## KV reuse

Window: 64 subwords. Recompute the target and up to four following words through
every layer; reuse earlier words' per-layer keys and values. Cached representations
retain the right context available when computed.

Anchor absolute positions until the next index would exceed 127, then recompute
the window from position zero and reset the cache. A repair runs the full window
from position zero and replaces only the current label, leaving the cache intact.

## Gate

For Small-L6, the input has 268 values, in this order:

1. Four fast-branch class probabilities.
2. The target word's final-subword hidden state: 256 values.
3. Eight scalars:
   - available future words / 4;
   - visible target-word length in subwords / 8;
   - target offset within the window / 63;
   - anchored target position / 127;
   - reused prefix length / 64;
   - window length / 64;
   - full-window refresh indicator;
   - newly arrived subwords since the previous decision / 8, initially zero.

Gate: `Linear(268,32) -> ReLU -> Linear(32,1)`, sigmoid score, 8,641 parameters.
Standardize with fit-set means/stds, std floor 1e-4; clip to [-8,8].
Training target: `fast != gold and full == gold`.

Use weighted binary cross-entropy with punctuation weights 1/5/2.5/1.5 and a
positive-class balance computed on the fit set. Train with AdamW, learning rate
0.001, weight decay 0.001, batch size 1024, 32 epochs and seed 42.

## Calibration

`prepare_gate_data.py` removes duplicate sentences and train/validation overlaps.
It creates 128 fit streams from the training pool and 48 calibration plus 48
validation streams from disjoint halves of the held-out 5%. Each stream contains
256 words, concatenated from reference sentences. Sampling seed: 20260909.

Thresholds must satisfy both conditions on the calibration set:

- W-F1 is at least the full model's W-F1.
- Each punctuation-class F1 is within 0.5 points of the full model or higher.

Minimize repair rate for gate/margin/entropy or mean depth for matched exit;
break ties by higher W-F1. Search up to 1025 midpoints between adjacent scores,
plus always/never-repair endpoints. Exit includes 0, 1, 0.99 and a full-model endpoint.

Margin uses `1 - (top1 probability - top2 probability)`. Entropy uses natural
logarithms. Each punctuation checkpoint gets its own gate and calibrated thresholds.
The separate validation streams are used only for evaluation.
