# Release validation

Validated on 22 September 2026, Python 3.13.3, Windows CPU,
PyTorch 2.8.0+cu126 and Transformers 4.55.0.

## Automated checks

- Complete offline suite: **18 passed** (149.03 seconds).
- After final importer/empty-ASR changes: **3 focused regression tests passed**
  (11.05 seconds), including two new tests; the repository contains 20 tests.
- An independent Astra review also ran the earlier 18-test suite successfully.

The tests cover an actual short pretraining run, deterministic resume with
identical weights, fine-tuning, frozen exit-head training, gate-data preparation,
feature collection, helpful-repair gate training, matched calibration, every
inference policy, checkpoint integrity, partial-word masking and empty-ASR
punctuation deletions. Synthetic gate examples in the CLI test test the training
and calibration path, not the paper's reported quality.

## Comparisons with original implementations

Using the original fine-tuned Small-L6 seed-42 checkpoint:

- Full-model logits: maximum absolute difference **0.0**.
- 225 consecutive cache windows, including eviction and position reset:
  logits and all 268 gate features have maximum absolute difference **0.0**.
- Full/cache predicted-label mismatches: **0**.

The published native Hugging Face encoder and the imported release encoder were
compared on padded batches of lengths 1, 7, 32, 64 and 128. Maximum hidden-state
absolute difference: **9.06e-6**; comparisons passed at atol=rtol=1e-5. Different
attention implementations can introduce small floating-point differences.
Machine-readable results: [real_checkpoint_parity.json](real_checkpoint_parity.json).

The independent reviewer also compared randomly initialized Small/Mini/Tiny
encoders and 165 consecutive cache windows per family with the archived runtime:
encoder outputs, cache logits and gate features were exactly equal. Batched
early exit chose the same depths as explicit all-head simulation; maximum logit
difference was 2.38e-7.

## Review outcome

The independent Astra review identified and verified fixes for sampler tail
coverage, unsupported HF dropout conversion, attached punctuation in the new
word-final path, model/tokenizer identity checks, and explicit training-mode
provenance on legacy import. Final review reported no remaining concrete blocker
in the reviewed changes. This is evidence from the listed tests and review,
not a guarantee that no undiscovered bug exists.

## Scope

No million-step retraining or full three-seed replication was performed for this
source release. The checks cover short training runs, inference behavior and
checkpoint compatibility as detailed above.
