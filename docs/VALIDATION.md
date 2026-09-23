# Release validation

Validated on 23 September 2026, Python 3.13.3, Windows CPU,
PyTorch 2.8.0+cu126 and Transformers 4.55.0.

## Automated checks

- Complete offline suite after the review fixes: **51 passed** (94.44 seconds).
- README smoke workflow passed using a local copy of the released Small-L6
  encoder: native HF import, fine-tuning on `examples/`, and inference with the
  actual tokenizer. Attached punctuation preserved the expected four input words.
- Before these fixes, all 20 tests in the published version passed independently.
- An independent Astra review also ran the earlier 18-test suite successfully.

The tests cover an actual short pretraining run, deterministic resume with
identical weights, fine-tuning, frozen exit-head training, gate-data preparation,
feature collection, helpful-repair gate training, matched calibration, every
inference policy, checkpoint integrity, partial-word masking and empty-ASR
punctuation deletions. Synthetic gate examples in the CLI test test the training
and calibration path, not the paper's reported quality.

The added regressions cover attached punctuation across training/inference/gate
preparation, rejection of punctuated aligned words, empty lexical inputs, final
sentence fragments and numeric endings, per-depth validation, evaluation windows,
HF conversion with and without an embedding projection, resolved Hub revisions,
legacy exit-head mapping, and exact preservation of reused K/V prefixes during
eviction. Native-HF tests use local synthetic models and require no download.

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

A subsequent read-only review of the published source used Claude Opus 5.5.
It identified input-normalization, exit-validation and CLI issues. The fixes and
regression checks listed above were applied after that review; the 51-test run
validates the resulting source. Claude did not perform a second review of the fixes.

## Scope

No million-step retraining or full three-seed replication was performed for this
source release. The checks cover short training runs, inference behavior and
checkpoint compatibility as detailed above.
