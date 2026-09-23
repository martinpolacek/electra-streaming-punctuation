# Source provenance

The implementation was prepared from the manuscript and its experimental source
archive. `source_hashes.json` records the exact source-file hashes used for the
initial audit; those source files are not runtime dependencies.

- Architecture and legacy conversion: archived `runtime_model.py`,
  `electra_moe_definition.py`, `finetune_scaling.py`, `multiexit.py`.
- Sentence preprocessing and block construction: archived `data_loader.py`.
- Historical word-final mask: `word_final/word_final_data.py`.
- Target-word early exit: `target_exit.py`.
- Cache/features: `committed_word_cache.py` and the current-only path in
  `cache_gate_benchmark_runtime.py`.
- Gate streams/training/calibration: `prepare_cache_gate_data.py`,
  `train_cache_gate.py`, `calibrate_conservative_cache.py`, `review_controls.py`.
- Pretraining: the original ELECTRA training objective, optimizer and scaling
  launch settings; the portable sampler is described in REPRODUCIBILITY.md.

Tokenizer: [AILabTUL/mELECTRA](https://huggingface.co/AILabTUL/mELECTRA), revision
`b4f2d443cff9c9ca2c9a63c246c6c343a9b5396d`, credited to AILabTUL under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The tokenizer is fetched
or supplied by the user and is not vendored in this source repository.
