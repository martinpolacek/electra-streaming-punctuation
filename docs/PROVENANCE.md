# Sources

Original experiment files are identified by SHA-256 in
[source_hashes.json](source_hashes.json).

| Component | Experiment source |
|---|---|
| Encoder and training | `runtime_model.py`, `electra_moe_definition.py`, `finetune_scaling.py`, `multiexit.py` |
| Sentence and block preparation | `data_loader.py` |
| Archived word-final mask | `word_final/word_final_data.py` |
| Early exit | `target_exit.py` |
| KV cache and features | `committed_word_cache.py`, `cache_gate_benchmark_runtime.py` |
| Gate data, training and calibration | `prepare_cache_gate_data.py`, `train_cache_gate.py`, `calibrate_conservative_cache.py`, `review_controls.py` |

Pretraining follows the ELECTRA objective and optimizer settings in
[METHOD.md](METHOD.md).

## Tokenizer

[AILabTUL/mELECTRA](https://huggingface.co/AILabTUL/mELECTRA), revision
`b4f2d443cff9c9ca2c9a63c246c6c343a9b5396d`, by AILabTUL,
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The scripts download it as needed or accept a local copy.
