# Safety post-training experiment

This repository implements only the model-training interventions and immediate text-only safety/utility manipulation checks for `Qwen/Qwen3-1.7B`. It intentionally does not implement model diffing, economic games, or cross-method behavioral analysis.

The default MVP matrix is M1 benign-control SFT, M2 direct safe-response SFT, and M3 cached constitutional critique-revision SFT at seeds 42, 123, and 456. M0 is the untouched conversational checkpoint. M4 (DPO after the corresponding M3 seed) is optional and never selected by default. All trained conditions save LoRA adapters rather than redundant base-model copies.

Start with [TRAINING_METHODS.md](TRAINING_METHODS.md) for data preparation, experimental controls, Kaggle commands, recovery behavior, audit interpretation, and methodological limitations. Real training/evaluation data are deliberately absent from Git; [data/README.md](data/README.md) defines the schemas, automatic binary source-trust workflow, and provenance contract. This simplified workflow does not perform independent human validation or support a separate dual-use category.
