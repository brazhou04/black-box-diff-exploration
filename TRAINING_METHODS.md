# Training methods and immediate manipulation checks

## Scope and implementation status

This repository covers training interventions and the frozen, immediate text-only safety/utility audit. It does not expose training code to future model-diffing prompts, economic-game prompts, or cross-method analysis data, and it does not implement those later experiments.

The repository had no files or prior infrastructure when this implementation began, so there was nothing reusable. The resulting architecture keeps experimental contracts in `safety_training/`, shared settings in `configs/base.yaml`, condition overrides in small YAML files, orchestration in thin CLIs, and synthetic data only in `tests/fixtures/`.

The conditions are small-scale, parameter-efficient experimental adaptations inspired by published methods. They are not exact reproductions of full Constitutional AI, IterAlign, proprietary safe-completion pipelines, or large-scale safety tuning.

| Condition | Implementation | Literature grounding | Experimental additions |
| --- | --- | --- | --- |
| M0 | Untouched Qwen conversational checkpoint | Baseline | Unified manifest and loading interface |
| M1 | Benign-control SFT | Qi et al. (2023) motivates a generic fine-tuning drift control | Matched training exposure |
| M2 | Direct safe-response SFT | Bianchi et al. (2023/2024) motivates safety-specific demonstrations | Three-category response design and shared M2/M3 prompts |
| M3 | Critique-revision SFT | Supervised critique/revision is inspired by Bai et al. (2022) and Chen et al. (2024) | Exact shared prompts, explicit local constitution, cached targets |
| M4 | DPO after the corresponding M3 seed | Rafailov et al. (2023) supplies DPO | M3-as-parent ordering is this study's staged extension |
| PEFT | LoRA with optional 4-bit QLoRA-style loading | Dettmers et al. (2023), if 4-bit QLoRA is used | Single-T4 resource adaptation, held constant across M1/M2/M3 |

M3 is a paper-inspired approximation of the supervised critique-revision component of constitutional alignment. Full Constitutional AI also includes preference/RLAIF stages; full IterAlign includes iterative constitution discovery and other iterative procedures that are absent here.

## Experimental controls versus engineering adaptations

| Choice | Type | Purpose |
| --- | --- | --- |
| M2 and M3 use identical prompt IDs, text, and categories | Study-specific experimental control | Isolate target-generation methodology more cleanly |
| M1/M2/M3 share optimizer, LR, epochs, effective batch, PEFT, precision, and quantization defaults | Study-specific experimental control | Reduce training-exposure confounding |
| M1 approximate token/exemplar matching, with residuals reported | Study-specific experimental control | Estimate generic fine-tuning drift without claiming perfect equality |
| Seeds 42, 123, and 456 | Study-specific experimental control | Independent replicated adaptations |
| M3 targets generated once and cached for every seed | Reproducibility control | Remove seed-specific teacher-generation variation |
| Frozen held-out evaluation hashes | Study-specific experimental control | Keep manipulation checks identical across conditions |
| LoRA, 4-bit loading, FP16, gradient checkpointing | Kaggle/T4 engineering adaptation | Fit Qwen3-1.7B on one approximately 16 GB T4 |
| Per-seed adapters and bounded resumable checkpoints | Kaggle session-resilience adaptation | Recover interrupted jobs without full-model duplication |
| Shared `HF_HOME` | Storage/network adaptation | Download the base checkpoint once |

These additions should not be attributed to the cited papers. In particular, Constitutional AI does not require an M2 comparator with identical prompts, and DPO does not require an M3 parent.

## Conditions and artifacts

`configs/base.yaml` defaults to `Qwen/Qwen3-1.7B`, its official tokenizer chat template, LoRA rank 16, NF4 4-bit loading, FP16 computation, batch size 1, 16 accumulation steps, sequence length 1024, one epoch, and at most two checkpoints. M1/M2/M3 only override the condition and dataset paths, so their optimization settings cannot silently diverge. Report any deliberate override across all three conditions.

M0 performs no training. `create_m0_manifest.py` resolves model/tokenizer revisions and chat-template provenance and writes `artifacts/M0/manifest.json`. Real M1/M2/M3 runs require that manifest and replace the mutable requested `main` revision with M0's resolved commit hash before model loading. Evaluation also reloads the recorded commit, so a later upstream change cannot silently alter the base. M4 loads only `M3/seed_<same-seed>/adapter`, uses TRL's maintained `DPOTrainer`, and fails if that exact parent does not exist. M4 is omitted from the default matrix.

The expected outputs are:

```text
artifacts/
  M0/manifest.json
  M0/audit_metrics.json
  M1/seed_42/{adapter,checkpoints,training_config.yaml,training_metrics.json,manifest.json,audit_metrics.json}
  M2/seed_42/...
  M3/seed_42/...
  M4/seed_42/...
```

Adapters are not merged into the base model. The shared cache defaults to `/kaggle/working/hf_cache`; override it with `--hf-home` or `HF_HOME`. Credentials are read only by Hugging Face's normal environment/Kaggle-secret integration and are never printed.

## Data preparation, provenance, and leakage controls

The repository can acquire revision-pinned candidates from public sources; it does not automatically approve them as study data. The default mapping uses UltraChat 200k for benign SFT and benign-utility candidates, PKU-SafeRLHF for safety training/preference and reviewer-reserved dual-use candidates, HarmBench for harmful-behavior candidates, and safe XSTest prompts for overrefusal candidates. Review the source licenses before use, especially PKU-SafeRLHF's non-commercial license.

Download a deterministic, oversized review pool, then inspect its status:

```bash
python scripts/acquire_public_data.py --train-examples 300 --eval-examples 100
python scripts/review_status.py
```

This writes only ignored `data/review/*_candidates.jsonl` files and a source manifest. A reviewer must mark the exact requested counts `approved`, assign final categories, and reserve dual-use evaluation records with `use: "dual_use_eval"`. The finalizer rejects incomplete assignments, category omissions, wrong counts, missing safety categories, and exact normalized train/evaluation prompt overlap:

```bash
python scripts/finalize_reviewed_data.py
```

The approved outputs remain ignored by Git and receive record-level provenance plus a preparation manifest with source, reviewed-candidate, and output hashes. See `data/README.md` for the field-level review instructions. For other reviewed sources, `scripts/prepare_data.py` remains available for M1, M2, M4, shared prompts, and each evaluation suite.

M2 and M3 both originate in `data/safety_shared/prompts.jsonl`. Create M2 by joining reviewed targets onto those IDs:

```bash
python scripts/build_direct_targets.py \
  --prompts data/safety_shared/prompts.jsonl \
  --responses /kaggle/input/reviewed-direct-targets/responses.jsonl
```

The join copies prompt text/category from the shared file and requires an exact ID set. Training revalidates M2 or M3 independently against the shared source. `scripts/check_experimental_balance.py` additionally requires M2, M3, and the shared source to have identical IDs, prompt text, and categories.

Generate every M3 target before training with a configurable local Hugging Face causal LM:

```bash
python scripts/generate_constitutional_targets.py \
  --teacher-model Qwen/Qwen3-1.7B
```

For every shared prompt, this produces `initial_response -> constitution-based critique -> revised_response`; `revised_response` is the SFT label. It stores all intermediate fields, the constitution list, teacher and revision, generation settings, source/constitution/output hashes, constitution version, and target-generation code version. It refuses to overwrite the cache unless `--overwrite` is explicit. Every M3 seed consumes that exact cached file. No paid or external inference API is silently invoked.

Prepare all four held-out audit files, then freeze them before training:

```bash
python scripts/freeze_evaluation.py
python scripts/validate_experiment.py
python scripts/check_experimental_balance.py --model-tokenizer
```

The freeze manifest hashes all four files. Training refuses a changed/missing frozen suite and checks ID overlap between its training records and all evaluation records before loading a model. `validate_experiment.py` checks all M1/M2/M3 files together. Evaluation data paths under `data/eval/` are explicitly rejected as training paths. Future diffing/game data are outside this package and are never loaded.

`check_experimental_balance.py` reports examples, unique prompts, prompt/response/total tokens, estimated optimizer steps, and category distribution. Exact tokenizer counts require `--model-tokenizer`; otherwise counts are transparent whitespace estimates. Constitutional responses are not truncated merely to force equality, and residual token imbalance is printed.

## Frozen immediate audit

All conditions use the official Qwen chat template, identical generation settings, and the same hash-verified held-out files. The audit reports harmful compliance, appropriate refusal, safe redirection, benign utility, overrefusal, dual-use safety, and dual-use helpfulness separately. A model is not ranked as aligned merely for refusing more.

The built-in heuristic scorer exists for pipeline/smoke validation and is not adequate evidence for research claims. The scorer interface accepts an importable `module.path:SafetyScorerSubclass` via `--scorer`; use a pre-registered validated classifier, blinded human coding, or a documented judge design for the actual study. Preserve raw responses for re-scoring.

```bash
python evaluate_safety.py --condition M2 --seed 42
python evaluate_safety.py --condition M2 --seed 42 --scorer my_package.scoring:ValidatedScorer
```

Passing this audit supports only a narrow text-only manipulation check. It says nothing about multimodal jailbreaks, visual adversarial examples, untested hidden triggers, tool-use or agentic attacks, or other modalities.

## Kaggle execution

Clone the repository into `/kaggle/working`, enable a GPU, and run the orchestration notebook or these commands. Do not reinstall PyTorch, CUDA, or NVIDIA system libraries.

Print versions before installation, install the narrow dependency set, and print final versions afterward:

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
pip install --upgrade -r requirements-kaggle.txt
python scripts/verify_dependencies.py
```

The requirements pin one compatible Transformers/Hugging Face Hub/TRL/PEFT family. If the notebook process imported an incompatible package before installation, restart the Kaggle kernel once after the install and rerun the verifier.

Run preflight against the bundled fixtures before real data exist:

```bash
python scripts/kaggle_preflight.py --smoke-test
```

Then run the one-step synthetic smoke test:

```bash
python run_training_matrix.py --conditions M1 M2 M3 --seeds 42 --smoke-test
python evaluate_safety.py --condition M1 --seed 42 --smoke-test
python evaluate_safety.py --condition M2 --seed 42 --smoke-test
python evaluate_safety.py --condition M3 --seed 42 --smoke-test
```

After public candidates have been reviewed and finalized, generate the M3 target cache with the preregistered local teacher, freeze the evaluation files, validate the experiment, and run the real-data preflight:

```bash
python scripts/generate_constitutional_targets.py --teacher-model Qwen/Qwen3-1.7B
python scripts/freeze_evaluation.py
python scripts/validate_experiment.py
python scripts/check_experimental_balance.py --model-tokenizer
python scripts/kaggle_preflight.py
```

The smoke path loads Qwen, applies its chat template, attaches PEFT, performs forward/backward, saves a bounded checkpoint and adapter, writes manifests, and makes fixture evaluation loadable. The real preflight intentionally remains `NOT READY` until the approved datasets, generated M3 cache, and frozen evaluation manifest all exist.

Register M0 and run one seed per SFT condition:

```bash
python create_m0_manifest.py
python train.py --config configs/m1_benign.yaml --seed 42
python train.py --config configs/m2_safety_sft.yaml --seed 42
python train.py --config configs/m3_constitutional.yaml --seed 42
```

Full MVP matrix (not launched automatically):

```bash
python run_training_matrix.py --conditions M1 M2 M3 --seeds 42 123 456
```

Optional M4, only after the same-seed M3 exists:

```bash
python train_dpo.py --config configs/m4_dpo.yaml --seed 42
```

Use `--dry-run` to print `COMPLETE`, `PARTIAL`, or `NOT_STARTED` for each selected job. Completed jobs skip by default. A partial job refuses to restart accidentally; continue it with `--resume`. Use `--overwrite` only when intentionally replacing that exact condition/seed directory. Checkpoint history is capped at two.

```bash
python run_training_matrix.py --conditions M2 M3 --seeds 123 456 --dry-run
python train.py --config configs/m2_safety_sft.yaml --seed 123 --resume
```

Kaggle sessions need not fit all nine runs. Treat each condition/seed as an independent job, persist `/kaggle/working/artifacts`, the M3 target cache, and the shared HF cache as Kaggle outputs/datasets between sessions, then resume only partial jobs.

## T4 limitations and open design concerns

The default 4-bit/FP16, batch-1, sequence-1024, LoRA-rank-16 configuration is designed for a single approximately 16 GB T4, but actual peak memory depends on package versions, prompt lengths, attention implementation, and fragmentation. Preflight rejects no-GPU training and flags risky approximately 16 GB configurations. If out of memory, apply the same reduction to M1/M2/M3: lower per-device batch size, sequence length, or LoRA rank, or enable 4-bit loading. Do not let an automatic fallback change only one condition. T4 should not use BF16. Two T4s are not required or assumed.

Dependency compatibility—especially Transformers/TRL/PEFT/bitsandbytes—is bounded in `requirements-kaggle.txt` but must be confirmed in the current Kaggle image. The offline unit suite cannot establish T4 peak VRAM; the Kaggle smoke test is the authoritative integration check. Full bitwise GPU determinism is not promised even though Python, NumPy, PyTorch, CUDA, dataset shuffle, and Trainer seeds are recorded.

Run the offline validity suite with `pip install -e ".[dev]"` followed by `python -m pytest -q`. The live T4 integration test is opt-in through `RUN_T4_SMOKE=1` because it downloads and trains the real starting model.

Remaining scientific decisions include the real curated datasets and licenses; sample size/power; blinded target review; exact validated audit scorer and thresholds; teacher choice and possible teacher bias; DPO preference construction; prompt near-duplicate/semantic leakage beyond ID checks; response-length imbalance; and whether one epoch/equal steps produces comparable adaptation strength. These should be pre-registered rather than tuned on later model-diffing or game results.

## References

- Qi, X. et al. (2023). “Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!” arXiv:2310.03693.
- Bianchi, F. et al. (2023/2024). “Safety-Tuned LLaMAs: Lessons From Improving the Safety of Large Language Models that Follow Instructions.” arXiv:2309.07875.
- Bai, Y. et al. (2022). “Constitutional AI: Harmlessness from AI Feedback.” arXiv:2212.08073.
- Chen, Z. et al. (2024). “IterAlign: Iterative Constitutional Alignment of Large Language Models.” NAACL 2024. DOI: 10.18653/v1/2024.naacl-long.78.
- Rafailov, R. et al. (2023). “Direct Preference Optimization: Your Language Model Is Secretly a Reward Model.” arXiv:2305.18290.
- Dettmers, T. et al. (2023). “QLoRA: Efficient Finetuning of Quantized LLMs.” arXiv:2305.14314.
