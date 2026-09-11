# Introspection adapters

## Scope and isolation

This subsystem implements a resource-adapted version of Shenoy et al. (2026), “Introspection Adapters: Training LLMs to Report Their Learned Behaviors” (arXiv:2604.16812). It is deliberately separate from the M0–M4 condition matrix. Existing condition code, datasets, adapters, and safety audits are inputs only and are never modified by the introspection commands.

The three downstream channels must remain independent:

```text
frozen target adapter        -> game (IA disabled)
frozen target adapter + IA   -> introspection report
frozen target/base artifacts -> model-diffing agent
```

Do not run the primary game with the IA active. Doing so changes the target model and turns the audit into an intervention.

## Data contracts

Real introspection data are ignored by Git. The tracked `data/introspection/organism_specs.example.jsonl` illustrates the organism specification schema:

- `organism_id`: stable unique ID;
- `behavior_family`: family used for leakage-resistant splitting;
- `introspection_target`: first-person ground-truth behavioral report;
- `split`: `sft`, `dpo`, or `eval`;
- `training_dataset`: JSONL with unique `id`, `prompt`, and `response` strings.

Create `data/introspection/organism_specs.jsonl` from that template. Every auxiliary organism must be derived from the exact revision recorded by M0. M0–M4 must not appear as IA-training organisms.

The training and held-out investigation question banks are separate. Do not add game prompts or model-diffing outputs to either bank.

## Pipeline

First create the normal M0 revision lock, then train the auxiliary behavior organisms:

```bash
python create_m0_manifest.py
python train_introspection_organisms.py \
  --config configs/introspection_sft.yaml \
  --specs data/introspection/organism_specs.jsonl
```

This produces frozen adapters below `artifacts/introspection_organisms/` and writes a hash-locked registry to `data/introspection/organisms.jsonl`.

Build IA training records from the registry and the training question bank:

```bash
python scripts/build_introspection_sft_examples.py
python scripts/build_introspection_dpo_pairs.py
```

The bundled DPO builder pairs the correct behavior report with an unrelated behavior report. This is a deterministic bootstrap. Keep the DPO organisms disjoint from SFT organisms.

Train the SFT IA, then refine it with DPO:

```bash
python train_introspection.py --config configs/introspection_sft.yaml
```

For the stronger refinement used in the paper, generate SFT-IA reports on the DPO organisms, grade each prediction from 1–10 against its hidden label, and build pairs using the paper's default score threshold and margin:

```bash
python evaluate_introspection_organisms.py --ia-stage sft --split dpo
python scripts/build_introspection_dpo_pairs_from_grades.py \
  --graded-predictions data/introspection/graded_sft_predictions.jsonl
python train_introspection_dpo.py --config configs/introspection_dpo.yaml
```

The grader output requires `id`, `organism_id`, `prompt`, `prediction`, and numeric `score` fields. Pair construction includes the ground-truth report at score 10, unrelated reports at score 1, requires a chosen score of at least 7 and a gap of at least 2, and samples across score-pair strata. The grading step is deliberately not silently delegated to the smoke-only safety heuristic; use a separately validated blind judge or human coding.

At each SFT step, the trainer activates one frozen behavior LoRA together with the trainable shared IA, accumulates losses across the configured number of organisms, and updates only the IA. DPO precomputes chosen/rejected reference log-probabilities under the frozen SFT IA before updating a separate copy. Both stages save only the introspection adapter.

Runs are independently resumable and live below:

```text
artifacts/introspection/ia_qwen3_1_7b_v1/seed_42/
  sft/{adapter,checkpoints,manifest.json,training_metrics.json}
  dpo/{adapter,checkpoints,manifest.json,training_metrics.json,reference_logps.jsonl}
```

## Evaluation

First evaluate on the known-answer `eval` organisms. Raw reports and ground-truth labels are written separately so a grader can be run blind to the answers:

```bash
python evaluate_introspection_organisms.py --ia-stage dpo
```

Then evaluate DPO and its no-IA baseline on completed target conditions:

```bash
python evaluate_introspection.py \
  --ia-stage dpo \
  --conditions M0 M1 M2 M3 \
  --seeds 42 123 456
```

Add `M4` only when all requested M4 seeds exist. For M4, the active stack is base + M4 + IA; M3 is not applied again because the saved M4 adapter already began from and updated the M3 adapter.

Evaluation writes raw, ungraded reports and a manifest containing hashes of the IA, every target, and the question bank. Freeze these outputs before normalizing them to behavioral hypotheses. Blind normalization should produce one JSONL file per channel with this shared schema:

```json
{"model_id":"M3_seed_42","behavior_id":"cooperation","signal":0.7,"evidence":["report IDs or game rows"]}
```

`signal` must be normalized to `[-1, 1]`, where sign has the same preregistered meaning in all channels. Then compare the channels:

```bash
python compare_introspection_channels.py \
  --introspection normalized/introspection.jsonl \
  --model-diff normalized/model_diff.jsonl \
  --game normalized/game.jsonl \
  --output artifacts/comparisons/ia_dpo
```

The comparison reports overlap, correlation, directional agreement, and absolute signal gaps without imputing missing hypotheses. Prompts or rounds nested within a seed are not independent model replicates.

## Interpretation limits

- Qwen3-1.7B was not one of the paper's reported model sizes; treat this as a feasibility study rather than an exact reproduction.
- SFT can produce fluent but hallucinated reports. Compare SFT and DPO adapters and report M0/no-IA false-positive rates.
- The IA should be evaluated on behavioral predictions, not whether it names the training algorithm.
- Agreement among IA, game, and model diffing is evidence of convergent validity, not proof of privileged self-knowledge.
- The default DPO pairs are a reproducible bootstrap. Judge-scored erroneous SFT reports are the stronger refinement dataset once a blinded grader is selected.
