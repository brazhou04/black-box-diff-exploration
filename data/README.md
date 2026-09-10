# Experimental data contract

No study dataset is committed by this implementation. The repository can download revision-pinned candidates from public sources, but it never treats source labels as final study labels. A human must review every selected record before the finalizer will populate these paths:

```text
data/
  benign_control/train.jsonl
  safety_shared/prompts.jsonl
  safety_direct/train.jsonl
  safety_constitutional/train.jsonl
  safety_constitutional/generation_manifest.json
  dpo/train.jsonl
  eval/harmful_test.jsonl
  eval/benign_utility_test.jsonl
  eval/overrefusal_test.jsonl
  eval/dual_use_test.jsonl
  eval/frozen_manifest.json
```

All training records require string `id` and `prompt` fields. M1 adds `response`; M2 adds `response` and `category`; M3 is generated from `safety_shared` and stores `initial_response`, `constitution`, `critique`, `revised_response`, and `condition`; M4 stores `chosen` and `rejected`. Safety categories are `clearly_benign`, `dual_use_or_ambiguous`, and `clearly_unsafe`.

Every externally derived training or evaluation record must contain a `provenance` object with dataset name, source, revision/version, license, original split, selection criteria, transformations, and final category. `scripts/prepare_data.py` supports M1, M2, M4, shared prompts, and each evaluation suite; M3 is produced by the dedicated cached target generator. Preserve any additional source identifiers.

## Public-source workflow

After installing `requirements-kaggle.txt`, download an oversized, deterministic review pool:

```bash
python scripts/acquire_public_data.py --train-examples 300 --eval-examples 100
python scripts/review_status.py
```

This creates only ignored files under `data/review/`. It resolves immutable source revisions and records them in `source_manifest.json`. The current source mapping is:

| Study material | Candidate source | Source split or subset |
| --- | --- | --- |
| M1 benign-control SFT | UltraChat 200k | `train_sft` |
| M2/M3 shared prompts and direct targets; optional M4 pairs | PKU-SafeRLHF | `train` |
| Benign utility evaluation | UltraChat 200k | `test_sft` |
| Harmful evaluation | HarmBench | text test behaviors |
| Overrefusal evaluation | XSTest | safe prompts only |
| Dual-use evaluation | PKU-SafeRLHF | records explicitly reserved during review |

Review the five `*_candidates.jsonl` files outside the training loop. For each record kept, change `review_status` from `pending` to `approved`; leave rejected records pending or set them to `rejected`. Also:

- M1 approvals require `final_category: "clearly_benign"`.
- Safety approvals require one of the three allowed `final_category` values and `use: "train"` or `use: "dual_use_eval"`.
- Every dual-use evaluation approval must use `final_category: "dual_use_or_ambiguous"`; these records are excluded from training.
- Benign, harmful, and overrefusal candidates still require prompt-level approval even when their source supplies a label.
- Check license/terms for the intended use. In particular, PKU-SafeRLHF is non-commercial (`CC-BY-NC-4.0`).

The finalizer requires exactly the requested number of approved examples in every group, all three safety-training categories, and no exact normalized train/evaluation prompt overlap:

```bash
python scripts/review_status.py
python scripts/finalize_reviewed_data.py
```

It creates `data/preparation_manifest.json` with output and reviewed-candidate hashes. It will not overwrite existing approved data unless `--overwrite` is explicit. Next, generate M3 targets, freeze the evaluation suite, and run the experiment validator as described in `TRAINING_METHODS.md`.

Synthetic records belong only in `tests/fixtures/` and have `synthetic_fixture: true`. They are not experimental data.
