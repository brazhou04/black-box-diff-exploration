# Binary source-trust data contract

The current exploratory design uses a binary `safe`/`unsafe` prompt taxonomy and trusts documented public-source labels without independent human review. It does not create a `dual_use_or_ambiguous` category or make separate dual-use claims.

The manifest-hashed recovery snapshot used by the current experiment is committed to Git so a fresh Kaggle checkout can continue without regenerating the expensive M3 targets. The preparation commands populate:

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
  eval/frozen_manifest.json
  preparation_manifest.json
```

All training records require string `id` and `prompt` fields. M1 adds `response`; M2 adds `response` and binary `category`; M3 is generated from `safety_shared` and stores `initial_response`, `constitution`, `critique`, `revised_response`, and `condition`; M4 stores `chosen` and `rejected`.

Every external record carries provenance containing its dataset, source, immutable revision, license, split, selection rule, transformation, and final operational category.

## Automatic public-source workflow

Install `requirements-kaggle.txt`, then run:

```bash
python scripts/acquire_public_data.py --train-examples 300 --eval-examples 100
python scripts/finalize_trusted_data.py
```

The acquisition step writes revision-pinned source candidates under `data/review/`. That directory name is retained for compatibility with previously acquired candidates; these files no longer require record-by-record approval.

The binary finalizer applies deterministic rules:

| Output | Source-trust rule |
| --- | --- |
| M1 benign control | First filtered UltraChat `train_sft` records |
| M2/M3 `safe` half | A disjoint set of filtered UltraChat `train_sft` records |
| M2/M3 `unsafe` half | PKU-SafeRLHF pairs with exactly one source-labeled safe response and one unsafe response |
| M2 direct unsafe target | The PKU response labeled safe by the source |
| M4 preference pairs | The unsafe-category PKU pairs only; source-safe response chosen and source-unsafe response rejected |
| Benign utility evaluation | UltraChat `test_sft` |
| Harmful evaluation | HarmBench text-test behaviors |
| Overrefusal evaluation | XSTest prompts whose source label is `safe` and whose type is not a contrast type |

M2/M3 are balanced as evenly as possible between `safe` and `unsafe`, with one extra safe record when the requested count is odd. M1 and M2/M3 use disjoint UltraChat records. Exact normalized prompt overlap between any training and evaluation output is rejected.

## Interpretation limits

This is a source-trust mapping, not a human-validated dataset. In particular:

- UltraChat is treated operationally as safe after basic filtering; it is not a prompt-safety benchmark.
- A PKU pair containing one unsafe response is treated operationally as an unsafe prompt, although PKU primarily labels responses rather than prompt intent.
- Ambiguous prompts are not independently identified. They may still be present through source-label error or mapping error, but they receive no separate category.
- M4 covers only the PKU unsafe subset because UltraChat does not provide rejected alternatives.
- PKU-SafeRLHF is `CC-BY-NC-4.0`; verify that the intended use is non-commercial and compatible with all source terms.

`data/preparation_manifest.json` explicitly records `human_review_completed: false`, the automatic mapping rules, source-candidate hashes, output hashes, and binary category counts. Use `--overwrite` only when intentionally replacing previously generated experimental files.

Synthetic records remain under `tests/fixtures/`, carry `synthetic_fixture: true`, and are never experimental data.
