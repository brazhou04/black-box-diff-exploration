# Experimental data contract

No study dataset is committed by this implementation. Populate these paths only through documented curation:

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

Every externally derived training or evaluation record must contain a `provenance` object with dataset name, source, revision/version, license, original split, selection criteria, transformations, and final category. `scripts/prepare_data.py` supports M1, M2, M4, shared prompts, and each evaluation suite; M3 is produced by the dedicated cached target generator. Preserve any additional source identifiers. Review licenses and terms before acquisition; the scripts do not scrape or silently download data.

Synthetic records belong only in `tests/fixtures/` and have `synthetic_fixture: true`. They are not experimental data.
