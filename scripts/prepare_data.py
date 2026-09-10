from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_training.datasets import SAFETY_CATEGORIES, validate_eval_records, validate_records, validate_unique_ids
from safety_training.io import atomic_write_jsonl, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize an explicitly supplied, curated JSONL subset")
    parser.add_argument("--input", required=True, help="Local reviewed JSONL; this script never scrapes")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--condition",
        required=True,
        choices=["M1", "M2", "M4", "shared", "eval_harmful", "eval_benign_utility", "eval_overrefusal", "eval_dual_use"],
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--original-split", required=True)
    parser.add_argument("--selection-criteria", required=True)
    parser.add_argument("--transformations", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {destination}")
    records = read_jsonl(args.input)
    validate_unique_ids(records, args.input)
    if not records:
        raise ValueError("Input dataset is empty")
    eval_suite = args.condition.removeprefix("eval_") if args.condition.startswith("eval_") else None
    output = []
    for record in records:
        category = record.get("category", "clearly_benign" if args.condition == "M1" else eval_suite)
        if args.condition in {"M2", "shared"} and category not in SAFETY_CATEGORIES:
            raise ValueError(f"Record {record['id']} needs a valid category")
        if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
            raise ValueError(f"Record {record['id']} needs a non-empty prompt")
        provenance = {
            "dataset_name": args.dataset_name,
            "source": args.source,
            "revision_or_version": args.revision,
            "license": args.license,
            "original_split": args.original_split,
            "selection_criteria": args.selection_criteria,
            "transformations": args.transformations,
            "final_category": category,
        }
        output.append({**record, "provenance": provenance})
    if eval_suite:
        validate_eval_records(output, eval_suite)
    elif args.condition != "shared":
        validate_records(output, args.condition)
    atomic_write_jsonl(destination, output)
    print(f"Wrote {len(output)} curated records to {destination}")


if __name__ == "__main__":
    main()
