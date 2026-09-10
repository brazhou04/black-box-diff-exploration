from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_training.datasets import validate_records, validate_unique_ids
from safety_training.io import atomic_write_jsonl, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Join supplied direct responses to the structurally shared M2/M3 prompts")
    parser.add_argument("--prompts", default="data/safety_shared/prompts.jsonl")
    parser.add_argument("--responses", required=True, help="JSONL with id and response")
    parser.add_argument("--output", default="data/safety_direct/train.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}")
    prompts = read_jsonl(args.prompts)
    responses = read_jsonl(args.responses)
    validate_unique_ids(prompts, "shared prompts")
    validate_unique_ids(responses, "direct responses")
    response_by_id = {record["id"]: record for record in responses}
    if set(response_by_id) != {record["id"] for record in prompts}:
        raise ValueError("Direct responses must have exactly the shared prompt IDs")
    records = []
    for prompt in prompts:
        response = response_by_id[prompt["id"]]
        records.append(
            {
                **prompt,
                "response": response["response"],
                "direct_target_provenance": response.get("provenance"),
            }
        )
    validate_records(records, "M2")
    atomic_write_jsonl(output, records)
    print(f"Wrote {len(records)} direct targets to {output}")


if __name__ == "__main__":
    main()
