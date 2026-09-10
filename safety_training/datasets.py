from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .config import REPO_ROOT, resolve_path
from .io import read_jsonl, sha256_file


SAFETY_CATEGORIES = {"clearly_benign", "dual_use_or_ambiguous", "clearly_unsafe"}
TRAIN_FIELDS = {
    "M1": {"id", "prompt", "response"},
    "M2": {"id", "prompt", "response", "category"},
    "M3": {
        "id",
        "prompt",
        "category",
        "initial_response",
        "constitution",
        "critique",
        "revised_response",
        "condition",
    },
    "M4": {"id", "prompt", "chosen", "rejected"},
}
EVAL_FILES = {"harmful", "benign_utility", "overrefusal", "dual_use"}
PROVENANCE_FIELDS = {
    "dataset_name",
    "source",
    "revision_or_version",
    "license",
    "original_split",
    "selection_criteria",
    "transformations",
    "final_category",
}


def validate_record_provenance(record: dict[str, Any], label: str) -> None:
    if record.get("synthetic_fixture"):
        return
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{label} record {record.get('id')} lacks provenance")
    missing = sorted(PROVENANCE_FIELDS - provenance.keys())
    if missing:
        raise ValueError(f"{label} record {record.get('id')} provenance missing {missing}")


def validate_unique_ids(records: list[dict[str, Any]], label: str) -> None:
    ids = [record.get("id") for record in records]
    missing = [index for index, value in enumerate(ids) if not isinstance(value, str) or not value]
    if missing:
        raise ValueError(f"{label}: missing/non-string IDs at record indices {missing[:5]}")
    duplicates = sorted(value for value, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"{label}: duplicate IDs: {duplicates[:10]}")


def validate_records(records: list[dict[str, Any]], condition: str) -> None:
    if condition not in TRAIN_FIELDS:
        raise ValueError(f"Unsupported training condition {condition}")
    if not records:
        raise ValueError(f"{condition}: dataset is empty")
    validate_unique_ids(records, condition)
    required = TRAIN_FIELDS[condition]
    for index, record in enumerate(records):
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(f"{condition} record {index} missing fields {missing}")
        for field in required & {"prompt", "response", "revised_response", "chosen", "rejected"}:
            if not isinstance(record[field], str) or not record[field].strip():
                raise ValueError(f"{condition} record {index} has empty {field}")
        if condition in {"M2", "M3"} and record["category"] not in SAFETY_CATEGORIES:
            raise ValueError(f"{condition} record {record['id']} has invalid category {record['category']!r}")
        if condition == "M3" and record["condition"] != "M3":
            raise ValueError(f"M3 record {record['id']} has condition={record['condition']!r}")
        if condition == "M4" and record["chosen"].strip() == record["rejected"].strip():
            raise ValueError(f"M4 record {record['id']} has identical chosen/rejected responses")
        validate_record_provenance(record, condition)


def load_training_records(path: str | Path, condition: str) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    validate_records(records, condition)
    return records


def response_for(record: dict[str, Any], condition: str) -> str:
    if condition == "M3":
        return record["revised_response"]
    return record["response"]


def validate_shared_safety_prompts(
    shared_path: str | Path, m2_path: str | Path, m3_path: str | Path
) -> dict[str, Any]:
    shared = read_jsonl(shared_path)
    m2 = load_training_records(m2_path, "M2")
    m3 = load_training_records(m3_path, "M3")
    validate_unique_ids(shared, "shared safety prompts")
    for record in shared:
        if not isinstance(record.get("prompt"), str) or record.get("category") not in SAFETY_CATEGORIES:
            raise ValueError(f"Invalid shared safety prompt record {record.get('id')}")
        validate_record_provenance(record, "shared safety prompts")
    shared_by_id = {record["id"]: record for record in shared}
    m2_by_id = {record["id"]: record for record in m2}
    m3_by_id = {record["id"]: record for record in m3}
    if set(m2_by_id) != set(m3_by_id) or set(m2_by_id) != set(shared_by_id):
        raise ValueError(
            "M2, M3, and safety_shared must have exactly the same IDs; "
            f"shared={len(shared_by_id)}, M2={len(m2_by_id)}, M3={len(m3_by_id)}"
        )
    mismatches: list[str] = []
    for record_id, source in shared_by_id.items():
        for label, record in (("M2", m2_by_id[record_id]), ("M3", m3_by_id[record_id])):
            if record["prompt"] != source["prompt"] or record["category"] != source["category"]:
                mismatches.append(f"{record_id}:{label}")
    if mismatches:
        raise ValueError(f"Prompt/category mismatches against safety_shared: {mismatches[:10]}")
    return {
        "matched_prompt_count": len(shared_by_id),
        "shared_sha256": sha256_file(shared_path),
        "m2_sha256": sha256_file(m2_path),
        "m3_sha256": sha256_file(m3_path),
    }


def validate_no_contamination(
    training_paths: list[str | Path], evaluation_paths: list[str | Path]
) -> dict[str, int]:
    training_ids: set[str] = set()
    evaluation_ids: set[str] = set()
    for path in training_paths:
        records = read_jsonl(path)
        validate_unique_ids(records, str(path))
        training_ids.update(record["id"] for record in records)
    for path in evaluation_paths:
        records = read_jsonl(path)
        validate_unique_ids(records, str(path))
        evaluation_ids.update(record["id"] for record in records)
    overlap = sorted(training_ids & evaluation_ids)
    if overlap:
        raise ValueError(f"Training/evaluation ID contamination detected: {overlap[:20]}")
    return {"training_ids": len(training_ids), "evaluation_ids": len(evaluation_ids), "overlap": 0}


def validate_eval_records(records: list[dict[str, Any]], suite: str) -> None:
    if suite not in EVAL_FILES:
        raise ValueError(f"Unknown evaluation suite {suite}")
    if not records:
        raise ValueError(f"Evaluation suite {suite} is empty")
    validate_unique_ids(records, f"eval/{suite}")
    for index, record in enumerate(records):
        if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
            raise ValueError(f"eval/{suite} record {index} has no prompt")
        validate_record_provenance(record, f"eval/{suite}")


def dataset_path(config: dict[str, Any], smoke_test: bool = False) -> Path:
    if smoke_test:
        return REPO_ROOT / "tests" / "fixtures" / "training" / f"{config['condition'].lower()}.jsonl"
    path = resolve_path(config["dataset"]["train"])
    eval_root = (REPO_ROOT / "data" / "eval").resolve()
    try:
        path.resolve().relative_to(eval_root)
    except ValueError:
        return path
    raise ValueError(f"Evaluation data cannot be used for training: {path}")


def token_exposure_report(
    path: str | Path,
    condition: str,
    tokenizer: Callable[[str], list[int]] | None = None,
    epochs: float = 1.0,
    effective_batch_size: int = 1,
) -> dict[str, Any]:
    records = load_training_records(path, condition)
    tokenize = tokenizer or (lambda text: text.split())
    prompt_tokens = sum(len(tokenize(record["prompt"])) for record in records)
    response_tokens = sum(len(tokenize(response_for(record, condition))) for record in records)
    categories = Counter(record.get("category", "benign_control") for record in records)
    return {
        "condition": condition,
        "number_of_examples": len(records),
        "unique_prompts": len({record["prompt"] for record in records}),
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": prompt_tokens + response_tokens,
        "estimated_optimizer_steps": math.ceil(len(records) * epochs / effective_batch_size),
        "category_distribution": dict(sorted(categories.items())),
        "token_count_method": "model_tokenizer" if tokenizer else "whitespace_estimate",
    }
