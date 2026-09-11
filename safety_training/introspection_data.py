from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Iterable

from .config import resolve_path
from .io import atomic_write_jsonl, read_jsonl


ORGANISM_SPLITS = {"sft", "dpo", "eval", "calibration"}
NO_ADAPTER = "NO_ADAPTER"


def sha256_tree(path: str | Path) -> str:
    """Hash a directory without depending on filesystem traversal order."""
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {root}")
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Adapter directory is empty: {root}")
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _require_text(record: dict[str, Any], field: str, label: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} requires non-empty string field {field!r}")
    return value.strip()


def load_organisms(
    path: str | Path,
    *,
    split: str | None = None,
    expected_model: str | None = None,
    expected_revision: str | None = None,
    verify_adapters: bool = True,
) -> list[dict[str, Any]]:
    if split is not None and split not in ORGANISM_SPLITS:
        raise ValueError(f"Unknown organism split {split!r}")
    records = read_jsonl(path)
    if not records:
        raise ValueError("Introspection organism manifest is empty")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        label = f"organism record {index}"
        organism_id = _require_text(raw, "organism_id", label)
        if organism_id in seen:
            raise ValueError(f"Duplicate organism_id {organism_id!r}")
        seen.add(organism_id)
        record_split = _require_text(raw, "split", label)
        if record_split not in ORGANISM_SPLITS:
            raise ValueError(f"{label} has invalid split {record_split!r}")
        record = dict(raw)
        record["organism_id"] = organism_id
        record["behavior_family"] = _require_text(raw, "behavior_family", label)
        record["introspection_target"] = _require_text(raw, "introspection_target", label)
        record["base_model"] = _require_text(raw, "base_model", label)
        record["base_revision"] = _require_text(raw, "base_revision", label)
        adapter_value = _require_text(raw, "adapter_path", label)
        if expected_model is not None and record["base_model"] != expected_model:
            raise ValueError(
                f"{organism_id}: base_model={record['base_model']!r} does not match {expected_model!r}"
            )
        if expected_revision is not None and record["base_revision"] != expected_revision:
            raise ValueError(
                f"{organism_id}: base_revision={record['base_revision']!r} does not match locked revision "
                f"{expected_revision!r}"
            )
        if adapter_value == NO_ADAPTER:
            record["_adapter_path"] = None
            if record_split not in {"calibration", "eval"}:
                raise ValueError(f"{organism_id}: {NO_ADAPTER} is allowed only for calibration/eval")
        else:
            adapter_path = resolve_path(adapter_value)
            record["_adapter_path"] = adapter_path
            if verify_adapters:
                actual = sha256_tree(adapter_path)
                declared = record.get("adapter_sha256")
                if declared is not None and declared != actual:
                    raise ValueError(f"{organism_id}: adapter_sha256 does not match {adapter_path}")
                record["_adapter_sha256"] = actual
        if split is None or record_split == split:
            normalized.append(record)
    if split is not None and not normalized:
        raise ValueError(f"No organisms found for split {split!r}")
    return normalized


def _organisms_by_id(organisms: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {record["organism_id"]: record for record in organisms}
    if not result:
        raise ValueError("At least one organism is required")
    return result


def validate_organisms_are_not_condition_targets(
    organisms: Iterable[dict[str, Any]], output_root: str | Path
) -> None:
    root = Path(output_root).resolve()
    condition_roots = [(root / condition).resolve() for condition in ("M0", "M1", "M2", "M3", "M4")]
    for organism in organisms:
        adapter_path = organism.get("_adapter_path")
        if adapter_path is None:
            continue
        resolved = Path(adapter_path).resolve()
        for condition_root in condition_roots:
            try:
                resolved.relative_to(condition_root)
            except ValueError:
                continue
            raise ValueError(
                f"{organism['organism_id']}: M0-M4 target adapters cannot be used to train or refine the IA"
            )


def load_sft_examples(
    path: str | Path, organisms: Iterable[dict[str, Any]], *, allowed_split: str = "sft"
) -> list[dict[str, Any]]:
    by_id = _organisms_by_id(organisms)
    records = read_jsonl(path)
    if not records:
        raise ValueError("Introspection SFT dataset is empty")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        label = f"introspection SFT record {index}"
        record_id = _require_text(raw, "id", label)
        if record_id in seen:
            raise ValueError(f"Duplicate introspection SFT id {record_id!r}")
        seen.add(record_id)
        organism_id = _require_text(raw, "organism_id", label)
        organism = by_id.get(organism_id)
        if organism is None:
            raise ValueError(f"{record_id}: unknown organism_id {organism_id!r}")
        if organism["split"] != allowed_split:
            raise ValueError(
                f"{record_id}: organism {organism_id!r} belongs to {organism['split']!r}, "
                f"not {allowed_split!r}"
            )
        normalized.append(
            {
                **raw,
                "id": record_id,
                "organism_id": organism_id,
                "prompt": _require_text(raw, "prompt", label),
                "response": _require_text(raw, "response", label),
            }
        )
    return normalized


def load_dpo_examples(
    path: str | Path, organisms: Iterable[dict[str, Any]], *, allowed_split: str = "dpo"
) -> list[dict[str, Any]]:
    by_id = _organisms_by_id(organisms)
    records = read_jsonl(path)
    if not records:
        raise ValueError("Introspection DPO dataset is empty")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        label = f"introspection DPO record {index}"
        record_id = _require_text(raw, "id", label)
        if record_id in seen:
            raise ValueError(f"Duplicate introspection DPO id {record_id!r}")
        seen.add(record_id)
        organism_id = _require_text(raw, "organism_id", label)
        organism = by_id.get(organism_id)
        if organism is None:
            raise ValueError(f"{record_id}: unknown organism_id {organism_id!r}")
        if organism["split"] != allowed_split:
            raise ValueError(
                f"{record_id}: organism {organism_id!r} belongs to {organism['split']!r}, "
                f"not {allowed_split!r}"
            )
        chosen = _require_text(raw, "chosen", label)
        rejected = _require_text(raw, "rejected", label)
        if chosen == rejected:
            raise ValueError(f"{record_id}: chosen and rejected responses are identical")
        normalized.append(
            {
                **raw,
                "id": record_id,
                "organism_id": organism_id,
                "prompt": _require_text(raw, "prompt", label),
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return normalized


def load_questions(path: str | Path) -> list[dict[str, str]]:
    records = read_jsonl(path)
    if not records:
        raise ValueError("Introspection question bank is empty")
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for index, raw in enumerate(records):
        label = f"introspection question {index}"
        question_id = _require_text(raw, "id", label)
        if question_id in seen:
            raise ValueError(f"Duplicate introspection question id {question_id!r}")
        seen.add(question_id)
        result.append({"id": question_id, "prompt": _require_text(raw, "prompt", label)})
    return result


def build_sft_examples(
    organism_manifest: str | Path,
    questions_path: str | Path,
    destination: str | Path,
) -> Path:
    organisms = load_organisms(organism_manifest, split="sft", verify_adapters=False)
    questions = load_questions(questions_path)
    rows = [
        {
            "id": f"{organism['organism_id']}__{question['id']}",
            "organism_id": organism["organism_id"],
            "prompt": question["prompt"],
            "response": organism["introspection_target"],
            "target_source": "organism_manifest",
        }
        for organism in organisms
        for question in questions
    ]
    atomic_write_jsonl(destination, rows)
    return Path(destination)


def build_dpo_pairs(
    organism_manifest: str | Path,
    questions_path: str | Path,
    destination: str | Path,
    *,
    seed: int = 42,
) -> Path:
    all_organisms = load_organisms(organism_manifest, verify_adapters=False)
    dpo_organisms = [record for record in all_organisms if record["split"] == "dpo"]
    if not dpo_organisms:
        raise ValueError("No DPO organisms found")
    questions = load_questions(questions_path)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for organism in dpo_organisms:
        negatives = [
            candidate
            for candidate in all_organisms
            if candidate["organism_id"] != organism["organism_id"]
            and candidate["introspection_target"] != organism["introspection_target"]
        ]
        if not negatives:
            raise ValueError(f"No unrelated negative target is available for {organism['organism_id']}")
        different_family = [
            candidate
            for candidate in negatives
            if candidate["behavior_family"] != organism["behavior_family"]
        ]
        pool = different_family or negatives
        for question in questions:
            rejected = rng.choice(pool)
            rows.append(
                {
                    "id": f"{organism['organism_id']}__{question['id']}",
                    "organism_id": organism["organism_id"],
                    "prompt": question["prompt"],
                    "chosen": organism["introspection_target"],
                    "rejected": rejected["introspection_target"],
                    "pair_source": "ground_truth_vs_unrelated_behavior",
                    "rejected_organism_id": rejected["organism_id"],
                }
            )
    atomic_write_jsonl(destination, rows)
    return Path(destination)


def build_dpo_pairs_from_grades(
    organism_manifest: str | Path,
    graded_predictions_path: str | Path,
    destination: str | Path,
    *,
    chosen_threshold: float = 7.0,
    minimum_margin: float = 2.0,
    max_pairs_per_organism: int = 100,
    seed: int = 42,
) -> Path:
    if not 1.0 <= chosen_threshold <= 10.0:
        raise ValueError("chosen_threshold must be in [1, 10]")
    if minimum_margin <= 0 or max_pairs_per_organism < 1:
        raise ValueError("minimum_margin and max_pairs_per_organism must be positive")
    organisms = load_organisms(organism_manifest, verify_adapters=False)
    by_id = {record["organism_id"]: record for record in organisms}
    dpo_ids = {record["organism_id"] for record in organisms if record["split"] == "dpo"}
    graded = read_jsonl(graded_predictions_path)
    if not graded:
        raise ValueError("Graded prediction dataset is empty")
    grouped: dict[tuple[str, str], list[tuple[str, float, str]]] = {}
    seen: set[str] = set()
    for index, raw in enumerate(graded):
        label = f"graded prediction {index}"
        prediction_id = _require_text(raw, "id", label)
        if prediction_id in seen:
            raise ValueError(f"Duplicate graded prediction id {prediction_id!r}")
        seen.add(prediction_id)
        organism_id = _require_text(raw, "organism_id", label)
        if organism_id not in dpo_ids:
            raise ValueError(f"{prediction_id}: organism {organism_id!r} is not in the DPO split")
        prompt = _require_text(raw, "prompt", label)
        prediction = _require_text(raw, "prediction", label)
        score = raw.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not 1.0 <= float(score) <= 10.0:
            raise ValueError(f"{prediction_id}: score must be numeric and in [1, 10]")
        grouped.setdefault((organism_id, prompt), []).append((prediction, float(score), prediction_id))
    rng = random.Random(seed)
    candidates_by_organism: dict[str, list[dict[str, Any]]] = {}
    for (organism_id, prompt), predictions in sorted(grouped.items()):
        organism = by_id[organism_id]
        candidates: dict[str, tuple[float, str]] = {
            text: (score, prediction_id) for text, score, prediction_id in predictions
        }
        candidates[organism["introspection_target"]] = (10.0, "ground_truth")
        distractors = [
            other
            for other in organisms
            if other["organism_id"] != organism_id
            and other["introspection_target"] != organism["introspection_target"]
        ]
        for distractor in distractors:
            candidates.setdefault(distractor["introspection_target"], (1.0, f"distractor:{distractor['organism_id']}"))
        values = [(text, score, source) for text, (score, source) in candidates.items()]
        pairs: list[dict[str, Any]] = []
        for chosen, chosen_score, chosen_source in values:
            if chosen_score < chosen_threshold:
                continue
            for rejected, rejected_score, rejected_source in values:
                if chosen == rejected or chosen_score - rejected_score < minimum_margin:
                    continue
                pairs.append(
                    {
                        "organism_id": organism_id,
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "chosen_score": chosen_score,
                        "rejected_score": rejected_score,
                        "chosen_source": chosen_source,
                        "rejected_source": rejected_source,
                        "pair_source": "judge_scored_sft_rollout",
                    }
                )
        candidates_by_organism.setdefault(organism_id, []).extend(pairs)
    output: list[dict[str, Any]] = []
    for organism_id, pairs in sorted(candidates_by_organism.items()):
        strata: dict[tuple[float, float], list[dict[str, Any]]] = {}
        for pair in pairs:
            strata.setdefault((pair["chosen_score"], pair["rejected_score"]), []).append(pair)
        selected: list[dict[str, Any]] = []
        for key in sorted(strata):
            selected.append(rng.choice(strata[key]))
        remaining = [pair for pair in pairs if pair not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: max(0, max_pairs_per_organism - len(selected))])
        if not selected:
            raise ValueError(f"No valid DPO pairs were produced for {organism_id}")
        for index, pair in enumerate(selected[:max_pairs_per_organism]):
            output.append({"id": f"{organism_id}__graded_pair_{index:04d}", **pair})
    atomic_write_jsonl(destination, output)
    return Path(destination)
