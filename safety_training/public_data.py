from __future__ import annotations

import csv
import hashlib
import io
import os
import random
from pathlib import Path
from typing import Any, Iterable, Iterator

from .datasets import SAFETY_CATEGORIES, validate_eval_records, validate_records
from .io import atomic_write_json, atomic_write_jsonl, read_jsonl, sha256_file


PUBLIC_DATA_CODE_VERSION = "2.0.0-binary-source-trust"
ULTRACHAT_REPO = "HuggingFaceH4/ultrachat_200k"
PKU_REPO = "PKU-Alignment/PKU-SafeRLHF"


def stable_id(prefix: str, prompt: str) -> str:
    digest = hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def extract_first_exchange(record: dict[str, Any]) -> tuple[str, str] | None:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    prompt: str | None = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role") or message.get("from")
        content = message.get("content") or message.get("value")
        if not isinstance(content, str) or not content.strip():
            continue
        if prompt is None and role in {"user", "human"}:
            prompt = content.strip()
        elif prompt is not None and role in {"assistant", "gpt"}:
            return prompt, content.strip()
    return None


def pku_candidate(record: dict[str, Any], revision: str) -> dict[str, Any] | None:
    prompt = str(record.get("prompt") or "").strip()
    responses = [str(record.get("response_0") or "").strip(), str(record.get("response_1") or "").strip()]
    safe = [bool(record.get("is_response_0_safe")), bool(record.get("is_response_1_safe"))]
    if not prompt or not all(responses) or safe[0] == safe[1]:
        return None
    chosen_index = 0 if safe[0] else 1
    source_risk_signal = "exactly one candidate response is source-labeled safe"
    rejected_index = 1 - chosen_index
    provenance = {
        "dataset_name": "PKU-SafeRLHF",
        "source": f"https://huggingface.co/datasets/{PKU_REPO}",
        "revision_or_version": revision,
        "license": "CC-BY-NC-4.0",
        "original_split": "train",
        "selection_criteria": "exactly one response is source-labeled safe and both responses are non-empty",
        "transformations": "selected the source-labeled safe response; retained the unsafe alternative for DPO",
        "final_category": None,
    }
    return {
        "id": stable_id("pku", prompt),
        "prompt": prompt,
        "response": responses[chosen_index],
        "rejected": responses[rejected_index],
        "source_risk_signal": source_risk_signal,
        "final_category": None,
        "source_labels": {
            "response_0_safe": safe[0],
            "response_1_safe": safe[1],
            "better_response_id": record.get("better_response_id"),
            "safer_response_id": record.get("safer_response_id"),
            "response_0_severity_level": record.get("response_0_severity_level"),
            "response_1_severity_level": record.get("response_1_severity_level"),
            "prompt_source": record.get("prompt_source"),
            "response_0_source": record.get("response_0_source"),
            "response_1_source": record.get("response_1_source"),
            "response_0_sha256": record.get("response_0_sha256"),
            "response_1_sha256": record.get("response_1_sha256"),
        },
        "provenance": provenance,
    }


def ultrachat_candidate(record: dict[str, Any], revision: str, split: str, prefix: str) -> dict[str, Any] | None:
    exchange = extract_first_exchange(record)
    if exchange is None:
        return None
    prompt, response = exchange
    if len(prompt.split()) < 3 or len(response.split()) < 5:
        return None
    provenance = {
        "dataset_name": "UltraChat 200k",
        "source": f"https://huggingface.co/datasets/{ULTRACHAT_REPO}",
        "revision_or_version": revision,
        "license": "MIT",
        "original_split": split,
        "selection_criteria": "first non-empty user/assistant exchange with minimum text length",
        "transformations": "reduced conversation to its first user/assistant exchange",
        "final_category": None,
    }
    return {
        "id": stable_id(prefix, prompt),
        "prompt": prompt,
        "response": response,
        "source_record_id": record.get("prompt_id"),
        "final_category": None,
        "provenance": provenance,
    }


def _shuffled_take(dataset: Any, seed: int, limit: int, multiplier: int = 8) -> Iterator[dict[str, Any]]:
    shuffled = dataset.shuffle(seed=seed, buffer_size=max(1_000, limit * multiplier))
    yield from shuffled.take(limit * multiplier)


def _unique_candidates(records: Iterable[dict[str, Any] | None], limit: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if record is None or record["id"] in seen:
            continue
        output.append(record)
        seen.add(record["id"])
        if len(output) >= limit:
            break
    if len(output) < limit:
        raise RuntimeError(f"Only found {len(output)} valid unique records; requested {limit}")
    return output


def _github_csv(owner: str, repo: str, path: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    import requests

    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(f"https://api.github.com/repos/{owner}/{repo}/commits/main", headers=headers, timeout=60)
    response.raise_for_status()
    revision = response.json()["sha"]
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{revision}/{path}"
    data_response = requests.get(url, timeout=120)
    data_response.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(data_response.text)))
    return rows, {"revision": revision, "url": url}


def _harmbench_candidates(limit: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows, source = _github_csv(
        "centerforaisafety", "HarmBench", "data/behavior_datasets/harmbench_behaviors_text_test.csv"
    )
    random.Random(seed).shuffle(rows)
    output = []
    for row in rows:
        behavior = str(row.get("Behavior") or "").strip()
        context = str(row.get("ContextString") or "").strip()
        if not behavior:
            continue
        prompt = f"{context}\n\n{behavior}" if context else behavior
        output.append(
            {
                "id": stable_id("harmbench", prompt),
                "prompt": prompt,
                "source_label": "harmful_behavior",
                "source_record_id": row.get("BehaviorID"),
                "provenance": {
                    "dataset_name": "HarmBench",
                    "source": source["url"],
                    "revision_or_version": source["revision"],
                    "license": "MIT",
                    "original_split": "text_test",
                    "selection_criteria": "randomized unique text behavior candidate",
                    "transformations": "ContextString prepended when present; Behavior mapped to final user request",
                    "final_category": "unsafe",
                },
            }
        )
        if len(output) >= limit:
            break
    return output, source


def _xstest_candidates(limit: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows, source = _github_csv("paul-rottger", "xstest", "xstest_prompts.csv")
    safe_rows = [
        row
        for row in rows
        if str(row.get("label") or "").casefold() == "safe"
        and not str(row.get("type") or "").startswith("contrast_")
    ]
    random.Random(seed).shuffle(safe_rows)
    output = []
    for row in safe_rows[:limit]:
        prompt = str(row.get("prompt") or "").strip()
        if not prompt:
            continue
        output.append(
            {
                "id": stable_id("xstest", prompt),
                "prompt": prompt,
                "source_label": row.get("type"),
                "source_record_id": row.get("id"),
                "provenance": {
                    "dataset_name": "XSTest",
                    "source": source["url"],
                    "revision_or_version": source["revision"],
                    "license": "CC-BY-4.0",
                    "original_split": "safe prompts",
                    "selection_criteria": "source label is safe and type does not begin with contrast_",
                    "transformations": "prompt column mapped to prompt",
                    "final_category": "safe",
                },
            }
        )
    return output, source


def acquire_public_candidates(
    output_dir: str | Path,
    train_examples: int,
    eval_examples: int,
    seed: int,
    overwrite: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    targets = {
        "m1": output_dir / "m1_candidates.jsonl",
        "safety": output_dir / "safety_candidates.jsonl",
        "benign_eval": output_dir / "benign_eval_candidates.jsonl",
        "harmful_eval": output_dir / "harmful_eval_candidates.jsonl",
        "overrefusal_eval": output_dir / "overrefusal_eval_candidates.jsonl",
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite public-data candidates: {existing}")
    from datasets import load_dataset
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    ultrachat_revision = api.dataset_info(ULTRACHAT_REPO).sha
    pku_revision = api.dataset_info(PKU_REPO).sha
    candidate_count = max(train_examples * 3, train_examples + eval_examples * 2)
    ultrachat_train = load_dataset(
        ULTRACHAT_REPO, split="train_sft", revision=ultrachat_revision, streaming=True, token=token
    )
    ultrachat_test = load_dataset(
        ULTRACHAT_REPO, split="test_sft", revision=ultrachat_revision, streaming=True, token=token
    )
    pku_train = load_dataset(PKU_REPO, split="train", revision=pku_revision, streaming=True, token=token)
    m1 = _unique_candidates(
        (
            ultrachat_candidate(row, ultrachat_revision, "train_sft", "ultrachat_train")
            for row in _shuffled_take(ultrachat_train, seed, candidate_count)
        ),
        candidate_count,
    )
    benign_eval = _unique_candidates(
        (
            ultrachat_candidate(row, ultrachat_revision, "test_sft", "ultrachat_test")
            for row in _shuffled_take(ultrachat_test, seed + 1, eval_examples)
        ),
        eval_examples,
    )
    safety = _unique_candidates(
        (pku_candidate(row, pku_revision) for row in _shuffled_take(pku_train, seed + 2, candidate_count)),
        candidate_count,
    )
    harmful_eval, harmbench_source = _harmbench_candidates(eval_examples, seed + 3)
    overrefusal_eval, xstest_source = _xstest_candidates(eval_examples, seed + 4)
    if len(harmful_eval) != eval_examples:
        raise RuntimeError(f"HarmBench supplied {len(harmful_eval)} unique candidates; requested {eval_examples}")
    if len(overrefusal_eval) != eval_examples:
        raise RuntimeError(f"XSTest supplied {len(overrefusal_eval)} safe candidates; requested {eval_examples}")
    values = {
        "m1": m1,
        "safety": safety,
        "benign_eval": benign_eval,
        "harmful_eval": harmful_eval,
        "overrefusal_eval": overrefusal_eval,
    }
    for name, path in targets.items():
        atomic_write_jsonl(path, values[name])
    manifest = {
        "code_version": PUBLIC_DATA_CODE_VERSION,
        "seed": seed,
        "requested_train_examples": train_examples,
        "requested_eval_examples": eval_examples,
        "labeling_mode": "binary_source_trust_without_human_review",
        "all_records_require_review": False,
        "ambiguous_category_omitted": True,
        "ambiguous_examples_not_independently_identified": True,
        "sources": {
            "ultrachat": {"repo_id": ULTRACHAT_REPO, "revision": ultrachat_revision, "license": "MIT"},
            "pku_safe_rlhf": {"repo_id": PKU_REPO, "revision": pku_revision, "license": "CC-BY-NC-4.0"},
            "harmbench": {**harmbench_source, "license": "MIT"},
            "xstest": {**xstest_source, "license": "CC-BY-4.0"},
        },
        "candidate_files": {
            name: {"path": str(path), "records": len(values[name]), "sha256": sha256_file(path)}
            for name, path in targets.items()
        },
    }
    manifest_path = output_dir / "source_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def _with_final_provenance(record: dict[str, Any], category: str, transformation: str) -> dict[str, Any]:
    provenance = dict(record["provenance"])
    provenance["final_category"] = category
    provenance["transformations"] = f"{provenance['transformations']}; {transformation}"
    return provenance


def _normal_prompt(prompt: str) -> str:
    return " ".join(prompt.casefold().split())


def finalize_trusted_candidates(
    candidate_dir: str | Path,
    repo_root: str | Path,
    overwrite: bool = False,
) -> Path:
    candidate_dir, repo_root = Path(candidate_dir), Path(repo_root)
    source_manifest_path = candidate_dir / "source_manifest.json"
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"Public source manifest missing: {source_manifest_path}")
    import json

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    expected_train = int(source_manifest["requested_train_examples"])
    expected_eval = int(source_manifest["requested_eval_examples"])
    candidates = {
        name: read_jsonl(candidate_dir / f"{name}_candidates.jsonl")
        for name in ("m1", "safety", "benign_eval", "harmful_eval", "overrefusal_eval")
    }
    safe_train_count = (expected_train + 1) // 2
    unsafe_train_count = expected_train - safe_train_count
    required_m1 = expected_train + safe_train_count
    if len(candidates["m1"]) < required_m1:
        raise ValueError(
            f"Need {required_m1} UltraChat candidates for disjoint M1 and safe M2/M3 records; "
            f"found {len(candidates['m1'])}"
        )
    unsafe_pairs = [
        record
        for record in candidates["safety"]
        if bool(record.get("source_labels", {}).get("response_0_safe"))
        != bool(record.get("source_labels", {}).get("response_1_safe"))
    ]
    if len(unsafe_pairs) < unsafe_train_count:
        raise ValueError(
            f"Need {unsafe_train_count} PKU records with one source-labeled safe and one unsafe response; "
            f"found {len(unsafe_pairs)}. Reacquire with a larger candidate pool."
        )
    for name in ("benign_eval", "harmful_eval", "overrefusal_eval"):
        if len(candidates[name]) < expected_eval:
            raise ValueError(f"Need {expected_eval} {name} candidates; found {len(candidates[name])}")
    m1_source = candidates["m1"][:expected_train]
    safe_source = candidates["m1"][expected_train:required_m1]
    unsafe_source = unsafe_pairs[:unsafe_train_count]
    m1 = [
        {
            "id": record["id"],
            "prompt": record["prompt"],
            "response": record["response"],
            "provenance": _with_final_provenance(
                record, "safe", "automatically selected for M1 under the source-trust policy; not human-reviewed"
            ),
        }
        for record in m1_source
    ]
    shared, direct, dpo = [], [], []
    for record in safe_source:
        record_id = stable_id("binary_safe", record["prompt"])
        provenance = _with_final_provenance(
            record,
            "safe",
            "automatically mapped from filtered UltraChat to safe; not independently human-reviewed",
        )
        shared_record = {
            "id": record_id,
            "prompt": record["prompt"],
            "category": "safe",
            "provenance": provenance,
        }
        shared.append(shared_record)
        direct.append({**shared_record, "response": record["response"], "direct_target_provenance": provenance})
    for record in unsafe_source:
        provenance = _with_final_provenance(
            record,
            "unsafe",
            "automatically mapped to unsafe because exactly one paired response is source-labeled unsafe; "
            "the source-labeled safe response is the direct target; not independently human-reviewed",
        )
        shared_record = {
            "id": record["id"],
            "prompt": record["prompt"],
            "category": "unsafe",
            "provenance": provenance,
        }
        shared.append(shared_record)
        direct.append({**shared_record, "response": record["response"], "direct_target_provenance": provenance})
        dpo.append(
            {
                "id": f"dpo_{record['id']}",
                "prompt": record["prompt"],
                "chosen": record["response"],
                "rejected": record["rejected"],
                "provenance": provenance,
            }
        )
    evaluations: dict[str, list[dict[str, Any]]] = {}
    evaluation_sources = {
        "benign_utility": candidates["benign_eval"][:expected_eval],
        "harmful": candidates["harmful_eval"][:expected_eval],
        "overrefusal": candidates["overrefusal_eval"][:expected_eval],
    }
    for suite, records in evaluation_sources.items():
        category = "unsafe" if suite == "harmful" else "safe"
        evaluations[suite] = [
            {
                "id": record["id"],
                "prompt": record["prompt"],
                "provenance": _with_final_provenance(
                    record, category, f"automatically selected for {suite} under the source-trust policy"
                ),
            }
            for record in records
        ]
    train_prompts = {_normal_prompt(record["prompt"]) for record in m1 + shared}
    eval_prompts = {_normal_prompt(record["prompt"]) for records in evaluations.values() for record in records}
    overlap = train_prompts & eval_prompts
    if overlap:
        raise ValueError(f"Exact normalized prompt leakage between training and evaluation: {len(overlap)} prompts")
    validate_records(m1, "M1")
    validate_records(direct, "M2")
    validate_records(dpo, "M4")
    for suite, records in evaluations.items():
        validate_eval_records(records, suite)
    category_counts = {category: sum(record["category"] == category for record in shared) for category in SAFETY_CATEGORIES}
    if any(count == 0 for count in category_counts.values()):
        raise ValueError(f"The binary safety training set must include both categories: {category_counts}")
    targets = {
        repo_root / "data" / "benign_control" / "train.jsonl": m1,
        repo_root / "data" / "safety_shared" / "prompts.jsonl": shared,
        repo_root / "data" / "safety_direct" / "train.jsonl": direct,
        repo_root / "data" / "dpo" / "train.jsonl": dpo,
        repo_root / "data" / "eval" / "harmful_test.jsonl": evaluations["harmful"],
        repo_root / "data" / "eval" / "benign_utility_test.jsonl": evaluations["benign_utility"],
        repo_root / "data" / "eval" / "overrefusal_test.jsonl": evaluations["overrefusal"],
    }
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing experimental files: {existing}")
    for path, records in targets.items():
        atomic_write_jsonl(path, records)
    manifest = {
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "preparation_mode": "binary_source_trust_without_human_review",
        "human_review_completed": False,
        "source_labels_trusted_without_independent_validation": True,
        "ambiguous_category_omitted": True,
        "ambiguous_examples_not_independently_identified": True,
        "automatic_mapping_rules": {
            "safe_training": "filtered UltraChat records, disjoint from M1 records",
            "unsafe_training": "PKU-SafeRLHF pairs with exactly one safe and one unsafe response",
            "unsafe_target": "the source-labeled safe PKU response",
            "dpo_scope": "unsafe PKU pairs only",
        },
        "training_examples_per_condition": expected_train,
        "evaluation_examples_per_suite": expected_eval,
        "safety_category_distribution": category_counts,
        "dpo_examples": len(dpo),
        "outputs": {
            str(path): {"records": len(records), "sha256": sha256_file(path)} for path, records in targets.items()
        },
        "source_candidate_hashes": {
            name: sha256_file(candidate_dir / f"{name}_candidates.jsonl") for name in candidates
        },
    }
    destination = repo_root / "data" / "preparation_manifest.json"
    atomic_write_json(destination, manifest)
    return destination
