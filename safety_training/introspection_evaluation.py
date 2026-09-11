from __future__ import annotations

import copy
import gc
import json
from pathlib import Path
from typing import Any

from .config import resolve_path
from .formatting import serialize_prompt
from .introspection_config import lock_to_m0, stage_dir
from .introspection_data import (
    load_organisms,
    load_questions,
    sha256_tree,
    validate_organisms_are_not_condition_targets,
)
from .introspection_modeling import BehaviorAdapterCache, activate_adapters, load_introspection_model
from .io import atomic_write_json, atomic_write_jsonl, sha256_file
from .modeling import configure_hf_cache, load_tokenizer
from .seeding import set_all_seeds


def _generate(model: Any, tokenizer: Any, prompt: str, generation: dict[str, Any]) -> str:
    import torch

    text = serialize_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    kwargs = {key: value for key, value in generation.items() if value is not None}
    kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
    kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)
    with torch.no_grad():
        output = model.generate(**inputs, **kwargs)
    tokens = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(tokens, skip_special_tokens=True).strip()


def _target(
    root: Path,
    condition: str,
    seed: int | None,
) -> tuple[str, Path | None, Path]:
    if condition == "M0":
        return "M0", None, root / "M0" / "manifest.json"
    if condition not in {"M1", "M2", "M3", "M4"}:
        raise ValueError(f"Unknown target condition {condition!r}")
    if seed is None:
        raise ValueError(f"{condition} requires a seed")
    run_dir = root / condition / f"seed_{int(seed)}"
    return f"{condition}_seed_{int(seed)}", run_dir / "adapter", run_dir / "manifest.json"


def _verify_target(
    config: dict[str, Any],
    model_id: str,
    adapter_path: Path | None,
    manifest_path: Path,
) -> dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Target manifest missing for {model_id}: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("starting_model") != config["model_name"]:
        raise ValueError(f"{model_id}: target base model differs from the IA base")
    revision = manifest.get("model_commit_hash") if model_id == "M0" else manifest.get("starting_revision")
    if revision != config["model_revision"]:
        raise ValueError(f"{model_id}: target revision differs from the IA base revision")
    if adapter_path is not None and not adapter_path.is_dir():
        raise FileNotFoundError(f"Target adapter missing for {model_id}: {adapter_path}")
    return manifest


def evaluate_introspection(
    config: dict[str, Any],
    *,
    ia_stage: str,
    conditions: list[str],
    seeds: list[int],
    questions_path: str | Path,
    output_root: str | Path | None = None,
    hf_home: str | Path | None = None,
    include_baseline: bool = True,
) -> Path:
    if ia_stage not in {"sft", "dpo"}:
        raise ValueError("ia_stage must be 'sft' or 'dpo'")
    config, m0_manifest_path = lock_to_m0(config, output_root)
    configure_hf_cache(config, hf_home)
    root = resolve_path(output_root or config["paths"]["output_root"])
    ia_dir = stage_dir(config, ia_stage, output_root)
    ia_adapter = ia_dir / "adapter"
    ia_manifest_path = ia_dir / "manifest.json"
    if not ia_adapter.is_dir() or not ia_manifest_path.exists():
        raise FileNotFoundError(f"Completed {ia_stage} IA not found: {ia_dir}")
    ia_manifest = json.loads(ia_manifest_path.read_text(encoding="utf-8"))
    if ia_manifest.get("base_model") != config["model_name"] or ia_manifest.get("base_revision") != config["model_revision"]:
        raise ValueError("IA manifest does not match the locked M0 base")
    questions_file = resolve_path(questions_path)
    questions = load_questions(questions_file)
    targets: list[tuple[str, str, int | None, Path | None, Path]] = []
    for condition in conditions:
        condition_seeds: list[int | None] = [None] if condition == "M0" else list(seeds)
        for seed in condition_seeds:
            model_id, adapter_path, manifest_path = _target(root, condition, seed)
            targets.append((model_id, condition, seed, adapter_path, manifest_path))
    output = ia_dir / "evaluations"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    target_hashes: dict[str, Any] = {}
    tokenizer = load_tokenizer(config)
    for model_index, (model_id, condition, seed, adapter_path, manifest_path) in enumerate(targets):
        _verify_target(config, model_id, adapter_path, manifest_path)
        set_all_seeds(int(config["seed"]) + model_index)
        target_config = copy.deepcopy(config)
        model = load_introspection_model(target_config, adapter_path=ia_adapter, trainable=False)
        cache = BehaviorAdapterCache(model, 1)
        behavior_name = cache.ensure(adapter_path)
        model.eval()
        for question in questions:
            if include_baseline:
                if behavior_name is None:
                    with model.disable_adapter():
                        response = _generate(model, tokenizer, question["prompt"], config["generation"])
                else:
                    activate_adapters(
                        model,
                        behavior_name,
                        include_introspection=False,
                        train_introspection=False,
                    )
                    response = _generate(model, tokenizer, question["prompt"], config["generation"])
                rows.append(
                    {
                        "report_id": f"{model_id}__no_ia__{question['id']}",
                        "model_id": model_id,
                        "condition": condition,
                        "seed": seed,
                        "question_id": question["id"],
                        "prompt": question["prompt"],
                        "response": response,
                        "introspection_adapter": None,
                        "adapter_stack": [condition] if condition != "M0" else [],
                    }
                )
            activate_adapters(
                model,
                behavior_name,
                include_introspection=True,
                train_introspection=False,
            )
            response = _generate(model, tokenizer, question["prompt"], config["generation"])
            rows.append(
                {
                    "report_id": f"{model_id}__{ia_stage}__{question['id']}",
                    "model_id": model_id,
                    "condition": condition,
                    "seed": seed,
                    "question_id": question["id"],
                    "prompt": question["prompt"],
                    "response": response,
                    "introspection_adapter": ia_stage,
                    "adapter_stack": ([condition] if condition != "M0" else []) + [f"IA_{ia_stage}"],
                }
            )
        target_hashes[model_id] = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "adapter_path": str(adapter_path) if adapter_path else None,
            "adapter_sha256": sha256_tree(adapter_path) if adapter_path else None,
        }
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    responses_path = output / "raw_reports.jsonl"
    atomic_write_jsonl(responses_path, rows)
    atomic_write_json(
        output / "manifest.json",
        {
            "manifest_version": "introspection-evaluation-1.0",
            "ia_stage": ia_stage,
            "ia_adapter_path": str(ia_adapter),
            "ia_adapter_sha256": sha256_tree(ia_adapter),
            "ia_manifest_sha256": sha256_file(ia_manifest_path),
            "m0_manifest_sha256": sha256_file(m0_manifest_path),
            "questions_path": str(questions_file),
            "questions_sha256": sha256_file(questions_file),
            "include_baseline": include_baseline,
            "target_hashes": target_hashes,
            "response_count": len(rows),
            "responses_path": str(responses_path),
            "comparison_warning": (
                "Game runs must use the target adapter without the introspection adapter. "
                "Normalize reports only after all audit channels are frozen."
            ),
        },
    )
    return responses_path


def evaluate_heldout_organisms(
    config: dict[str, Any],
    *,
    ia_stage: str,
    questions_path: str | Path,
    output_root: str | Path | None = None,
    hf_home: str | Path | None = None,
    include_baseline: bool = True,
    organism_split: str = "eval",
) -> Path:
    if ia_stage not in {"sft", "dpo"}:
        raise ValueError("ia_stage must be 'sft' or 'dpo'")
    config, m0_manifest_path = lock_to_m0(config, output_root)
    configure_hf_cache(config, hf_home)
    root = resolve_path(output_root or config["paths"]["output_root"])
    ia_dir = stage_dir(config, ia_stage, output_root)
    ia_adapter = ia_dir / "adapter"
    ia_manifest_path = ia_dir / "manifest.json"
    if not ia_adapter.is_dir() or not ia_manifest_path.exists():
        raise FileNotFoundError(f"Completed {ia_stage} IA not found: {ia_dir}")
    ia_manifest = json.loads(ia_manifest_path.read_text(encoding="utf-8"))
    if ia_manifest.get("base_model") != config["model_name"] or ia_manifest.get("base_revision") != config["model_revision"]:
        raise ValueError("IA manifest does not match the locked M0 base")
    organism_path = resolve_path(config["introspection"]["organism_manifest"])
    organisms = load_organisms(
        organism_path,
        split=organism_split,
        expected_model=config["model_name"],
        expected_revision=config["model_revision"],
    )
    validate_organisms_are_not_condition_targets(organisms, root)
    questions_file = resolve_path(questions_path)
    questions = load_questions(questions_file)
    output = ia_dir / "evaluations" / f"{organism_split}_organisms"
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(config)
    model = load_introspection_model(config, adapter_path=ia_adapter, trainable=False)
    cache = BehaviorAdapterCache(
        model,
        int(config["introspection"][ia_stage].get("max_loaded_behavior_adapters", 8)),
    )
    rows: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    adapter_hashes: dict[str, str | None] = {}
    model.eval()
    for index, organism in enumerate(organisms):
        set_all_seeds(int(config["seed"]) + index)
        behavior_name = cache.ensure(organism["_adapter_path"])
        labels.append(
            {
                "organism_id": organism["organism_id"],
                "behavior_family": organism["behavior_family"],
                "introspection_target": organism["introspection_target"],
            }
        )
        adapter_hashes[organism["organism_id"]] = organism.get("_adapter_sha256")
        for question in questions:
            if include_baseline:
                if behavior_name is None:
                    with model.disable_adapter():
                        response = _generate(model, tokenizer, question["prompt"], config["generation"])
                else:
                    activate_adapters(
                        model,
                        behavior_name,
                        include_introspection=False,
                        train_introspection=False,
                    )
                    response = _generate(model, tokenizer, question["prompt"], config["generation"])
                rows.append(
                    {
                        "report_id": f"{organism['organism_id']}__no_ia__{question['id']}",
                        "organism_id": organism["organism_id"],
                        "question_id": question["id"],
                        "prompt": question["prompt"],
                        "response": response,
                        "introspection_adapter": None,
                    }
                )
            activate_adapters(
                model,
                behavior_name,
                include_introspection=True,
                train_introspection=False,
            )
            rows.append(
                {
                    "report_id": f"{organism['organism_id']}__{ia_stage}__{question['id']}",
                    "organism_id": organism["organism_id"],
                    "question_id": question["id"],
                    "prompt": question["prompt"],
                    "response": _generate(model, tokenizer, question["prompt"], config["generation"]),
                    "introspection_adapter": ia_stage,
                }
            )
    reports_path = output / "raw_reports.jsonl"
    labels_path = output / "ground_truth_labels.jsonl"
    atomic_write_jsonl(reports_path, rows)
    atomic_write_jsonl(labels_path, labels)
    atomic_write_json(
        output / "manifest.json",
        {
            "manifest_version": "introspection-heldout-evaluation-1.0",
            "ia_stage": ia_stage,
            "organism_split": organism_split,
            "ia_adapter_sha256": sha256_tree(ia_adapter),
            "ia_manifest_sha256": sha256_file(ia_manifest_path),
            "m0_manifest_sha256": sha256_file(m0_manifest_path),
            "organism_manifest_sha256": sha256_file(organism_path),
            "questions_sha256": sha256_file(questions_file),
            "adapter_hashes": adapter_hashes,
            "organism_count": len(organisms),
            "report_count": len(rows),
            "include_baseline": include_baseline,
            "raw_reports_path": str(reports_path),
            "ground_truth_labels_path": str(labels_path),
        },
    )
    return reports_path
