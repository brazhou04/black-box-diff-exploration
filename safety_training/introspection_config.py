from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .config import load_yaml, resolve_path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_config_chain(path: Path, seen: set[Path]) -> tuple[dict[str, Any], Path | None]:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Circular introspection config inheritance at {resolved}")
    seen.add(resolved)
    raw = load_yaml(resolved)
    base_value = raw.pop("base_config", None)
    if not base_value:
        return raw, None
    base_path = Path(base_value)
    if not base_path.is_absolute():
        base_path = (resolved.parent / base_path).resolve()
    base, _ = _load_config_chain(base_path, seen)
    return _deep_merge(base, raw), base_path


def load_introspection_config(path: str | Path, seed: int | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config, base_path = _load_config_chain(config_path, set())
    if base_path is None:
        raise ValueError("Introspection config requires base_config")
    if seed is not None:
        config["seed"] = int(seed)
    config["_config_path"] = str(config_path)
    config["_base_config_path"] = str(base_path)
    validate_introspection_config(config)
    return config


def validate_introspection_config(config: dict[str, Any]) -> None:
    required = ("model_name", "quantization", "generation", "paths", "introspection")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Introspection config missing sections: {missing}")
    section = config["introspection"]
    for key in (
        "experiment_name",
        "organism_manifest",
        "sft_dataset",
        "dpo_dataset",
        "peft",
        "organism_training",
        "sft",
        "dpo",
    ):
        if key not in section:
            raise ValueError(f"introspection section missing {key!r}")
    if int(section["peft"].get("r", 0)) < 1:
        raise ValueError("introspection.peft.r must be positive")
    for stage in ("sft", "dpo"):
        values = section[stage]
        if int(values.get("batch_size", 0)) < 1:
            raise ValueError(f"introspection.{stage}.batch_size must be positive")
        if int(values.get("organisms_per_step", 0)) < 1:
            raise ValueError(f"introspection.{stage}.organisms_per_step must be positive")
        if int(values.get("max_seq_length", 0)) < 32:
            raise ValueError(f"introspection.{stage}.max_seq_length is implausibly small")
    if float(section["dpo"].get("beta", 0.0)) <= 0:
        raise ValueError("introspection.dpo.beta must be positive")


def introspection_root(config: dict[str, Any], output_root: str | Path | None = None) -> Path:
    root = resolve_path(output_root or config["paths"]["output_root"])
    experiment = config["introspection"]["experiment_name"]
    seed = int(config["seed"])
    return root / "introspection" / experiment / f"seed_{seed}"


def stage_dir(config: dict[str, Any], stage: str, output_root: str | Path | None = None) -> Path:
    if stage not in {"sft", "dpo"}:
        raise ValueError(f"Unknown introspection stage {stage!r}")
    return introspection_root(config, output_root) / stage


def lock_to_m0(config: dict[str, Any], output_root: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    locked = copy.deepcopy(config)
    root = resolve_path(output_root or locked["paths"]["output_root"])
    manifest_path = root / "M0" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"M0 revision lock missing: {manifest_path}. Create M0 before training or evaluating an IA."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("starting_model") != locked["model_name"]:
        raise ValueError("M0 manifest model does not match the introspection base model")
    revision = manifest.get("model_commit_hash") or manifest.get("starting_revision")
    if not revision:
        raise ValueError("M0 manifest has no resolved model revision")
    locked["requested_model_revision"] = locked.get("model_revision")
    locked["model_revision"] = revision
    locked["tokenizer_revision"] = manifest.get("tokenizer_commit_hash") or revision
    return locked, manifest_path
