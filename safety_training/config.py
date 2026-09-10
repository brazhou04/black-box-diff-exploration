from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_CONDITIONS = {"M0", "M1", "M2", "M3", "M4"}
DEFAULT_SEEDS = (42, 123, 456)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def load_config(path: str | Path, seed: int | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    raw = load_yaml(path)
    parent = raw.pop("extends", None)
    config = _deep_merge(load_yaml(path.parent / parent), raw) if parent else raw
    if seed is not None:
        config["seed"] = int(seed)
    validate_config(config)
    config["_config_path"] = str(path)
    return config


def validate_config(config: dict[str, Any]) -> None:
    condition = config.get("condition")
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(VALID_CONDITIONS)}, got {condition!r}")
    required_sections = ("training", "peft", "quantization", "checkpointing", "paths")
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Missing shared config sections: {missing}")
    if config["peft"].get("method") != "lora":
        raise ValueError("This implementation currently supports only LoRA PEFT")
    if int(config["training"]["per_device_batch_size"]) < 1:
        raise ValueError("per_device_batch_size must be positive")
    if int(config["training"]["max_seq_length"]) < 32:
        raise ValueError("max_seq_length is implausibly small")


def resolve_path(value: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve repository-relative paths while preserving absolute Kaggle overrides."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (repo_root / path).resolve()


def condition_config_path(condition: str) -> Path:
    names = {
        "M1": "m1_benign.yaml",
        "M2": "m2_safety_sft.yaml",
        "M3": "m3_constitutional.yaml",
        "M4": "m4_dpo.yaml",
    }
    if condition not in names:
        raise ValueError(f"No training config for {condition}")
    return REPO_ROOT / "configs" / names[condition]


def artifact_dir(config: dict[str, Any], output_root: str | Path | None = None) -> Path:
    root = resolve_path(output_root or config["paths"]["output_root"])
    condition = config["condition"]
    return root / condition / f"seed_{int(config['seed'])}"

