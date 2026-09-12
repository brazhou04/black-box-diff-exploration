from __future__ import annotations

from pathlib import Path
from typing import Any

from safety_training.config import load_yaml, resolve_path
from safety_training.io import sha256_json


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_diffing_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(path)
    validate_diffing_config(config)
    config["_config_path"] = str(Path(path).resolve())
    return config


def validate_diffing_config(config: dict[str, Any]) -> None:
    required = {"experiment_id", "method", "investigator", "sampling", "paths"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Model-diffing config missing sections: {missing}")

    experiment_id = config["experiment_id"]
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("experiment_id must be a non-empty string")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in experiment_id):
        raise ValueError("experiment_id may contain only letters, digits, underscores, and hyphens")

    method = config["method"]
    for key in ("max_turns", "max_prompts_per_turn", "max_samples_per_prompt"):
        if int(method.get(key, 0)) < 1:
            raise ValueError(f"method.{key} must be positive")
    if int(method["max_prompts_per_turn"]) > 5:
        raise ValueError("method.max_prompts_per_turn cannot exceed the paper's limit of 5")
    if int(method["max_samples_per_prompt"]) > 5:
        raise ValueError("method.max_samples_per_prompt cannot exceed the paper's limit of 5")
    alpha = float(method.get("alpha", 0.0))
    if not 0.0 < alpha < 1.0:
        raise ValueError("method.alpha must lie strictly between zero and one")
    if int(method.get("min_validation_prompts", 0)) < 1:
        raise ValueError("method.min_validation_prompts must be positive")
    if not method.get("stateless_targets"):
        raise ValueError("Target conversations must remain stateless for this protocol")
    if not method.get("hide_target_reasoning"):
        raise ValueError("Target reasoning must be hidden from the investigator")

    sampling = config["sampling"]
    if not sampling.get("do_sample"):
        raise ValueError("Black-box replication sampling requires sampling to be enabled")
    if float(sampling.get("temperature", 0.0)) <= 0:
        raise ValueError("sampling.temperature must be positive")
    top_p = float(sampling.get("top_p", 0.0))
    if not 0.0 < top_p <= 1.0:
        raise ValueError("sampling.top_p must lie in (0, 1]")
    if int(sampling.get("max_new_tokens", 0)) < 1:
        raise ValueError("sampling.max_new_tokens must be positive")
    if not isinstance(sampling.get("master_seed"), int):
        raise ValueError("sampling.master_seed must be an integer")


def config_for_manifest(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def diffing_config_hash(config: dict[str, Any]) -> str:
    return sha256_json(config_for_manifest(config))


def artifact_root(config: dict[str, Any], override: str | Path | None = None) -> Path:
    return resolve_path(override or config["paths"]["artifact_root"])


def session_root(config: dict[str, Any], override: str | Path | None = None) -> Path:
    return resolve_path(override or config["paths"]["session_root"])

