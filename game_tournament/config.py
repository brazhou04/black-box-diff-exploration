from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from safety_training.config import REPO_ROOT, VALID_CONDITIONS, load_yaml, resolve_path
from safety_training.io import sha256_json

from .definitions import get_game_specs
from .prompting import build_prompt_variants


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_tournament_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(path)
    validate_tournament_config(config)
    config["_config_path"] = str(Path(path).resolve())
    return config


def with_overrides(
    config: dict[str, Any],
    *,
    rounds: int | None = None,
    runs: int | None = None,
    conditions: list[str] | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    if rounds is not None:
        updated["experiment"]["rounds_per_episode"] = int(rounds)
    if runs is not None:
        updated["experiment"]["runs_per_ordered_matchup"] = int(runs)
    if conditions is not None:
        updated["agents"]["conditions"] = list(conditions)
    if seeds is not None:
        updated["agents"]["seeds"] = [int(seed) for seed in seeds]
    validate_tournament_config(updated)
    return updated


def validate_tournament_config(config: dict[str, Any]) -> None:
    required = {"tournament_id", "agents", "experiment", "sampling", "prompting", "paths"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Tournament config missing sections: {missing}")
    tournament_id = config["tournament_id"]
    if not isinstance(tournament_id, str) or not tournament_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in tournament_id):
        raise ValueError("tournament_id may contain only letters, digits, underscores, and hyphens")

    conditions = list(config["agents"].get("conditions", []))
    unknown = sorted(set(conditions) - VALID_CONDITIONS)
    if unknown or not conditions:
        raise ValueError(f"Invalid or empty agent conditions: {unknown}")
    if len(conditions) != len(set(conditions)):
        raise ValueError("Agent conditions must be unique")
    seeds = list(config["agents"].get("seeds", []))
    if any(not isinstance(seed, int) for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("Agent seeds must be unique integers")
    if any(condition != "M0" for condition in conditions) and not seeds:
        raise ValueError("At least one training seed is required for M1-M4")

    experiment = config["experiment"]
    if int(experiment.get("rounds_per_episode", 0)) < 1:
        raise ValueError("rounds_per_episode must be positive")
    if int(experiment.get("runs_per_ordered_matchup", 0)) < 1:
        raise ValueError("runs_per_ordered_matchup must be positive")
    if int(experiment.get("inference_batch_episodes", 1)) < 1:
        raise ValueError("inference_batch_episodes must be positive")
    get_game_specs(experiment.get("games", []))

    sampling = config["sampling"]
    if sampling.get("method") != "categorical":
        raise ValueError("Only constrained categorical action sampling is supported")
    temperature = sampling.get("temperature")
    if temperature is not None and float(temperature) <= 0:
        raise ValueError("temperature must be null (model default) or positive; temperature zero is not used")
    if not isinstance(sampling.get("master_seed"), int):
        raise ValueError("sampling.master_seed must be an integer")
    build_prompt_variants(config["prompting"])


def effective_temperature(config: dict[str, Any]) -> float:
    configured = config["sampling"].get("temperature")
    return 1.0 if configured is None else float(configured)


def config_for_manifest(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def tournament_config_hash(config: dict[str, Any]) -> str:
    return sha256_json(config_for_manifest(config))


def artifact_root(config: dict[str, Any], override: str | Path | None = None) -> Path:
    return resolve_path(override or config["paths"]["artifact_root"])


def tournament_dir(config: dict[str, Any], output_root: str | Path | None = None) -> Path:
    if output_root is not None:
        root = resolve_path(output_root) / "game_tournaments"
    else:
        root = resolve_path(config["paths"]["tournament_root"])
    return root / config["tournament_id"]
