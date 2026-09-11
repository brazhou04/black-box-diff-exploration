from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from safety_training.io import atomic_write_json, atomic_write_jsonl, read_jsonl, sha256_file

from .tournament import derive_seed


EXCLUDED_METRICS = {"round_count", "coordinated_transition_count"}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _aligned_name(name: str, focal_player: int) -> str:
    if focal_player not in {1, 2}:
        raise ValueError(f"Invalid focal player {focal_player!r}")
    player1 = "focal" if focal_player == 1 else "baseline"
    player2 = "baseline" if focal_player == 1 else "focal"
    return (
        name.replace("player1", "__PLAYER1__")
        .replace("player2", "__PLAYER2__")
        .replace("__PLAYER1__", player1)
        .replace("__PLAYER2__", player2)
    )


def _aligned_metrics(metrics: dict[str, Any], focal_player: int) -> dict[str, float]:
    return {
        _aligned_name(name, focal_player): float(value)
        for name, value in metrics.items()
        if name not in EXCLUDED_METRICS
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }


def _paired_record(
    intervention: dict[str, Any], control: dict[str, Any]
) -> dict[str, Any]:
    focal_player = int(intervention["focal_player"])
    focal_agent = str(intervention["focal_agent_id"])
    if intervention["game"] != control["game"]:
        raise ValueError("Paired episodes have different games")
    if intervention["run_index"] != control["run_index"]:
        raise ValueError("Paired episodes have different run indices")
    if intervention["prompt_variant_id"] != control["prompt_variant_id"]:
        raise ValueError("Paired episodes have different prompt variants")
    expected_focal = control["player1_id"] if focal_player == 1 else control["player2_id"]
    expected_baseline = control["player2_id"] if focal_player == 1 else control["player1_id"]
    if expected_focal != focal_agent or expected_baseline != "M0":
        raise ValueError(
            f"Control {control['episode_id']} is not {focal_agent} versus M0 in the expected seats"
        )

    prompted_metrics = _aligned_metrics(intervention["metrics"], focal_player)
    control_metrics = _aligned_metrics(control["metrics"], focal_player)
    metric_names = sorted(set(prompted_metrics) & set(control_metrics))
    condition_key = "player1_condition" if focal_player == 1 else "player2_condition"
    seed_key = "player1_training_seed" if focal_player == 1 else "player2_training_seed"
    return {
        "intervention_episode_id": intervention["episode_id"],
        "control_episode_id": control["episode_id"],
        "game": intervention["game"],
        "run_index": intervention["run_index"],
        "prompt_variant_id": intervention["prompt_variant_id"],
        "focal_player": focal_player,
        "focal_agent_id": focal_agent,
        "focal_condition": control[condition_key],
        "focal_training_seed": control[seed_key],
        "metrics": {
            name: {
                "unprompted": control_metrics[name],
                "prompted": prompted_metrics[name],
                "paired_delta": prompted_metrics[name] - control_metrics[name],
            }
            for name in metric_names
        },
    }


def _aggregate_pairs(
    rows: list[dict[str, Any]], group_id: str, bootstrap_samples: int
) -> dict[str, Any]:
    metric_names = sorted({name for row in rows for name in row["metrics"]})
    metrics: dict[str, Any] = {}
    rng = random.Random(derive_seed(20250923, group_id, "paired_episode_bootstrap"))
    for metric in metric_names:
        values = [row["metrics"][metric] for row in rows if metric in row["metrics"]]
        if not values:
            continue
        deltas = [float(value["paired_delta"]) for value in values]
        clusters: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if metric in row["metrics"]:
                clusters[row["control_episode_id"]].append(
                    float(row["metrics"][metric]["paired_delta"])
                )
        cluster_values = list(clusters.values())
        bootstrapped = []
        for _ in range(bootstrap_samples):
            sampled_clusters = rng.choices(cluster_values, k=len(cluster_values))
            bootstrapped.append(mean(value for cluster in sampled_clusters for value in cluster))
        metrics[metric] = {
            "unprompted_mean": mean(float(value["unprompted"]) for value in values),
            "prompted_mean": mean(float(value["prompted"]) for value in values),
            "mean_paired_delta": mean(deltas),
            "pair_count": len(values),
            "control_episode_cluster_count": len(cluster_values),
            "paired_control_episode_cluster_bootstrap_95_ci": [
                _percentile(bootstrapped, 0.025),
                _percentile(bootstrapped, 0.975),
            ],
        }
    return {"pair_count": len(rows), "metrics": metrics}


def _grouped(
    rows: list[dict[str, Any]], fields: tuple[str, ...], bootstrap_samples: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    return [
        {
            **dict(zip(fields, key)),
            **_aggregate_pairs(group, "|".join(map(str, key)), bootstrap_samples),
        }
        for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0])))
    ]


def analyze_prompt_intervention(
    control_root: str | Path,
    intervention_root: str | Path,
    *,
    bootstrap_samples: int = 1000,
) -> Path:
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    control_path = Path(control_root) / "episodes.jsonl"
    intervention_path = Path(intervention_root) / "episodes.jsonl"
    controls = {row["episode_id"]: row for row in read_jsonl(control_path)}
    interventions = read_jsonl(intervention_path)
    if not interventions:
        raise ValueError(f"No intervention episodes found at {intervention_path}")
    missing = sorted(
        {
            row.get("paired_control_episode_id")
            for row in interventions
            if row.get("paired_control_episode_id") not in controls
        }
    )
    if missing:
        preview = ", ".join(str(value) for value in missing[:5])
        raise ValueError(
            f"{len(missing)} paired control episodes are missing from {control_path}: {preview}"
        )
    pairs = [
        _paired_record(row, controls[row["paired_control_episode_id"]])
        for row in interventions
    ]
    pair_path = Path(intervention_root) / "paired_episode_effects.jsonl"
    atomic_write_jsonl(pair_path, pairs)
    payload = {
        "schema_version": "1.0",
        "design": "one prompted focal player versus the same unprompted M0 baseline",
        "control_episodes_path": str(control_path),
        "control_episodes_sha256": sha256_file(control_path),
        "intervention_episodes_path": str(intervention_path),
        "intervention_episodes_sha256": sha256_file(intervention_path),
        "paired_episode_effects_path": str(pair_path),
        "paired_episode_effects_sha256": sha256_file(pair_path),
        "paired_episode_count": len(pairs),
        "bootstrap_samples": bootstrap_samples,
        "effect_definition": "paired_delta = prompted focal episode metric - matched unprompted episode metric",
        "inference_caveat": (
            "Intervals resample matched episodes. Training-seed replication, not the number of game "
            "episodes, remains the inferential limit for model-condition effects."
        ),
        "by_focal_agent_game_and_seat": _grouped(
            pairs,
            ("focal_agent_id", "game", "focal_player"),
            bootstrap_samples,
        ),
        "by_focal_agent_and_game": _grouped(
            pairs, ("focal_agent_id", "game"), bootstrap_samples
        ),
        "by_condition_game_and_seat": _grouped(
            pairs,
            ("focal_condition", "game", "focal_player"),
            bootstrap_samples,
        ),
        "by_condition_and_game": _grouped(
            pairs, ("focal_condition", "game"), bootstrap_samples
        ),
    }
    destination = Path(intervention_root) / "paired_analysis.json"
    atomic_write_json(destination, payload)
    return destination
