from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from safety_training.io import atomic_write_json, atomic_write_jsonl, read_jsonl, sha256_file

from .tournament import derive_seed


IDENTITY_FIELDS = {
    "round_count",
    "total_payoff_player1",
    "total_payoff_player2",
    "coordinated_transition_count",
}

PAIRED_EXCLUDED_METRICS = {"round_count", "coordinated_transition_count"}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot take a percentile of an empty collection")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _aggregate_group(
    rows: list[dict[str, Any]], group_id: str, bootstrap_samples: int
) -> dict[str, Any]:
    metric_names = sorted(
        {
            name
            for row in rows
            for name, value in row["metrics"].items()
            if isinstance(value, (int, float)) and name not in IDENTITY_FIELDS
        }
    )
    aggregates: dict[str, Any] = {}
    rng = random.Random(derive_seed(20250923, group_id, "episode_bootstrap"))
    for metric in metric_names:
        values = [float(row["metrics"][metric]) for row in rows if row["metrics"].get(metric) is not None]
        if not values:
            continue
        estimate = mean(values)
        bootstrapped = [mean(rng.choices(values, k=len(values))) for _ in range(bootstrap_samples)]
        aggregates[metric] = {
            "mean": estimate,
            "episode_count": len(values),
            "episode_cluster_bootstrap_95_ci": [
                _percentile(bootstrapped, 0.025),
                _percentile(bootstrapped, 0.975),
            ],
        }
    return {"episode_count": len(rows), "metrics": aggregates}


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
        if name not in PAIRED_EXCLUDED_METRICS
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }


def _paired_record(
    intervention: dict[str, Any], control: dict[str, Any]
) -> dict[str, Any]:
    focal_player = int(intervention["focal_player"])
    focal_agent = str(intervention["focal_agent_id"])
    for field in ("game", "run_index", "prompt_variant_id"):
        if intervention[field] != control[field]:
            raise ValueError(f"Paired episodes have different {field}")
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
            bootstrapped.append(
                mean(value for cluster in sampled_clusters for value in cluster)
            )
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


def _group_pairs(
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


def _prompt_intervention_analysis(
    root: Path,
    control_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    controls = {row["episode_id"]: row for row in control_rows}
    missing = sorted(
        {
            row.get("paired_control_episode_id")
            for row in intervention_rows
            if row.get("paired_control_episode_id") not in controls
        }
    )
    if missing:
        preview = ", ".join(str(value) for value in missing[:5])
        raise ValueError(
            f"{len(missing)} paired control episodes are missing from the combined tournament: {preview}"
        )
    pairs = [
        _paired_record(row, controls[row["paired_control_episode_id"]])
        for row in intervention_rows
    ]
    pair_path = root / "paired_episode_effects.jsonl"
    atomic_write_jsonl(pair_path, pairs)
    return {
        "design": "one prompted focal player versus the same unprompted M0 baseline",
        "paired_episode_count": len(pairs),
        "paired_episode_effects_path": str(pair_path),
        "paired_episode_effects_sha256": sha256_file(pair_path),
        "effect_definition": (
            "paired_delta = prompted focal episode metric - matched unprompted episode metric"
        ),
        "inference_caveat": (
            "Intervals resample matched control-episode clusters. Training-seed replication, "
            "not the number of game episodes, remains the inferential limit for "
            "model-condition effects."
        ),
        "by_focal_agent_game_and_seat": _group_pairs(
            pairs, ("focal_agent_id", "game", "focal_player"), bootstrap_samples
        ),
        "by_focal_agent_and_game": _group_pairs(
            pairs, ("focal_agent_id", "game"), bootstrap_samples
        ),
        "by_condition_game_and_seat": _group_pairs(
            pairs, ("focal_condition", "game", "focal_player"), bootstrap_samples
        ),
        "by_condition_and_game": _group_pairs(
            pairs, ("focal_condition", "game"), bootstrap_samples
        ),
    }


def analyze_tournament(root: str | Path, bootstrap_samples: int = 1000) -> Path:
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    root = Path(root)
    rows = read_jsonl(root / "episodes.jsonl")
    if not rows:
        raise ValueError(f"No completed episode summaries found at {root}")
    control_rows = [row for row in rows if row.get("paired_control_episode_id") is None]
    intervention_rows = [row for row in rows if row.get("paired_control_episode_id") is not None]
    exact_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    condition_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in control_rows:
        exact_groups[(row["game"], row["player1_id"], row["player2_id"])].append(row)
        condition_groups[(row["game"], row["player1_condition"], row["player2_condition"])].append(row)

    exact = []
    for key, group in sorted(exact_groups.items()):
        game, player1, player2 = key
        exact.append(
            {
                "game": game,
                "player1_id": player1,
                "player2_id": player2,
                **_aggregate_group(group, "|".join(key), bootstrap_samples),
            }
        )
    conditions = []
    for key, group in sorted(condition_groups.items()):
        game, condition1, condition2 = key
        conditions.append(
            {
                "game": game,
                "player1_condition": condition1,
                "player2_condition": condition2,
                **_aggregate_group(group, "|".join(key), bootstrap_samples),
            }
        )
    payload = {
        "schema_version": "1.0",
        "completed_episode_count": len(rows),
        "unprompted_episode_count": len(control_rows),
        "prompt_intervention_episode_count": len(intervention_rows),
        "bootstrap_samples": bootstrap_samples,
        "inference_caveat": (
            "Intervals resample complete episodes, not rounds. Condition-level intervals are descriptive; "
            "training-seed replication remains the inferential limit for training-method claims."
        ),
        "by_agent_matchup": exact,
        "by_condition_matchup": conditions,
    }
    if intervention_rows:
        payload["prompt_intervention"] = _prompt_intervention_analysis(
            root, control_rows, intervention_rows, bootstrap_samples
        )
    destination = root / "analysis.json"
    atomic_write_json(destination, payload)
    return destination
