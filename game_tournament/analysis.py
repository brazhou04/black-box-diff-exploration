from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from safety_training.io import atomic_write_json, read_jsonl

from .tournament import derive_seed


IDENTITY_FIELDS = {
    "round_count",
    "total_payoff_player1",
    "total_payoff_player2",
    "coordinated_transition_count",
}


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


def analyze_tournament(root: str | Path, bootstrap_samples: int = 1000) -> Path:
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    root = Path(root)
    rows = read_jsonl(root / "episodes.jsonl")
    if not rows:
        raise ValueError(f"No completed episode summaries found at {root}")
    exact_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    condition_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
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
        "bootstrap_samples": bootstrap_samples,
        "inference_caveat": (
            "Intervals resample complete episodes, not rounds. Condition-level intervals are descriptive; "
            "training-seed replication remains the inferential limit for training-method claims."
        ),
        "by_agent_matchup": exact,
        "by_condition_matchup": conditions,
    }
    destination = root / "analysis.json"
    atomic_write_json(destination, payload)
    return destination
