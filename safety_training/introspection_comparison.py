from __future__ import annotations

import math
from pathlib import Path
from statistics import mean
from typing import Any

from .io import atomic_write_json, atomic_write_jsonl, read_jsonl


CHANNELS = ("introspection", "model_diff", "game")


def _load_channel(path: str | Path, channel: str) -> dict[tuple[str, str], dict[str, Any]]:
    records = read_jsonl(path)
    if not records:
        raise ValueError(f"{channel} comparison input is empty")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for index, record in enumerate(records):
        model_id = record.get("model_id")
        behavior_id = record.get("behavior_id")
        signal = record.get("signal")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(f"{channel} record {index} has no model_id")
        if not isinstance(behavior_id, str) or not behavior_id:
            raise ValueError(f"{channel} record {index} has no behavior_id")
        if not isinstance(signal, (int, float)) or isinstance(signal, bool) or not math.isfinite(float(signal)):
            raise ValueError(f"{channel} record {index} has non-numeric signal")
        if not -1.0 <= float(signal) <= 1.0:
            raise ValueError(f"{channel} record {index} signal must be normalized to [-1, 1]")
        key = (model_id, behavior_id)
        if key in result:
            raise ValueError(f"{channel} has duplicate model/behavior key {key}")
        result[key] = {**record, "signal": float(signal)}
    return result


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else None


def _sign(value: float, tolerance: float) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def compare_channels(
    introspection_path: str | Path,
    model_diff_path: str | Path,
    game_path: str | Path,
    destination: str | Path,
    *,
    sign_tolerance: float = 0.05,
) -> Path:
    if not 0.0 <= sign_tolerance <= 1.0:
        raise ValueError("sign_tolerance must be in [0, 1]")
    sources = {
        "introspection": _load_channel(introspection_path, "introspection"),
        "model_diff": _load_channel(model_diff_path, "model_diff"),
        "game": _load_channel(game_path, "game"),
    }
    keys = sorted(set().union(*(set(values) for values in sources.values())))
    joined: list[dict[str, Any]] = []
    for model_id, behavior_id in keys:
        row: dict[str, Any] = {"model_id": model_id, "behavior_id": behavior_id}
        for channel in CHANNELS:
            record = sources[channel].get((model_id, behavior_id))
            row[f"{channel}_signal"] = record["signal"] if record else None
            row[f"{channel}_evidence"] = record.get("evidence") if record else None
        joined.append(row)
    pair_metrics: dict[str, Any] = {}
    for left, right in (
        ("introspection", "game"),
        ("model_diff", "game"),
        ("introspection", "model_diff"),
    ):
        overlap = [
            row
            for row in joined
            if row[f"{left}_signal"] is not None and row[f"{right}_signal"] is not None
        ]
        left_values = [row[f"{left}_signal"] for row in overlap]
        right_values = [row[f"{right}_signal"] for row in overlap]
        pair_metrics[f"{left}__{right}"] = {
            "overlap": len(overlap),
            "pearson": _pearson(left_values, right_values),
            "sign_agreement": (
                mean(
                    _sign(left_value, sign_tolerance) == _sign(right_value, sign_tolerance)
                    for left_value, right_value in zip(left_values, right_values)
                )
                if overlap
                else None
            ),
            "mean_absolute_signal_gap": (
                mean(abs(left_value - right_value) for left_value, right_value in zip(left_values, right_values))
                if overlap
                else None
            ),
        }
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(destination_path / "joined_channels.jsonl", joined)
    atomic_write_json(
        destination_path / "comparison_metrics.json",
        {
            "schema_version": "audit-channel-comparison-1.0",
            "sign_tolerance": sign_tolerance,
            "input_counts": {channel: len(values) for channel, values in sources.items()},
            "union_count": len(joined),
            "triple_overlap": sum(
                all(row[f"{channel}_signal"] is not None for channel in CHANNELS) for row in joined
            ),
            "pair_metrics": pair_metrics,
            "statistical_caveat": (
                "Prompt and game-round observations nested within one model seed are not independent model replicates."
            ),
        },
    )
    return destination_path / "comparison_metrics.json"
