from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from safety_training.io import atomic_write_json, read_jsonl


def two_sided_binomial_pvalue(successes: int, trials: int) -> float:
    """Exact two-sided p-value for H0: p=0.5, summing outcomes no likelier than observed."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("Binomial counts are invalid")
    if trials == 0:
        return 1.0
    observed = math.comb(trials, successes) / (2**trials)
    probability = sum(
        math.comb(trials, count) / (2**trials)
        for count in range(trials + 1)
        if math.comb(trials, count) / (2**trials) <= observed + 1e-15
    )
    return min(1.0, probability)


def score_validation_codings(session_dir: str | Path, codings_path: str | Path) -> dict[str, Any]:
    session = Path(session_dir).resolve()
    report = json.loads((session / "report.json").read_text(encoding="utf-8"))
    if report.get("result") != "difference_found":
        raise ValueError("Only a difference_found report can be statistically validated")
    expected = report.get("expected_model")
    if expected not in {"A", "B"}:
        raise ValueError("Report must name expected_model A or B")

    observations = {row["pair_id"]: row for row in read_jsonl(session / "observations.jsonl")}
    codings = read_jsonl(codings_path)
    if not codings:
        raise ValueError("No comparative codings were supplied")
    seen: set[str] = set()
    validation_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for coding in codings:
        pair_id = coding.get("pair_id")
        if pair_id in seen:
            raise ValueError(f"Duplicate coding for pair_id {pair_id!r}")
        seen.add(pair_id)
        if pair_id not in observations:
            raise ValueError(f"Coding references unknown pair_id {pair_id!r}")
        verdict = coding.get("verdict")
        if verdict not in {"A", "B", "neither"}:
            raise ValueError(f"Coding {pair_id!r} verdict must be A, B, or neither")
        observation = observations[pair_id]
        if observation["phase"] == "validate":
            validation_rows.append((observation, coding))
    if not validation_rows:
        raise ValueError("Codings contain no held-out validation-phase observations")

    by_prompt: dict[str, list[str]] = {}
    for observation, coding in validation_rows:
        by_prompt.setdefault(observation["prompt_id"], []).append(coding["verdict"])

    prompt_verdicts: dict[str, str] = {}
    for prompt_id, verdicts in by_prompt.items():
        votes_a = verdicts.count("A")
        votes_b = verdicts.count("B")
        if votes_a > votes_b:
            prompt_verdicts[prompt_id] = "A"
        elif votes_b > votes_a:
            prompt_verdicts[prompt_id] = "B"
        else:
            prompt_verdicts[prompt_id] = "neither"

    directional_samples = [coding["verdict"] for _, coding in validation_rows if coding["verdict"] in {"A", "B"}]
    expected_sample_wins = sum(verdict == expected for verdict in directional_samples)
    directional_prompts = [verdict for verdict in prompt_verdicts.values() if verdict in {"A", "B"}]
    expected_prompt_wins = sum(verdict == expected for verdict in directional_prompts)
    p_value = two_sided_binomial_pvalue(expected_prompt_wins, len(directional_prompts))
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    method = manifest["config"]["method"]
    distinct_prompts = len(by_prompt)
    minimum_prompts = int(method["min_validation_prompts"])
    alpha = float(method["alpha"])
    metrics = {
        "hypothesis": report["hypothesis"],
        "expected_model": expected,
        "validation_pairs": len(validation_rows),
        "directional_sample_pairs": len(directional_samples),
        "expected_model_sample_wins": expected_sample_wins,
        "sample_directional_accuracy": (
            expected_sample_wins / len(directional_samples) if directional_samples else None
        ),
        "sample_directional_frequency": len(directional_samples) / len(validation_rows),
        "distinct_validation_prompts": distinct_prompts,
        "prompt_verdicts": prompt_verdicts,
        "directional_prompts": len(directional_prompts),
        "expected_model_prompt_wins": expected_prompt_wins,
        "prompt_directional_accuracy": (
            expected_prompt_wins / len(directional_prompts) if directional_prompts else None
        ),
        "minimum_validation_prompts": minimum_prompts,
        "two_sided_exact_prompt_level_binomial_p": p_value,
        "alpha": alpha,
        "passes_prompt_count": distinct_prompts >= minimum_prompts,
        "nominally_significant": bool(directional_prompts) and p_value < alpha,
        "validated": distinct_prompts >= minimum_prompts and bool(directional_prompts) and p_value < alpha,
        "caveat": (
            "Nominal p-values do not correct for adaptive hypothesis search or multiple comparisons. "
            "Use blinded coding and independent replications for research claims."
        ),
    }
    atomic_write_json(session / "validation_metrics.json", metrics)
    return metrics
