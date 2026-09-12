from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from safety_training.config import REPO_ROOT, load_yaml
from safety_training.io import read_jsonl

from .agent import run_investigator
from .config import artifact_root, load_diffing_config, session_root
from .inference import SharedAdapterTargetPair
from .registry import load_target_pair
from .session import SessionRecorder
from .statistics import score_validation_codings


DEFAULT_SEEDS_PATH = Path(__file__).with_name("seed_prompts.jsonl")


def _seed_prompt(seed_id: str | None, prompt: str | None, path: str | Path) -> tuple[str, str]:
    if prompt:
        return seed_id or "custom", prompt
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Seed prompt bank is empty: {path}")
    if seed_id is None:
        row = rows[0]
    else:
        matches = [item for item in rows if item.get("id") == seed_id]
        if not matches:
            raise ValueError(f"Unknown seed prompt ID {seed_id!r}")
        row = matches[0]
    return str(row["id"]), str(row["prompt"])


def _investigator_model(config: dict[str, Any], override: str | None) -> str:
    model = override or os.environ.get("OPENAI_MODEL") or config["investigator"].get("model")
    if not model:
        raise ValueError("Supply --investigator-model or set OPENAI_MODEL; no mutable API alias is assumed")
    return str(model)


def _ordered_pair(
    first: Any,
    second: Any,
    *,
    randomize: bool,
    master_seed: int,
    session_id: str,
) -> tuple[Any, Any]:
    if not randomize:
        return first, second
    rng = random.Random(f"{master_seed}:{session_id}")
    return (second, first) if rng.randrange(2) else (first, second)


def _preflight(args: argparse.Namespace) -> None:
    config = load_diffing_config(args.config)
    root = artifact_root(config, args.artifact_root)
    first, second = load_target_pair(root, args.model_a, args.model_b)
    print(
        json.dumps(
            {
                "status": "READY",
                "artifact_root": str(root),
                "targets": [first.as_dict(), second.as_dict()],
                "same_target_control": first.id == second.id,
            },
            indent=2,
        )
    )


def _run_agent(args: argparse.Namespace) -> None:
    config = load_diffing_config(args.config)
    root = artifact_root(config, args.artifact_root)
    requested_a, requested_b = load_target_pair(root, args.model_a, args.model_b)
    target_a, target_b = _ordered_pair(
        requested_a,
        requested_b,
        randomize=bool(config["method"]["randomize_model_order"]) and not args.fixed_order,
        master_seed=int(config["sampling"]["master_seed"]),
        session_id=args.session_id,
    )
    seed_id, seed_prompt = _seed_prompt(args.seed_id, args.seed_prompt, args.seed_prompts)
    investigator_model = _investigator_model(config, args.investigator_model)
    output_root = session_root(config, args.output_root)
    recorder = SessionRecorder.create(
        output_root,
        args.session_id,
        config=config,
        target_a=target_a,
        target_b=target_b,
        seed_prompt_id=seed_id,
        seed_prompt=seed_prompt,
        investigator_model=investigator_model,
        requested_pair=(args.model_a, args.model_b),
    )
    base_config = load_yaml(REPO_ROOT / "configs" / "base.yaml")
    backend = SharedAdapterTargetPair(
        target_a,
        target_b,
        base_config,
        hf_home=args.hf_home or config["paths"]["hf_home"],
    )
    report = run_investigator(backend, recorder, config, investigator_model)
    print(json.dumps({"session": str(recorder.path), "report": report}, indent=2))


def _score(args: argparse.Namespace) -> None:
    metrics = score_validation_codings(args.session, args.codings)
    print(json.dumps(metrics, indent=2))


def _list_seeds(args: argparse.Namespace) -> None:
    for row in read_jsonl(args.seed_prompts):
        print(f"{row['id']}\t{row['category']}\t{row['prompt']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open-ended black-box behavioral model-diffing agent")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Validate target artifacts without loading model weights")
    preflight.add_argument("--model-a", required=True)
    preflight.add_argument("--model-b", required=True)
    preflight.add_argument("--artifact-root")
    preflight.set_defaults(func=_preflight)

    agent = subparsers.add_parser("agent", help="Run the investigator and local target models end to end")
    agent.add_argument("--model-a", required=True)
    agent.add_argument("--model-b", required=True)
    agent.add_argument("--session-id", required=True)
    agent.add_argument("--artifact-root")
    agent.add_argument("--output-root")
    agent.add_argument("--hf-home")
    agent.add_argument("--investigator-model")
    agent.add_argument("--seed-id")
    agent.add_argument("--seed-prompt")
    agent.add_argument("--seed-prompts", default=str(DEFAULT_SEEDS_PATH))
    agent.add_argument("--fixed-order", action="store_true", help="Disable blinded A/B order randomization")
    agent.set_defaults(func=_run_agent)

    score = subparsers.add_parser("score", help="Score blinded held-out pairwise codings for a completed report")
    score.add_argument("--session", required=True)
    score.add_argument("--codings", required=True)
    score.set_defaults(func=_score)

    seeds = subparsers.add_parser("seeds", help="List the broad seed-prompt bank")
    seeds.add_argument("--seed-prompts", default=str(DEFAULT_SEEDS_PATH))
    seeds.set_defaults(func=_list_seeds)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

