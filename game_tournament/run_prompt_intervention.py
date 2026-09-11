from __future__ import annotations

import argparse
import copy
from pathlib import Path

from safety_training.config import REPO_ROOT

from .config import (
    DEFAULT_CONFIG_PATH,
    artifact_root,
    effective_temperature,
    load_tournament_config,
    tournament_dir,
    with_overrides,
)
from .inference import ModelActionPolicy
from .prompt_intervention import (
    build_prompt_intervention_schedule,
    load_constitution_intervention,
)
from .registry import discover_agents
from .tournament import run_tournament


DEFAULT_TOURNAMENT_ID = "constitutional_prompt_vs_m0_v1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a one-player constitutional-prompt intervention against unprompted M0"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--constitution", default=str(REPO_ROOT / "configs" / "constitution.yaml")
    )
    parser.add_argument("--tournament-id", default=DEFAULT_TOURNAMENT_ID)
    parser.add_argument("--control-tournament-id", default="paper_prompts_v1")
    parser.add_argument("--conditions", nargs="+", choices=["M0", "M1", "M2", "M3", "M4"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--hf-home")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = load_tournament_config(args.config)
    config = with_overrides(
        config,
        rounds=args.rounds,
        runs=args.runs,
        conditions=args.conditions,
        seeds=args.seeds,
    )
    if "M0" not in config["agents"]["conditions"]:
        raise ValueError("Include M0 in --conditions; it is the fixed unprompted baseline")
    config = copy.deepcopy(config)
    config["tournament_id"] = args.tournament_id
    config["design"] = {
        "type": "single_player_prompt_intervention",
        "baseline": "M0",
        "focal_players": "all selected checkpoints, each in both seats",
        "paired_control_tournament_id": args.control_tournament_id,
    }
    if args.smoke_test:
        config["tournament_id"] = f"{config['tournament_id']}_smoke"
        config["design"]["paired_control_tournament_id"] = (
            f"{args.control_tournament_id}_smoke"
        )
        config = with_overrides(config, rounds=3, runs=1)

    intervention = load_constitution_intervention(Path(args.constitution))
    config["prompt_intervention"] = intervention
    models_root = artifact_root(config, args.output_root)
    base_agents = discover_agents(config, models_root)
    policy_agents, schedule = build_prompt_intervention_schedule(
        config, base_agents, intervention
    )
    rounds = int(config["experiment"]["rounds_per_episode"])
    destination = tournament_dir(config, args.output_root)

    print(f"Prompted focal checkpoints: {len(base_agents)}")
    print("Baseline opponent: M0 (always unprompted)")
    print("Focal seats: player 1 and player 2")
    print(f"Episodes: {len(schedule)}")
    print(f"Rounds: {len(schedule) * rounds}")
    print(f"Model decisions: {len(schedule) * rounds * 2}")
    print(f"Intervention: {intervention['id']} ({intervention['source_path']})")
    print(f"Destination: {destination}")
    if args.dry_run:
        return

    policy = ModelActionPolicy(
        policy_agents,
        temperature=effective_temperature(config),
        hf_home=args.hf_home or config["paths"].get("hf_home"),
        disable_thinking=bool(config["sampling"].get("disable_thinking", True)),
    )
    result = run_tournament(
        destination,
        config,
        policy_agents,
        policy,
        max_episodes=args.max_episodes,
        schedule=schedule,
    )
    print(
        f"Prompt intervention tournament {result['status']}: "
        f"{result['completed_episodes']}/{result['expected_episodes']} episodes "
        f"at {result['root']}"
    )


if __name__ == "__main__":
    main()
