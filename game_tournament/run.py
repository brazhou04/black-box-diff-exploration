from __future__ import annotations

import argparse
import copy

from .config import (
    DEFAULT_CONFIG_PATH,
    artifact_root,
    effective_temperature,
    load_tournament_config,
    tournament_dir,
    with_overrides,
)
from .inference import ModelActionPolicy
from .prompting import build_prompt_variants
from .registry import discover_agents
from .tournament import build_schedule, run_tournament


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the repeated Prisoner's Dilemma, Stag Hunt, and Battle of the Sexes tournament"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--conditions", nargs="+", choices=["M0", "M1", "M2", "M3", "M4"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--runs", type=int, help="Independent episodes per ordered matchup and game")
    parser.add_argument("--rounds", type=int, help="Repeated rounds within each episode")
    parser.add_argument("--output-root", help="Artifact root containing M0-M4; tournament is written below it")
    parser.add_argument("--hf-home")
    parser.add_argument("--max-episodes", type=int, help="Stop after this many new episodes; resume later")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Use one three-round episode per matchup")
    args = parser.parse_args()

    config = load_tournament_config(args.config)
    config = with_overrides(
        config,
        rounds=args.rounds,
        runs=args.runs,
        conditions=args.conditions,
        seeds=args.seeds,
    )
    if args.smoke_test:
        config = copy.deepcopy(config)
        config["tournament_id"] = f"{config['tournament_id']}_smoke"
        config = with_overrides(config, rounds=3, runs=1)

    models_root = artifact_root(config, args.output_root)
    agents = discover_agents(config, models_root)
    schedule = build_schedule(config, agents)
    rounds = int(config["experiment"]["rounds_per_episode"])
    variants = build_prompt_variants(config["prompting"])
    destination = tournament_dir(config, args.output_root)

    print(f"Agents: {len(agents)} ({', '.join(agent.id for agent in agents)})")
    print(f"Prompt variants: {len(variants)}")
    print(f"Episodes: {len(schedule)}")
    print(f"Rounds: {len(schedule) * rounds}")
    print(f"Model decisions: {len(schedule) * rounds * 2}")
    configured_temperature = config["sampling"].get("temperature")
    if configured_temperature is None:
        print(f"Temperature: model default (effective {effective_temperature(config):g})")
    else:
        print(f"Temperature: configured {effective_temperature(config):g}")
    print(f"Destination: {destination}")
    if args.dry_run:
        return

    hf_home = args.hf_home or config["paths"].get("hf_home")
    policy = ModelActionPolicy(
        agents,
        temperature=effective_temperature(config),
        hf_home=hf_home,
        disable_thinking=bool(config["sampling"].get("disable_thinking", True)),
    )
    result = run_tournament(
        destination,
        config,
        agents,
        policy,
        max_episodes=args.max_episodes,
    )
    print(
        f"Tournament {result['status']}: {result['completed_episodes']}/"
        f"{result['expected_episodes']} episodes at {result['root']}"
    )


if __name__ == "__main__":
    main()
