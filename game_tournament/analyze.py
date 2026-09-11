from __future__ import annotations

import argparse

from .analysis import analyze_tournament
from .config import DEFAULT_CONFIG_PATH, load_tournament_config, tournament_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed repeated-game episodes")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-root")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    config = load_tournament_config(args.config)
    destination = analyze_tournament(
        tournament_dir(config, args.output_root),
        bootstrap_samples=args.bootstrap_samples,
    )
    print(f"Tournament analysis: {destination}")


if __name__ == "__main__":
    main()
