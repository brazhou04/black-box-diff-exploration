from __future__ import annotations

import argparse

from .prompt_intervention_analysis import analyze_prompt_intervention


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare prompted focal episodes with matched unprompted tournament controls"
    )
    parser.add_argument(
        "--control-root",
        default="/kaggle/working/artifacts/game_tournaments/paper_prompts_v1",
    )
    parser.add_argument(
        "--intervention-root",
        default=(
            "/kaggle/working/artifacts/game_tournaments/"
            "constitutional_prompt_vs_m0_v1"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    destination = analyze_prompt_intervention(
        args.control_root,
        args.intervention_root,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(f"Paired prompt-intervention analysis: {destination}")


if __name__ == "__main__":
    main()
