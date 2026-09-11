from __future__ import annotations

import argparse

from safety_training.introspection_comparison import compare_channels


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare normalized IA, model-diff, and game signals")
    parser.add_argument("--introspection", required=True)
    parser.add_argument("--model-diff", required=True)
    parser.add_argument("--game", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sign-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    path = compare_channels(
        args.introspection,
        args.model_diff,
        args.game,
        args.output,
        sign_tolerance=args.sign_tolerance,
    )
    print(path)


if __name__ == "__main__":
    main()

