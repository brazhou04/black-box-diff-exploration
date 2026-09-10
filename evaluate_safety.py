from __future__ import annotations

import argparse

from safety_training.evaluation import evaluate_condition


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen immediate post-training safety/utility audit")
    parser.add_argument("--condition", required=True, choices=["M0", "M1", "M2", "M3", "M4"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--hf-home")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--scorer", help="heuristic or importable module.path:SafetyScorerClass")
    args = parser.parse_args()
    path = evaluate_condition(args.condition, args.seed, args.output_root, args.hf_home, args.smoke_test, args.scorer)
    print(f"Audit metrics: {path}")


if __name__ == "__main__":
    main()
