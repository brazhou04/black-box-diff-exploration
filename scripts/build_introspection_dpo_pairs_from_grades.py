from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_training.introspection_data import build_dpo_pairs_from_grades


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IA DPO pairs from judge-scored SFT reports")
    parser.add_argument("--organisms", default="data/introspection/organisms.jsonl")
    parser.add_argument("--graded-predictions", required=True)
    parser.add_argument("--output", default="data/introspection/dpo.jsonl")
    parser.add_argument("--chosen-threshold", type=float, default=7.0)
    parser.add_argument("--minimum-margin", type=float, default=2.0)
    parser.add_argument("--max-pairs-per-organism", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = build_dpo_pairs_from_grades(
        args.organisms,
        args.graded_predictions,
        args.output,
        chosen_threshold=args.chosen_threshold,
        minimum_margin=args.minimum_margin,
        max_pairs_per_organism=args.max_pairs_per_organism,
        seed=args.seed,
    )
    print(path)


if __name__ == "__main__":
    main()

