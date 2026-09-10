from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_training.public_data import acquire_public_candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download revision-pinned public sources into review candidates; does not create approved study data"
    )
    parser.add_argument("--output-dir", default="data/review")
    parser.add_argument("--train-examples", type=int, default=300)
    parser.add_argument("--eval-examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.train_examples < 3 or args.eval_examples < 1:
        parser.error("train-examples must be at least 3 and eval-examples must be positive")
    path = acquire_public_candidates(
        args.output_dir, args.train_examples, args.eval_examples, args.seed, args.overwrite
    )
    print(f"Public source manifest: {path}")
    print("NEXT: review every candidate; set review_status=approved, final_category, and use=train or dual_use_eval.")
    print("No experimental training/evaluation files were created automatically.")


if __name__ == "__main__":
    main()
