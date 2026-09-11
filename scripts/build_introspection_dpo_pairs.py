from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_training.introspection_data import build_dpo_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IA DPO ground-truth/unrelated-behavior pairs")
    parser.add_argument("--organisms", default="data/introspection/organisms.jsonl")
    parser.add_argument("--questions", default="data/introspection/train_questions.jsonl")
    parser.add_argument("--output", default="data/introspection/dpo.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(build_dpo_pairs(args.organisms, args.questions, args.output, seed=args.seed))


if __name__ == "__main__":
    main()
