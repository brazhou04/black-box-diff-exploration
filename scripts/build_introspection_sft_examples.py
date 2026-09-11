from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_training.introspection_data import build_sft_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IA SFT examples from frozen organism labels")
    parser.add_argument("--organisms", default="data/introspection/organisms.jsonl")
    parser.add_argument("--questions", default="data/introspection/train_questions.jsonl")
    parser.add_argument("--output", default="data/introspection/sft.jsonl")
    args = parser.parse_args()
    print(build_sft_examples(args.organisms, args.questions, args.output))


if __name__ == "__main__":
    main()
