from __future__ import annotations

import argparse

from safety_training.introspection_config import load_introspection_config
from safety_training.introspection_evaluation import evaluate_heldout_organisms


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an IA on known-answer held-out behavior organisms")
    parser.add_argument("--config", default="configs/introspection_sft.yaml")
    parser.add_argument("--ia-stage", choices=["sft", "dpo"], default="dpo")
    parser.add_argument("--questions", default="data/introspection/eval_questions.jsonl")
    parser.add_argument("--split", choices=["dpo", "eval", "calibration"], default="eval")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--hf-home")
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()
    config = load_introspection_config(args.config, args.seed)
    path = evaluate_heldout_organisms(
        config,
        ia_stage=args.ia_stage,
        questions_path=args.questions,
        output_root=args.output_root,
        hf_home=args.hf_home,
        include_baseline=not args.no_baseline,
        organism_split=args.split,
    )
    print(path)


if __name__ == "__main__":
    main()
