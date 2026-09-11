from __future__ import annotations

import argparse

from safety_training.config import DEFAULT_SEEDS
from safety_training.introspection_config import load_introspection_config
from safety_training.introspection_evaluation import evaluate_introspection


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IA-on and IA-off reports for frozen M0-M4 targets")
    parser.add_argument("--config", default="configs/introspection_sft.yaml")
    parser.add_argument("--ia-stage", choices=["sft", "dpo"], default="dpo")
    parser.add_argument("--conditions", nargs="+", choices=["M0", "M1", "M2", "M3", "M4"], default=["M0", "M1", "M2", "M3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--questions", default="data/introspection/eval_questions.jsonl")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--hf-home")
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()
    config = load_introspection_config(args.config, args.seed)
    path = evaluate_introspection(
        config,
        ia_stage=args.ia_stage,
        conditions=args.conditions,
        seeds=args.seeds,
        questions_path=args.questions,
        output_root=args.output_root,
        hf_home=args.hf_home,
        include_baseline=not args.no_baseline,
    )
    print(path)


if __name__ == "__main__":
    main()

