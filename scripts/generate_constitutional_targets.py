from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_training.constitutional import generate_constitutional_targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and cache M3 critique-revision targets")
    parser.add_argument("--prompts", default="data/safety_shared/prompts.jsonl")
    parser.add_argument("--output", default="data/safety_constitutional/train.jsonl")
    parser.add_argument("--constitution", default="configs/constitution.yaml")
    parser.add_argument("--teacher-model", required=True, help="Local/Hugging Face causal LM; no API is called")
    parser.add_argument("--teacher-revision", default="main")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--generation-seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = generate_constitutional_targets(
        args.prompts,
        args.output,
        args.constitution,
        args.teacher_model,
        args.teacher_revision,
        args.max_new_tokens,
        args.do_sample,
        args.temperature,
        args.top_p,
        args.overwrite,
        args.limit,
        args.generation_seed,
    )
    print(f"Cached targets: {path}")


if __name__ == "__main__":
    main()
