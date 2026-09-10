from __future__ import annotations

import argparse

from safety_training.config import load_config
from safety_training.training import run_sft


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one M1/M2/M3 LoRA adapter")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--hf-home")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    path = run_sft(
        load_config(args.config, args.seed),
        output_root=args.output_root,
        hf_home=args.hf_home,
        resume=args.resume,
        overwrite=args.overwrite,
        smoke_test=args.smoke_test,
    )
    print(f"Artifacts: {path}")


if __name__ == "__main__":
    main()

