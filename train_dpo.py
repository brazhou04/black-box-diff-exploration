from __future__ import annotations

import argparse

from safety_training.config import load_config
from safety_training.dpo import run_dpo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optional M3-to-M4 DPO")
    parser.add_argument("--config", default="configs/m4_dpo.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--hf-home")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    path = run_dpo(
        load_config(args.config, args.seed), args.output_root, args.hf_home, args.smoke_test, args.resume, args.overwrite
    )
    print(f"Artifacts: {path}")


if __name__ == "__main__":
    main()

