from __future__ import annotations

import argparse
from pathlib import Path

from safety_training.config import DEFAULT_SEEDS, REPO_ROOT, artifact_dir, condition_config_path, load_config, resolve_path
from safety_training.dpo import run_dpo
from safety_training.run_state import run_status
from safety_training.training import run_sft


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independently resumable training condition/seed jobs")
    parser.add_argument("--conditions", nargs="+", default=["M1", "M2", "M3"], choices=["M1", "M2", "M3", "M4"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-root")
    parser.add_argument("--hf-home")
    args = parser.parse_args()
    if "M4" in args.conditions and args.conditions == ["M4"]:
        print("M4 is optional and disabled in the default matrix; explicit selection accepted.")
    for condition in args.conditions:
        for seed in args.seeds:
            config = load_config(condition_config_path(condition), seed)
            root = resolve_path(args.output_root or config["paths"]["output_root"])
            if args.smoke_test:
                root = root / "_smoke"
            path = artifact_dir(config, root)
            print(f"{condition} seed {seed}: {run_status(path)} -> {path}")
            if args.dry_run:
                continue
            if condition == "M4":
                run_dpo(config, args.output_root, args.hf_home, args.smoke_test, args.resume, args.overwrite)
            else:
                run_sft(config, args.output_root, args.hf_home, args.smoke_test, args.resume, args.overwrite)


if __name__ == "__main__":
    main()

