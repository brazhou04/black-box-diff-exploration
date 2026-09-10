from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_training.config import load_yaml, resolve_path
from safety_training.evaluation import freeze_evaluation_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and freeze immediate post-training evaluation data")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_yaml(args.config)
    output = args.output or config["evaluation"]["frozen_manifest"]
    missing = [resolve_path(path) for path in config["evaluation"]["datasets"].values() if not resolve_path(path).exists()]
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "Real evaluation data are not bundled with the repository. Missing:\n"
            f"{lines}\nAcquire and finalize the public source-trust data first. This step is not needed before a synthetic smoke test."
        )
    print(freeze_evaluation_suite(config["evaluation"]["datasets"], output))


if __name__ == "__main__":
    main()
