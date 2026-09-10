from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_training.config import REPO_ROOT, condition_config_path, load_config, resolve_path
from safety_training.datasets import load_training_records, validate_no_contamination, validate_shared_safety_prompts
from safety_training.evaluation import verify_frozen_suite


def main() -> None:
    configs = {condition: load_config(condition_config_path(condition), 42) for condition in ("M1", "M2", "M3")}
    train_paths = [resolve_path(config["dataset"]["train"]) for config in configs.values()]
    for condition, path in zip(configs, train_paths):
        load_training_records(path, condition)
    validate_shared_safety_prompts(
        resolve_path(configs["M2"]["dataset"]["shared_prompts"]),
        resolve_path(configs["M2"]["dataset"]["train"]),
        resolve_path(configs["M3"]["dataset"]["train"]),
    )
    eval_paths, frozen = verify_frozen_suite(configs["M1"])
    result = validate_no_contamination(train_paths, list(eval_paths.values()))
    print(f"VALID: {result}; frozen suite hash={frozen['suite_hash']}")


if __name__ == "__main__":
    main()
