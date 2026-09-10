from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_training.config import condition_config_path, load_config, resolve_path
from safety_training.datasets import token_exposure_report, validate_shared_safety_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Report training exposure and enforce M2/M3 prompt matching")
    parser.add_argument("--conditions", nargs="+", default=["M1", "M2", "M3"], choices=["M1", "M2", "M3"])
    parser.add_argument("--seed", type=int, default=42, help="Used only to resolve the shared optimization config")
    parser.add_argument("--model-tokenizer", action="store_true", help="Download/load the Qwen tokenizer for exact counts")
    args = parser.parse_args()
    configs = {condition: load_config(condition_config_path(condition), args.seed) for condition in args.conditions}
    tokenizer_fn = None
    if args.model_tokenizer:
        from safety_training.modeling import load_tokenizer

        tokenizer = load_tokenizer(next(iter(configs.values())))
        tokenizer_fn = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]
    reports = []
    for condition, config in configs.items():
        training = config["training"]
        effective_batch = int(training["per_device_batch_size"]) * int(training["gradient_accumulation_steps"])
        reports.append(
            token_exposure_report(
                resolve_path(config["dataset"]["train"]),
                condition,
                tokenizer_fn,
                float(training["epochs"]),
                effective_batch,
            )
        )
    if {"M2", "M3"}.issubset(configs):
        matching = validate_shared_safety_prompts(
            resolve_path(configs["M2"]["dataset"]["shared_prompts"]),
            resolve_path(configs["M2"]["dataset"]["train"]),
            resolve_path(configs["M3"]["dataset"]["train"]),
        )
        print("M2/M3 structural match:", json.dumps(matching, indent=2))
    print(json.dumps(reports, indent=2))
    if reports:
        totals = [report["total_tokens"] for report in reports]
        print(f"Residual total-token range: {min(totals)}..{max(totals)} (reported, not forced equal)")


if __name__ == "__main__":
    main()
