from __future__ import annotations

import inspect
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from .config import REPO_ROOT, artifact_dir, resolve_path
from .datasets import dataset_path, load_training_records, validate_no_contamination
from .evaluation import verify_frozen_suite
from .formatting import serialize_assistant_completion, serialize_prompt
from .io import atomic_write_json, sha256_file
from .modeling import configure_hf_cache, load_model, load_tokenizer
from .provenance import common_manifest, dataset_manifest, validate_manifest
from .run_state import run_status
from .runtime import assert_t4_feasible
from .seeding import set_all_seeds
from .training import prepare_run


def _supported(cls: Any, values: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(cls).parameters
    return {key: value for key, value in values.items() if key in parameters}


def parent_adapter_dir(config: dict[str, Any], root: Path, smoke_test: bool = False) -> Path:
    parent_root = root / "_smoke" if smoke_test else root
    return parent_root / config["parent"]["condition"] / f"seed_{int(config['seed'])}" / "adapter"


def run_dpo(
    config: dict[str, Any],
    output_root: str | Path | None = None,
    hf_home: str | Path | None = None,
    smoke_test: bool = False,
    resume: bool = False,
    overwrite: bool = False,
) -> Path:
    config = copy.deepcopy(config)
    if config["condition"] != "M4":
        raise ValueError("DPO requires condition M4")
    assert_t4_feasible(config, smoke_test=smoke_test)
    root = resolve_path(output_root or config["paths"]["output_root"])
    parent = parent_adapter_dir(config, root, smoke_test)
    if not parent.is_dir():
        raise FileNotFoundError(f"M4 seed {config['seed']} requires its corresponding M3 adapter: {parent}")
    parent_manifest_path = parent.parent / "manifest.json"
    if not parent_manifest_path.exists():
        raise FileNotFoundError(f"M3 parent manifest missing: {parent_manifest_path}")
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest.get("condition") != "M3" or parent_manifest.get("training_seed") != int(config["seed"]):
        raise ValueError("M4 parent manifest is not the corresponding M3 seed")
    if parent_manifest.get("starting_model") != config["model_name"]:
        raise ValueError("M4 and its M3 parent do not share the configured base model")
    if not smoke_test:
        resolved_revision = parent_manifest.get("starting_revision")
        if not resolved_revision:
            raise ValueError("M3 parent manifest has no resolved starting revision")
        config["requested_model_revision"] = config.get("model_revision")
        config["model_revision"] = resolved_revision
        config["tokenizer_revision"] = parent_manifest.get("tokenizer_revision") or resolved_revision
    run_dir, checkpoint, skipped = prepare_run(config, output_root, smoke_test, resume, overwrite)
    if skipped:
        return run_dir
    configure_hf_cache(config, hf_home)
    seed_record = set_all_seeds(int(config["seed"]))
    train_path = dataset_path(config, smoke_test)
    if smoke_test:
        frozen = {
            "suite_hash": "smoke-fixtures",
            "files": {
                suite: {"sha256": sha256_file(REPO_ROOT / "tests" / "fixtures" / "eval" / f"{suite}.jsonl")}
                for suite in ("harmful", "benign_utility", "overrefusal", "dual_use")
            },
        }
        eval_paths = [REPO_ROOT / "tests" / "fixtures" / "eval" / f"{suite}.jsonl" for suite in frozen["files"]]
    else:
        eval_path_map, frozen = verify_frozen_suite(config)
        eval_paths = list(eval_path_map.values())
    validate_no_contamination([train_path], eval_paths)
    records = load_training_records(train_path, "M4")
    tokenizer = load_tokenizer(config)
    model = load_model(config, trainable=True)
    from peft import PeftModel, prepare_model_for_kbit_training

    if config["quantization"].get("enabled") and config["quantization"].get("load_in_4bit"):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(model, parent, is_trainable=True)
    rows = [
        {
            "id": row["id"],
            "prompt": serialize_prompt(tokenizer, row["prompt"]),
            "chosen": serialize_assistant_completion(tokenizer, row["prompt"], row["chosen"]),
            "rejected": serialize_assistant_completion(tokenizer, row["prompt"], row["rejected"]),
        }
        for row in records
    ]
    training_token_count = sum(
        len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"])
        + len(tokenizer(row["chosen"], add_special_tokens=False)["input_ids"])
        + len(tokenizer(row["rejected"], add_special_tokens=False)["input_ids"])
        for row in rows
    )
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    training = config["training"]
    dpo_values = {
        "output_dir": str(run_dir / "checkpoints"),
        "num_train_epochs": float(training["epochs"]),
        "learning_rate": float(training["learning_rate"]),
        "per_device_train_batch_size": int(training["per_device_batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "warmup_ratio": float(training["warmup_ratio"]),
        "weight_decay": float(training["weight_decay"]),
        "max_grad_norm": float(training.get("max_grad_norm", 1.0)),
        "logging_steps": 1 if smoke_test else int(training.get("logging_steps", 5)),
        "save_strategy": "steps",
        "save_steps": 1 if smoke_test else int(config["checkpointing"]["save_steps"]),
        "save_total_limit": int(config["checkpointing"]["save_total_limit"]),
        "max_steps": 1 if smoke_test else int(training.get("max_steps", -1)),
        "fp16": True,
        "bf16": False,
        "gradient_checkpointing": True,
        "report_to": "none",
        "seed": int(config["seed"]),
        "data_seed": int(config["seed"]),
        "beta": float(config["dpo"]["beta"]),
        "loss_type": config["dpo"].get("loss_type", "sigmoid"),
        "max_length": int(training["max_seq_length"]),
        "max_prompt_length": int(config["dpo"]["max_prompt_length"]),
    }
    dpo_args = DPOConfig(**_supported(DPOConfig, dpo_values))
    trainer_values = {
        "model": model,
        "ref_model": None,
        "args": dpo_args,
        "train_dataset": Dataset.from_list(rows),
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
    }
    trainer = DPOTrainer(**_supported(DPOTrainer, trainer_values))
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    atomic_write_json(run_dir / "training_metrics.json", result.metrics)
    with (run_dir / "training_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({key: value for key, value in config.items() if not key.startswith("_")}, handle, sort_keys=False)
    paths, hashes = dataset_manifest([train_path])
    manifest = common_manifest(config, REPO_ROOT, tokenizer, model)
    manifest.update(
        parent_adapter=str(parent),
        parent_adapter_manifest_hash=sha256_file(parent_manifest_path),
        preference_dataset_hash=sha256_file(train_path),
        dataset_paths=paths,
        dataset_hashes=hashes,
        number_of_examples=len(records),
        training_token_count=training_token_count,
        optimizer_steps=int(trainer.state.global_step),
        evaluation_dataset_hashes={suite: entry["sha256"] for suite, entry in frozen["files"].items()},
        evaluation_suite_hash=frozen["suite_hash"],
        dpo_beta=float(config["dpo"]["beta"]),
        dpo_configuration=config["dpo"],
        seeds=seed_record,
        adapter_path=str(adapter_dir),
        smoke_test=smoke_test,
    )
    validate_manifest(manifest)
    atomic_write_json(run_dir / "manifest.json", manifest)
    return run_dir
