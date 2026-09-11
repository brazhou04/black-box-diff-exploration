from __future__ import annotations

import shutil
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from .config import REPO_ROOT, artifact_dir, resolve_path
from .datasets import dataset_path, load_training_records, response_for, token_exposure_report, validate_no_contamination
from .evaluation import verify_frozen_suite
from .formatting import SupervisedCollator, tokenize_supervised_example
from .io import atomic_write_json, read_jsonl, sha256_file
from .modeling import attach_lora, configure_hf_cache, load_model, load_tokenizer
from .provenance import common_manifest, dataset_manifest, validate_manifest
from .run_state import latest_checkpoint, run_status
from .runtime import assert_t4_feasible
from .seeding import set_all_seeds


def _validate_against_shared(train_path: Path, shared_path: Path, condition: str) -> None:
    shared = {record["id"]: record for record in read_jsonl(shared_path)}
    records = load_training_records(train_path, condition)
    if {record["id"] for record in records} != set(shared):
        raise ValueError(f"{condition} IDs must exactly match data/safety_shared/prompts.jsonl")
    for record in records:
        source = shared[record["id"]]
        if record["prompt"] != source.get("prompt") or record["category"] != source.get("category"):
            raise ValueError(f"{condition} prompt/category mismatch for {record['id']}")


def _safe_overwrite(run_dir: Path, output_root: Path) -> None:
    resolved = run_dir.resolve()
    resolved.relative_to(output_root.resolve())
    if resolved == output_root.resolve():
        raise ValueError("Refusing to remove the output root itself")
    if resolved.exists():
        shutil.rmtree(resolved)


def prepare_run(
    config: dict[str, Any],
    output_root: str | Path | None,
    smoke_test: bool,
    resume: bool,
    overwrite: bool,
) -> tuple[Path, Path | None, bool]:
    root = resolve_path(output_root or config["paths"]["output_root"])
    if smoke_test:
        root = root / "_smoke"
    run_dir = artifact_dir(config, root)
    status = run_status(run_dir)
    print(f"{config['condition']} seed {config['seed']}: {status}")
    if status == "COMPLETE" and not overwrite:
        print("Completed run detected; skipping. Use --overwrite to replace it.")
        return run_dir, None, True
    if overwrite:
        _safe_overwrite(run_dir, root)
        status = "NOT_STARTED"
    checkpoint = latest_checkpoint(run_dir)
    if status == "PARTIAL" and not resume:
        raise RuntimeError("Partial run found. Re-run with --resume or explicitly use --overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, checkpoint if resume else None, False


def run_sft(
    config: dict[str, Any],
    output_root: str | Path | None = None,
    hf_home: str | Path | None = None,
    smoke_test: bool = False,
    resume: bool = False,
    overwrite: bool = False,
) -> Path:
    config = copy.deepcopy(config)
    if config["condition"] not in {"M1", "M2", "M3"}:
        raise ValueError("run_sft only supports M1, M2, and M3")
    warnings = assert_t4_feasible(config, smoke_test=smoke_test)
    for warning in warnings:
        print(f"WARNING: {warning}")
    run_dir, checkpoint, skipped = prepare_run(config, output_root, smoke_test, resume, overwrite)
    if skipped:
        return run_dir
    if not smoke_test:
        baseline_path = run_dir.parent.parent / "M0" / "manifest.json"
        if not baseline_path.exists():
            raise FileNotFoundError(
                f"M0 revision lock missing: {baseline_path}. Run python create_m0_manifest.py with the same output root first."
            )
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline.get("starting_model") != config["model_name"]:
            raise ValueError("M0 manifest model does not match this condition's configured starting model")
        resolved_revision = baseline.get("model_commit_hash")
        if not resolved_revision:
            raise ValueError("M0 manifest has no resolved model commit hash")
        config["requested_model_revision"] = config.get("model_revision")
        config["model_revision"] = resolved_revision
        config["tokenizer_revision"] = baseline.get("tokenizer_commit_hash") or resolved_revision
    configure_hf_cache(config, hf_home)
    seed_record = set_all_seeds(int(config["seed"]))
    train_path = dataset_path(config, smoke_test)
    if smoke_test:
        frozen = {
            "suite_hash": "smoke-fixtures",
            "files": {
                suite: {"sha256": sha256_file(REPO_ROOT / "tests" / "fixtures" / "eval" / f"{suite}.jsonl")}
                for suite in ("harmful", "benign_utility", "overrefusal")
            },
        }
        eval_paths = [REPO_ROOT / "tests" / "fixtures" / "eval" / f"{suite}.jsonl" for suite in frozen["files"]]
    else:
        eval_path_map, frozen = verify_frozen_suite(config)
        eval_paths = list(eval_path_map.values())
    validate_no_contamination([train_path], eval_paths)
    if config["condition"] in {"M2", "M3"}:
        shared_path = (
            REPO_ROOT / "tests" / "fixtures" / "training" / "safety_shared.jsonl"
            if smoke_test
            else resolve_path(config["dataset"]["shared_prompts"])
        )
        _validate_against_shared(train_path, shared_path, config["condition"])
    records = load_training_records(train_path, config["condition"])
    if config["condition"] == "M3" and not smoke_test:
        generation_path = resolve_path(config["dataset"]["target_generation_metadata"])
        if not generation_path.exists():
            raise FileNotFoundError("M3 target-generation manifest missing; cache targets before training")
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        constitution_path = resolve_path(config["dataset"]["constitution"])
        expected_hashes = {
            "constitutional_dataset_hash": sha256_file(train_path),
            "source_prompts_hash": sha256_file(shared_path),
            "constitution_hash": sha256_file(constitution_path),
        }
        mismatched = [key for key, value in expected_hashes.items() if generation.get(key) != value]
        if mismatched:
            raise ValueError(f"Stale or modified M3 target cache; generation manifest hash mismatch: {mismatched}")
    tokenizer = load_tokenizer(config)
    model = attach_lora(load_model(config, trainable=True), config)
    if config["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    examples = [
        tokenize_supervised_example(
            tokenizer,
            record["prompt"],
            response_for(record, config["condition"]),
            int(config["training"]["max_seq_length"]),
        )
        for record in records
    ]
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=float(config["training"]["epochs"]),
        learning_rate=float(config["training"]["learning_rate"]),
        per_device_train_batch_size=int(config["training"]["per_device_batch_size"]),
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        warmup_ratio=float(config["training"]["warmup_ratio"]),
        weight_decay=float(config["training"]["weight_decay"]),
        max_grad_norm=float(config["training"].get("max_grad_norm", 1.0)),
        logging_steps=1 if smoke_test else int(config["training"].get("logging_steps", 5)),
        save_strategy="steps",
        save_steps=1 if smoke_test else int(config["checkpointing"]["save_steps"]),
        save_total_limit=int(config["checkpointing"]["save_total_limit"]),
        max_steps=1 if smoke_test else int(config["training"].get("max_steps", -1)),
        fp16=True,
        bf16=False,
        gradient_checkpointing=bool(config["training"].get("gradient_checkpointing", True)),
        report_to="none",
        remove_unused_columns=False,
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(examples),
        data_collator=SupervisedCollator(tokenizer),
    )
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    atomic_write_json(run_dir / "training_metrics.json", result.metrics)
    with (run_dir / "training_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({key: value for key, value in config.items() if not key.startswith("_")}, handle, sort_keys=False)
    paths, hashes = dataset_manifest([train_path])
    effective_batch = int(config["training"]["per_device_batch_size"]) * int(
        config["training"]["gradient_accumulation_steps"]
    )
    exposure = token_exposure_report(
        train_path,
        config["condition"],
        tokenizer=lambda text: tokenizer(text, add_special_tokens=False)["input_ids"],
        epochs=float(config["training"]["epochs"]),
        effective_batch_size=effective_batch,
    )
    manifest = common_manifest(config, REPO_ROOT, tokenizer, model)
    truncated_examples = [example["prompt_tokens_truncated"] for example in examples if example["prompt_tokens_truncated"]]
    manifest.update(
        dataset_paths=paths,
        dataset_hashes=hashes,
        number_of_examples=len(records),
        training_token_count=sum(len(example["input_ids"]) for example in examples),
        prompt_truncation={
            "strategy": "left-truncate oldest prompt tokens; preserve the complete assistant target",
            "examples_truncated": len(truncated_examples),
            "total_prompt_tokens_truncated": sum(truncated_examples),
            "maximum_prompt_tokens_truncated": max(truncated_examples, default=0),
        },
        content_token_exposure=exposure,
        optimizer_steps=int(trainer.state.global_step),
        evaluation_dataset_hashes={suite: entry["sha256"] for suite, entry in frozen["files"].items()},
        evaluation_suite_hash=frozen["suite_hash"],
        seeds=seed_record,
        adapter_path=str(adapter_dir),
        smoke_test=smoke_test,
        full_bitwise_determinism_promised=False,
    )
    if config["condition"] == "M3":
        constitution_path = resolve_path(config["dataset"]["constitution"])
        generation_path = resolve_path(config["dataset"]["target_generation_metadata"])
        constitution_config = yaml.safe_load(constitution_path.read_text(encoding="utf-8"))
        manifest.update(
            constitution_hash=sha256_file(constitution_path),
            constitution_version=constitution_config.get("version"),
            constitutional_dataset_hash=sha256_file(train_path),
        )
        if generation_path.exists():
            generation = json.loads(generation_path.read_text(encoding="utf-8"))
            manifest.update(
                teacher_model=generation.get("teacher_model"),
                teacher_revision=generation.get("teacher_revision"),
                generation_settings=generation.get("generation_settings"),
                target_generation_config=generation,
            )
    validate_manifest(manifest)
    atomic_write_json(run_dir / "manifest.json", manifest)
    return run_dir
