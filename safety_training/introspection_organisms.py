from __future__ import annotations

import copy
import gc
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from .config import resolve_path
from .formatting import SupervisedCollator, tokenize_supervised_example
from .introspection_config import lock_to_m0
from .introspection_data import ORGANISM_SPLITS, sha256_tree
from .io import atomic_write_json, atomic_write_jsonl, read_jsonl, sha256_file
from .modeling import attach_lora, configure_hf_cache, load_model, load_tokenizer
from .run_state import latest_checkpoint
from .seeding import set_all_seeds


def load_organism_specs(path: str | Path) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    if not records:
        raise ValueError("Organism specification file is empty")
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        label = f"organism specification {index}"
        record = dict(raw)
        for field in ("organism_id", "behavior_family", "introspection_target", "split", "training_dataset"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} requires non-empty string field {field!r}")
            record[field] = value.strip()
        if record["organism_id"] in seen:
            raise ValueError(f"Duplicate organism_id {record['organism_id']!r}")
        seen.add(record["organism_id"])
        if record["split"] not in ORGANISM_SPLITS - {"calibration"}:
            raise ValueError(f"{label} has invalid trainable split {record['split']!r}")
        result.append(record)
    return result


def _load_behavior_records(path: Path) -> list[dict[str, str]]:
    records = read_jsonl(path)
    if not records:
        raise ValueError(f"Behavior dataset is empty: {path}")
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for index, record in enumerate(records):
        label = f"{path} record {index}"
        values: dict[str, str] = {}
        for field in ("id", "prompt", "response"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} requires non-empty string field {field!r}")
            values[field] = value.strip()
        if values["id"] in seen:
            raise ValueError(f"{path}: duplicate id {values['id']!r}")
        seen.add(values["id"])
        result.append(values)
    return result


def _safe_overwrite(destination: Path, root: Path) -> None:
    resolved = destination.resolve()
    resolved.relative_to(root.resolve())
    if resolved == root.resolve():
        raise ValueError("Refusing to remove the organism artifact root")
    if destination.exists():
        shutil.rmtree(destination)


def train_introspection_organisms(
    config: dict[str, Any],
    *,
    specs_path: str | Path,
    output_root: str | Path | None = None,
    hf_home: str | Path | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> Path:
    config, m0_manifest_path = lock_to_m0(config, output_root)
    configure_hf_cache(config, hf_home)
    specs_file = resolve_path(specs_path)
    specs = load_organism_specs(specs_file)
    root = resolve_path(output_root or config["paths"]["output_root"])
    organism_root = root / "introspection_organisms"
    organism_root.mkdir(parents=True, exist_ok=True)
    values = config["introspection"]["organism_training"]
    registry: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        organism_id = spec["organism_id"]
        destination = organism_root / organism_id
        adapter_path = destination / "adapter"
        manifest_path = destination / "manifest.json"
        complete = adapter_path.is_dir() and manifest_path.exists()
        if overwrite:
            _safe_overwrite(destination, organism_root)
            complete = False
        if not complete:
            checkpoint = latest_checkpoint(destination)
            if checkpoint is not None and not resume:
                raise RuntimeError(f"Partial organism {organism_id} found; use --resume or --overwrite")
            destination.mkdir(parents=True, exist_ok=True)
            seed = int(config["seed"]) + index
            set_all_seeds(seed)
            dataset_path = resolve_path(spec["training_dataset"])
            records = _load_behavior_records(dataset_path)
            tokenizer = load_tokenizer(config)
            model = attach_lora(load_model(config, trainable=True), config)
            if bool(values.get("gradient_checkpointing", True)):
                model.gradient_checkpointing_enable()
            examples = [
                tokenize_supervised_example(
                    tokenizer,
                    record["prompt"],
                    record["response"],
                    int(values["max_seq_length"]),
                )
                for record in records
            ]
            from datasets import Dataset
            from transformers import Trainer, TrainingArguments

            arguments = TrainingArguments(
                output_dir=str(destination / "checkpoints"),
                num_train_epochs=float(values["epochs"]),
                learning_rate=float(values["learning_rate"]),
                per_device_train_batch_size=int(values["batch_size"]),
                gradient_accumulation_steps=int(values["gradient_accumulation_steps"]),
                warmup_ratio=float(values.get("warmup_ratio", 0.03)),
                weight_decay=float(values.get("weight_decay", 0.0)),
                max_grad_norm=float(values.get("max_grad_norm", 1.0)),
                logging_steps=int(values.get("logging_steps", 5)),
                save_strategy="steps",
                save_steps=int(values.get("save_steps", 25)),
                save_total_limit=int(values.get("save_total_limit", 2)),
                max_steps=int(values.get("max_steps", -1)),
                fp16=True,
                bf16=False,
                gradient_checkpointing=bool(values.get("gradient_checkpointing", True)),
                report_to="none",
                remove_unused_columns=False,
                seed=seed,
                data_seed=seed,
            )
            trainer = Trainer(
                model=model,
                args=arguments,
                train_dataset=Dataset.from_list(examples),
                data_collator=SupervisedCollator(tokenizer),
            )
            result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
            trainer.save_model(str(adapter_path))
            tokenizer.save_pretrained(adapter_path)
            with (destination / "training_config.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(values, handle, sort_keys=False)
            atomic_write_json(destination / "training_metrics.json", result.metrics)
            atomic_write_json(
                manifest_path,
                {
                    "manifest_version": "introspection-organism-1.0",
                    "artifact_type": "introspection_behavior_organism",
                    "organism_id": organism_id,
                    "behavior_family": spec["behavior_family"],
                    "introspection_target": spec["introspection_target"],
                    "split": spec["split"],
                    "base_model": config["model_name"],
                    "base_revision": config["model_revision"],
                    "seed": seed,
                    "training_dataset": str(dataset_path),
                    "training_dataset_sha256": sha256_file(dataset_path),
                    "m0_manifest_sha256": sha256_file(m0_manifest_path),
                    "adapter_path": str(adapter_path),
                    "optimizer_steps": int(trainer.state.global_step),
                    "training": copy.deepcopy(values),
                    "peft": copy.deepcopy(config["peft"]),
                    "quantization": copy.deepcopy(config["quantization"]),
                },
            )
            del trainer, model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("base_model") != config["model_name"] or manifest.get("base_revision") != config["model_revision"]:
            raise ValueError(f"Completed organism {organism_id} does not match the locked base")
        registry.append(
            {
                "organism_id": organism_id,
                "behavior_family": spec["behavior_family"],
                "introspection_target": spec["introspection_target"],
                "split": spec["split"],
                "base_model": config["model_name"],
                "base_revision": config["model_revision"],
                "adapter_path": str(adapter_path),
                "adapter_sha256": sha256_tree(adapter_path),
                "organism_manifest_path": str(manifest_path),
                "organism_manifest_sha256": sha256_file(manifest_path),
            }
        )
    registry_path = resolve_path(config["introspection"]["organism_manifest"])
    atomic_write_jsonl(registry_path, registry)
    return registry_path

