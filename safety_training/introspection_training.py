from __future__ import annotations

import copy
import gc
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .config import REPO_ROOT, resolve_path
from .formatting import SupervisedCollator, tokenize_supervised_example
from .introspection_config import introspection_root, lock_to_m0, stage_dir
from .introspection_data import (
    load_dpo_examples,
    load_organisms,
    load_sft_examples,
    sha256_tree,
    validate_organisms_are_not_condition_targets,
)
from .introspection_modeling import (
    BehaviorAdapterCache,
    activate_adapters,
    load_introspection_model,
    save_introspection_adapter,
    sequence_log_probs,
)
from .io import atomic_write_json, atomic_write_jsonl, sha256_file, sha256_json
from .modeling import configure_hf_cache, load_tokenizer
from .run_state import latest_checkpoint
from .runtime import git_commit, hardware_info, package_versions
from .seeding import set_all_seeds


def build_adapter_schedule(
    records: list[dict[str, Any]],
    *,
    epochs: int,
    batch_size: int,
    organisms_per_step: int,
    seed: int,
) -> list[list[list[dict[str, Any]]]]:
    """Build a deterministic schedule of homogeneous organism mini-batches."""
    if epochs < 1 or batch_size < 1 or organisms_per_step < 1:
        raise ValueError("epochs, batch_size, and organisms_per_step must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["organism_id"]].append(record)
    if not grouped:
        raise ValueError("Cannot build a schedule from an empty dataset")
    rng = random.Random(seed)
    schedule: list[list[list[dict[str, Any]]]] = []
    for _ in range(epochs):
        batches: dict[str, list[list[dict[str, Any]]]] = {}
        for organism_id, items in sorted(grouped.items()):
            shuffled = list(items)
            rng.shuffle(shuffled)
            batches[organism_id] = [
                shuffled[index : index + batch_size]
                for index in range(0, len(shuffled), batch_size)
            ]
        while batches:
            available = sorted(batches)
            sampled = rng.sample(available, min(organisms_per_step, len(available)))
            step_batches: list[list[dict[str, Any]]] = []
            for organism_id in sampled:
                step_batches.append(batches[organism_id].pop(0))
                if not batches[organism_id]:
                    del batches[organism_id]
            schedule.append(step_batches)
    return schedule


def schedule_hash(schedule: list[list[list[dict[str, Any]]]]) -> str:
    ids = [[[record["id"] for record in batch] for batch in step] for step in schedule]
    return sha256_json(ids)


def _prepare_stage(
    config: dict[str, Any],
    stage: str,
    output_root: str | Path | None,
    resume: bool,
    overwrite: bool,
) -> tuple[Path, Path | None, bool]:
    destination = stage_dir(config, stage, output_root)
    root = introspection_root(config, output_root).resolve()
    resolved = destination.resolve()
    resolved.relative_to(root)
    complete = (destination / "manifest.json").exists() and (destination / "adapter").is_dir()
    if complete and not overwrite:
        return destination, None, True
    if overwrite and destination.exists():
        if resolved == root:
            raise ValueError("Refusing to remove the introspection experiment root")
        shutil.rmtree(destination)
    checkpoint = latest_checkpoint(destination)
    if checkpoint is not None and not resume:
        raise RuntimeError(f"Partial {stage} run found; use --resume or --overwrite")
    if resume and checkpoint is None:
        raise RuntimeError(f"No {stage} checkpoint is available to resume")
    destination.mkdir(parents=True, exist_ok=True)
    return destination, checkpoint, False


def _optimizer_state_load(optimizer: Any, path: Path) -> None:
    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    optimizer.load_state_dict(state)


def _rotate_checkpoints(checkpoint_root: Path, limit: int) -> None:
    if limit < 1:
        return
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_root.glob("checkpoint-*"):
        try:
            candidates.append((int(path.name.rsplit("-", 1)[1]), path))
        except (IndexError, ValueError):
            continue
    for _, path in sorted(candidates)[:-limit]:
        resolved = path.resolve()
        resolved.relative_to(checkpoint_root.resolve())
        shutil.rmtree(resolved)


def _save_checkpoint(
    model: Any,
    optimizer: Any,
    destination: Path,
    *,
    global_step: int,
    schedule_digest: str,
    save_total_limit: int,
) -> Path:
    import torch

    checkpoint = destination / "checkpoints" / f"checkpoint-{global_step}"
    save_introspection_adapter(model, checkpoint)
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    atomic_write_json(
        checkpoint / "trainer_state.json",
        {"global_step": global_step, "schedule_hash": schedule_digest},
    )
    _rotate_checkpoints(destination / "checkpoints", save_total_limit)
    return checkpoint


def _resume_step(checkpoint: Path | None, expected_schedule_hash: str) -> int:
    if checkpoint is None:
        return 0
    state_path = checkpoint / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Checkpoint state missing: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("schedule_hash") != expected_schedule_hash:
        raise ValueError("Training schedule changed since the checkpoint was written")
    return int(state["global_step"])


def _manifest_base(
    config: dict[str, Any],
    *,
    stage: str,
    m0_manifest_path: Path,
    organism_manifest_path: Path,
    dataset_path: Path,
    organisms: list[dict[str, Any]],
    training_values: dict[str, Any],
    schedule_digest: str,
    adapter_path: Path,
) -> dict[str, Any]:
    versions = package_versions()
    hardware = hardware_info()
    return {
        "manifest_version": "introspection-1.0",
        "artifact_type": "introspection_adapter",
        "stage": stage,
        "experiment_name": config["introspection"]["experiment_name"],
        "seed": int(config["seed"]),
        "base_model": config["model_name"],
        "base_revision": config["model_revision"],
        "tokenizer_revision": config.get("tokenizer_revision"),
        "m0_manifest_path": str(m0_manifest_path),
        "m0_manifest_sha256": sha256_file(m0_manifest_path),
        "organism_manifest_path": str(organism_manifest_path),
        "organism_manifest_sha256": sha256_file(organism_manifest_path),
        "training_dataset_path": str(dataset_path),
        "training_dataset_sha256": sha256_file(dataset_path),
        "organism_ids": sorted(record["organism_id"] for record in organisms),
        "behavior_families": sorted({record["behavior_family"] for record in organisms}),
        "organism_adapter_hashes": {
            record["organism_id"]: record.get("_adapter_sha256")
            for record in organisms
        },
        "training": training_values,
        "ia_peft": config["introspection"]["peft"],
        "quantization": config["quantization"],
        "schedule_hash": schedule_digest,
        "adapter_path": str(adapter_path),
        "adapter_sha256": sha256_tree(adapter_path),
        "hardware": hardware,
        "packages": versions,
        "git_commit": git_commit(REPO_ROOT),
        "methodological_reference": "Shenoy et al. (2026), arXiv:2604.16812",
        "methodological_caveat": (
            "Resource-adapted implementation for Qwen3-1.7B; results are not an exact reproduction."
        ),
    }


def _training_values(config: dict[str, Any], stage: str) -> dict[str, Any]:
    return copy.deepcopy(config["introspection"][stage])


def _optimizer(model: Any, learning_rate: float, weight_decay: float) -> Any:
    import torch

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable IA parameters are available to the optimizer")
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def _move(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def _model_device(model: Any) -> Any:
    return next(model.parameters()).device


def _limit_schedule(schedule: list[Any], max_steps: int) -> list[Any]:
    return schedule if max_steps < 0 else schedule[:max_steps]


def run_introspection_sft(
    config: dict[str, Any],
    *,
    output_root: str | Path | None = None,
    hf_home: str | Path | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> Path:
    config, m0_manifest_path = lock_to_m0(config, output_root)
    destination, checkpoint, skipped = _prepare_stage(config, "sft", output_root, resume, overwrite)
    if skipped:
        return destination
    configure_hf_cache(config, hf_home)
    set_all_seeds(int(config["seed"]))
    section = config["introspection"]
    training = _training_values(config, "sft")
    organism_path = resolve_path(section["organism_manifest"])
    dataset_path = resolve_path(section["sft_dataset"])
    organisms = load_organisms(
        organism_path,
        split="sft",
        expected_model=config["model_name"],
        expected_revision=config["model_revision"],
    )
    validate_organisms_are_not_condition_targets(
        organisms, resolve_path(output_root or config["paths"]["output_root"])
    )
    records = load_sft_examples(dataset_path, organisms)
    schedule = build_adapter_schedule(
        records,
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        organisms_per_step=int(training["organisms_per_step"]),
        seed=int(config["seed"]),
    )
    schedule = _limit_schedule(schedule, int(training.get("max_steps", -1)))
    digest = schedule_hash(schedule)
    start_step = _resume_step(checkpoint, digest)
    tokenizer = load_tokenizer(config)
    model = load_introspection_model(config, adapter_path=checkpoint, trainable=True)
    if bool(training.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
    optimizer = _optimizer(
        model,
        float(training["learning_rate"]),
        float(training.get("weight_decay", 0.0)),
    )
    if checkpoint is not None:
        _optimizer_state_load(optimizer, checkpoint / "optimizer.pt")
    collator = SupervisedCollator(tokenizer)
    by_id = {record["organism_id"]: record for record in organisms}
    cache = BehaviorAdapterCache(model, int(training.get("max_loaded_behavior_adapters", 8)))
    losses: list[float] = []
    model.train()
    import torch

    for zero_based_step, step_batches in enumerate(schedule[start_step:], start=start_step):
        optimizer.zero_grad(set_to_none=True)
        step_losses: list[float] = []
        for records_for_organism in step_batches:
            organism_id = records_for_organism[0]["organism_id"]
            behavior_name = cache.ensure(by_id[organism_id]["_adapter_path"])
            activate_adapters(
                model,
                behavior_name,
                include_introspection=True,
                train_introspection=True,
            )
            encoded = [
                tokenize_supervised_example(
                    tokenizer,
                    record["prompt"],
                    record["response"],
                    int(training["max_seq_length"]),
                )
                for record in records_for_organism
            ]
            batch = _move(collator(encoded), _model_device(model))
            loss = model(**batch).loss
            (loss / len(step_batches)).backward()
            step_losses.append(float(loss.detach().cpu()))
        max_grad_norm = float(training.get("max_grad_norm", 1.0))
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_grad_norm,
            )
        optimizer.step()
        global_step = zero_based_step + 1
        losses.append(mean(step_losses))
        save_steps = int(training.get("save_steps", 25))
        if save_steps > 0 and global_step % save_steps == 0:
            _save_checkpoint(
                model,
                optimizer,
                destination,
                global_step=global_step,
                schedule_digest=digest,
                save_total_limit=int(training.get("save_total_limit", 2)),
            )
    activate_adapters(model, None, include_introspection=True, train_introspection=False)
    adapter_path = save_introspection_adapter(model, destination / "adapter")
    tokenizer.save_pretrained(adapter_path)
    completed_steps = len(schedule)
    metrics = {
        "completed_steps": completed_steps,
        "resumed_from_step": start_step,
        "new_step_count": max(0, completed_steps - start_step),
        "mean_new_loss": mean(losses) if losses else None,
        "number_of_examples": len(records),
        "number_of_organisms": len(organisms),
    }
    atomic_write_json(destination / "training_metrics.json", metrics)
    manifest = _manifest_base(
        config,
        stage="sft",
        m0_manifest_path=m0_manifest_path,
        organism_manifest_path=organism_path,
        dataset_path=dataset_path,
        organisms=organisms,
        training_values=training,
        schedule_digest=digest,
        adapter_path=adapter_path,
    )
    manifest.update(metrics)
    atomic_write_json(destination / "manifest.json", manifest)
    return destination


def _encode_pair(tokenizer: Any, record: dict[str, Any], max_length: int) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        tokenize_supervised_example(tokenizer, record["prompt"], record["chosen"], max_length),
        tokenize_supervised_example(tokenizer, record["prompt"], record["rejected"], max_length),
    )


def _compute_reference_logps(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    organisms: list[dict[str, Any]],
    *,
    max_length: int,
    max_loaded: int,
) -> list[dict[str, Any]]:
    import torch

    collator = SupervisedCollator(tokenizer)
    by_id = {record["organism_id"]: record for record in organisms}
    cache = BehaviorAdapterCache(model, max_loaded)
    device = _model_device(model)
    model.eval()
    result: list[dict[str, Any]] = []
    with torch.no_grad():
        for record in records:
            behavior_name = cache.ensure(by_id[record["organism_id"]]["_adapter_path"])
            activate_adapters(
                model,
                behavior_name,
                include_introspection=True,
                train_introspection=False,
            )
            chosen, rejected = _encode_pair(tokenizer, record, max_length)
            chosen_batch = _move(collator([chosen]), device)
            rejected_batch = _move(collator([rejected]), device)
            chosen_logp = sequence_log_probs(model, **chosen_batch)
            rejected_logp = sequence_log_probs(model, **rejected_batch)
            result.append(
                {
                    **record,
                    "ref_chosen_logp": float(chosen_logp[0].cpu()),
                    "ref_rejected_logp": float(rejected_logp[0].cpu()),
                }
            )
    return result


def run_introspection_dpo(
    config: dict[str, Any],
    *,
    output_root: str | Path | None = None,
    hf_home: str | Path | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> Path:
    config, m0_manifest_path = lock_to_m0(config, output_root)
    destination, checkpoint, skipped = _prepare_stage(config, "dpo", output_root, resume, overwrite)
    if skipped:
        return destination
    sft_adapter = stage_dir(config, "sft", output_root) / "adapter"
    sft_manifest_path = stage_dir(config, "sft", output_root) / "manifest.json"
    if not sft_adapter.is_dir() or not sft_manifest_path.exists():
        raise FileNotFoundError("A completed SFT introspection adapter is required before DPO")
    configure_hf_cache(config, hf_home)
    set_all_seeds(int(config["seed"]))
    section = config["introspection"]
    training = _training_values(config, "dpo")
    organism_path = resolve_path(section["organism_manifest"])
    dataset_path = resolve_path(section["dpo_dataset"])
    organisms = load_organisms(
        organism_path,
        split="dpo",
        expected_model=config["model_name"],
        expected_revision=config["model_revision"],
    )
    validate_organisms_are_not_condition_targets(
        organisms, resolve_path(output_root or config["paths"]["output_root"])
    )
    records = load_dpo_examples(dataset_path, organisms)
    reference_path = destination / "reference_logps.jsonl"
    reference_manifest_path = destination / "reference_logps_manifest.json"
    sft_hash = sha256_tree(sft_adapter)
    if checkpoint is None:
        tokenizer = load_tokenizer(config)
        model = load_introspection_model(config, adapter_path=sft_adapter, trainable=False)
        reference_records = _compute_reference_logps(
            model,
            tokenizer,
            records,
            organisms,
            max_length=int(training["max_seq_length"]),
            max_loaded=int(training.get("max_loaded_behavior_adapters", 8)),
        )
        atomic_write_jsonl(reference_path, reference_records)
        atomic_write_json(
            reference_manifest_path,
            {
                "source_dataset_sha256": sha256_file(dataset_path),
                "sft_adapter_sha256": sft_hash,
                "records": len(reference_records),
            },
        )
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    else:
        if not reference_path.exists() or not reference_manifest_path.exists():
            raise FileNotFoundError("DPO reference log probabilities are missing")
        reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
        if reference_manifest.get("source_dataset_sha256") != sha256_file(dataset_path):
            raise ValueError("DPO dataset changed since reference log probabilities were computed")
        if reference_manifest.get("sft_adapter_sha256") != sft_hash:
            raise ValueError("SFT IA changed since DPO reference log probabilities were computed")
        from .io import read_jsonl

        reference_records = read_jsonl(reference_path)
        tokenizer = load_tokenizer(config)
    schedule = build_adapter_schedule(
        reference_records,
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        organisms_per_step=int(training["organisms_per_step"]),
        seed=int(config["seed"]),
    )
    schedule = _limit_schedule(schedule, int(training.get("max_steps", -1)))
    digest = schedule_hash(schedule)
    start_step = _resume_step(checkpoint, digest)
    model = load_introspection_model(
        config,
        adapter_path=checkpoint if checkpoint is not None else sft_adapter,
        trainable=True,
    )
    optimizer = _optimizer(
        model,
        float(training["learning_rate"]),
        float(training.get("weight_decay", 0.0)),
    )
    if checkpoint is not None:
        _optimizer_state_load(optimizer, checkpoint / "optimizer.pt")
    collator = SupervisedCollator(tokenizer)
    by_id = {record["organism_id"]: record for record in organisms}
    cache = BehaviorAdapterCache(model, int(training.get("max_loaded_behavior_adapters", 8)))
    beta = float(training["beta"])
    max_length = int(training["max_seq_length"])
    losses: list[float] = []
    model.train()
    import torch
    import torch.nn.functional as functional

    for zero_based_step, step_batches in enumerate(schedule[start_step:], start=start_step):
        optimizer.zero_grad(set_to_none=True)
        step_losses: list[float] = []
        for records_for_organism in step_batches:
            organism_id = records_for_organism[0]["organism_id"]
            behavior_name = cache.ensure(by_id[organism_id]["_adapter_path"])
            activate_adapters(
                model,
                behavior_name,
                include_introspection=True,
                train_introspection=True,
            )
            pairs = [_encode_pair(tokenizer, record, max_length) for record in records_for_organism]
            chosen_batch = _move(collator([pair[0] for pair in pairs]), _model_device(model))
            rejected_batch = _move(collator([pair[1] for pair in pairs]), _model_device(model))
            policy_chosen = sequence_log_probs(model, **chosen_batch)
            policy_rejected = sequence_log_probs(model, **rejected_batch)
            ref_chosen = torch.tensor(
                [record["ref_chosen_logp"] for record in records_for_organism],
                device=policy_chosen.device,
                dtype=policy_chosen.dtype,
            )
            ref_rejected = torch.tensor(
                [record["ref_rejected_logp"] for record in records_for_organism],
                device=policy_rejected.device,
                dtype=policy_rejected.dtype,
            )
            logits = beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))
            loss = -functional.logsigmoid(logits).mean()
            (loss / len(step_batches)).backward()
            step_losses.append(float(loss.detach().cpu()))
        max_grad_norm = float(training.get("max_grad_norm", 1.0))
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_grad_norm,
            )
        optimizer.step()
        global_step = zero_based_step + 1
        losses.append(mean(step_losses))
        save_steps = int(training.get("save_steps", 25))
        if save_steps > 0 and global_step % save_steps == 0:
            _save_checkpoint(
                model,
                optimizer,
                destination,
                global_step=global_step,
                schedule_digest=digest,
                save_total_limit=int(training.get("save_total_limit", 2)),
            )
    activate_adapters(model, None, include_introspection=True, train_introspection=False)
    adapter_path = save_introspection_adapter(model, destination / "adapter")
    tokenizer.save_pretrained(adapter_path)
    completed_steps = len(schedule)
    metrics = {
        "completed_steps": completed_steps,
        "resumed_from_step": start_step,
        "new_step_count": max(0, completed_steps - start_step),
        "mean_new_loss": mean(losses) if losses else None,
        "number_of_examples": len(reference_records),
        "number_of_organisms": len(organisms),
        "beta": beta,
    }
    atomic_write_json(destination / "training_metrics.json", metrics)
    manifest = _manifest_base(
        config,
        stage="dpo",
        m0_manifest_path=m0_manifest_path,
        organism_manifest_path=organism_path,
        dataset_path=dataset_path,
        organisms=organisms,
        training_values=training,
        schedule_digest=digest,
        adapter_path=adapter_path,
    )
    manifest.update(
        metrics,
        sft_adapter_path=str(sft_adapter),
        sft_adapter_sha256=sft_hash,
        sft_manifest_sha256=sha256_file(sft_manifest_path),
        reference_logps_path=str(reference_path),
        reference_logps_sha256=sha256_file(reference_path),
    )
    atomic_write_json(destination / "manifest.json", manifest)
    return destination
