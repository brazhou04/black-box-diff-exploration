from __future__ import annotations

from pathlib import Path

from .config import REPO_ROOT, load_yaml, resolve_path
from .io import atomic_write_json
from .modeling import configure_hf_cache, load_tokenizer
from .provenance import common_manifest, validate_manifest


def create_m0_manifest(
    base_config_path: str | Path = REPO_ROOT / "configs" / "base.yaml",
    output_root: str | Path | None = None,
    hf_home: str | Path | None = None,
) -> Path:
    config = load_yaml(base_config_path)
    config.update(condition="M0", seed=None)
    configure_hf_cache(config, hf_home)
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(config["model_name"], revision=config.get("model_revision"))
    resolved_revision = getattr(model_config, "_commit_hash", None)
    if not resolved_revision:
        raise RuntimeError("Could not resolve the M0 Hugging Face model commit")
    config["requested_model_revision"] = config.get("model_revision")
    config["model_revision"] = resolved_revision
    config["tokenizer_revision"] = resolved_revision
    tokenizer = load_tokenizer(config)
    root = resolve_path(output_root or config["paths"]["output_root"])
    destination = root / "M0" / "manifest.json"
    manifest = common_manifest(config, REPO_ROOT, tokenizer=tokenizer)
    manifest.update(
        model_commit_hash=resolved_revision,
        dataset_paths=[],
        dataset_hashes={},
        evaluation_dataset_hashes={},
        number_of_examples=0,
        training_token_count=0,
        optimizer_steps=0,
        training_hyperparameters=None,
        peft_configuration=None,
        quantization_configuration=None,
        training_performed=False,
        model_interface={"base_model": config["model_name"], "adapter_path": None},
    )
    freeze_path = resolve_path(config["evaluation"]["frozen_manifest"])
    if freeze_path.exists():
        import json

        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        manifest["evaluation_dataset_hashes"] = {
            suite: entry["sha256"] for suite, entry in frozen.get("files", {}).items()
        }
        manifest["evaluation_suite_hash"] = frozen.get("suite_hash")
    validate_manifest(manifest)
    atomic_write_json(destination, manifest)
    return destination
