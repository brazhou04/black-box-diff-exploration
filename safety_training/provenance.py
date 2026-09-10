from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import sha256_file, sha256_json
from .runtime import git_commit, hardware_info, package_versions


REFERENCES = {
    "M1": ["Qi et al. (2023), arXiv:2310.03693"],
    "M2": ["Bianchi et al. (2023/2024), arXiv:2309.07875"],
    "M3": [
        "Bai et al. (2022), arXiv:2212.08073",
        "Chen et al. (2024), NAACL 2024, DOI:10.18653/v1/2024.naacl-long.78",
    ],
    "M4": ["Rafailov et al. (2023), arXiv:2305.18290"],
}

CAVEATS = {
    "M0": "Untouched conversational checkpoint; unified manifest is a study interface choice.",
    "M1": "Benign fine-tuning drift control inspired by Qi et al.; exact exposure matching is study-specific.",
    "M2": "Direct safe-response SFT inspired by safety-tuning literature, not proprietary safe-completion training.",
    "M3": "Paper-inspired approximation of supervised critique-revision, not full Constitutional AI or IterAlign.",
    "M4": "DPO applied after M3 is a staged study extension, not an ordering required by DPO.",
}

REQUIRED_MANIFEST_FIELDS = {
    "condition",
    "starting_model",
    "starting_revision",
    "training_seed",
    "dataset_paths",
    "dataset_hashes",
    "evaluation_dataset_hashes",
    "number_of_examples",
    "training_token_count",
    "optimizer_steps",
    "training_hyperparameters",
    "peft_configuration",
    "quantization_configuration",
    "hardware",
    "precision",
    "python_version",
    "pytorch_version",
    "transformers_version",
    "peft_version",
    "trl_version",
    "cuda_version",
    "git_commit",
    "methodological_references",
    "methodological_caveats",
}


def validate_manifest(manifest: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_MANIFEST_FIELDS - manifest.keys())
    if missing:
        raise ValueError(f"Manifest missing required provenance fields: {missing}")


def common_manifest(
    config: dict[str, Any],
    repo_root: Path,
    tokenizer: Any | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    versions = package_versions()
    hardware = hardware_info()
    condition = config["condition"]
    manifest: dict[str, Any] = {
        "manifest_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition": condition,
        "starting_model": config["model_name"],
        "starting_revision": config.get("model_revision"),
        "requested_starting_revision": config.get("requested_model_revision", config.get("model_revision")),
        "tokenizer_revision": config.get("tokenizer_revision"),
        "training_seed": config.get("seed"),
        "training_hyperparameters": config.get("training"),
        "peft_configuration": config.get("peft"),
        "quantization_configuration": config.get("quantization"),
        "precision": config.get("quantization", {}).get("compute_dtype", "float16"),
        "hardware": hardware,
        "packages": versions,
        "python_version": hardware["python_version"],
        "pytorch_version": versions["torch"],
        "transformers_version": versions["transformers"],
        "peft_version": versions["peft"],
        "trl_version": versions["trl"],
        "cuda_version": hardware["pytorch_cuda_version"],
        "git_commit": git_commit(repo_root),
        "methodological_references": REFERENCES.get(condition, []),
        "methodological_caveats": CAVEATS[condition],
    }
    if tokenizer is not None:
        template = getattr(tokenizer, "chat_template", None)
        manifest["chat_template"] = template
        manifest["chat_template_sha256"] = sha256_json(template)
        manifest["tokenizer_commit_hash"] = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    if model is not None:
        manifest["model_commit_hash"] = getattr(getattr(model, "config", None), "_commit_hash", None)
    return manifest


def dataset_manifest(paths: list[Path]) -> tuple[list[str], dict[str, str]]:
    return [str(path) for path in paths], {str(path): sha256_file(path) for path in paths}
