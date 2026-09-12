from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safety_training.io import sha256_file


MODEL_ID_PATTERN = re.compile(r"^(M[1-4])_seed_([0-9]+)$")


@dataclass(frozen=True)
class TargetSpec:
    id: str
    condition: str
    training_seed: int | None
    run_dir: Path
    manifest_path: Path
    adapter_path: Path | None
    model_name: str
    model_revision: str
    tokenizer_revision: str
    manifest_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "condition": self.condition,
            "training_seed": self.training_seed,
            "run_dir": str(self.run_dir),
            "manifest_path": str(self.manifest_path),
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "manifest_sha256": self.manifest_sha256,
        }


def parse_target_id(target_id: str) -> tuple[str, int | None]:
    if target_id == "M0":
        return "M0", None
    match = MODEL_ID_PATTERN.fullmatch(target_id)
    if not match:
        raise ValueError("Target IDs must be M0 or M1_seed_N through M4_seed_N")
    return match.group(1), int(match.group(2))


def load_target(artifact_root: str | Path, target_id: str) -> TargetSpec:
    root = Path(artifact_root).resolve()
    condition, seed = parse_target_id(target_id)
    run_dir = root / "M0" if condition == "M0" else root / condition / f"seed_{seed}"
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Completed target manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("condition") != condition:
        raise ValueError(f"Manifest condition mismatch at {manifest_path}")
    if seed is not None and int(manifest.get("training_seed", -1)) != seed:
        raise ValueError(f"Manifest training seed mismatch at {manifest_path}")

    model_name = manifest.get("starting_model")
    revision = manifest.get("model_commit_hash") if condition == "M0" else manifest.get("starting_revision")
    if not model_name or not revision:
        raise ValueError(f"Manifest lacks a resolved base-model identity: {manifest_path}")
    tokenizer_revision = manifest.get("tokenizer_revision") or revision
    adapter_path = None if condition == "M0" else run_dir / "adapter"
    if adapter_path is not None and not adapter_path.is_dir():
        raise FileNotFoundError(f"Completed adapter not found: {adapter_path}")

    return TargetSpec(
        id=target_id,
        condition=condition,
        training_seed=seed,
        run_dir=run_dir,
        manifest_path=manifest_path,
        adapter_path=adapter_path,
        model_name=str(model_name),
        model_revision=str(revision),
        tokenizer_revision=str(tokenizer_revision),
        manifest_sha256=sha256_file(manifest_path),
    )


def load_target_pair(artifact_root: str | Path, model_a: str, model_b: str) -> tuple[TargetSpec, TargetSpec]:
    first = load_target(artifact_root, model_a)
    second = load_target(artifact_root, model_b)
    first_identity = (first.model_name, first.model_revision, first.tokenizer_revision)
    second_identity = (second.model_name, second.model_revision, second.tokenizer_revision)
    if first_identity != second_identity:
        raise ValueError(
            "Diff targets do not share the exact pinned base/tokenizer identity: "
            f"{first.id}={first_identity}, {second.id}={second_identity}"
        )
    return first, second

