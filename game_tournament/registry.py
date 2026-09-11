from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safety_training.io import sha256_file


@dataclass(frozen=True)
class AgentSpec:
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
    prompt_intervention: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
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
        if self.prompt_intervention is not None:
            payload["prompt_intervention"] = self.prompt_intervention
        return payload


def _agent_from_manifest(condition: str, seed: int | None, run_dir: Path) -> AgentSpec:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Completed {condition} manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("condition") != condition:
        raise ValueError(f"Manifest condition mismatch at {manifest_path}")
    recorded_seed = manifest.get("training_seed")
    if seed is not None and int(recorded_seed) != seed:
        raise ValueError(f"Manifest training seed mismatch at {manifest_path}")
    model_name = manifest.get("starting_model")
    revision = manifest.get("model_commit_hash") if condition == "M0" else manifest.get("starting_revision")
    if not model_name or not revision:
        raise ValueError(f"Manifest lacks a resolved base model identity: {manifest_path}")
    tokenizer_revision = manifest.get("tokenizer_revision") or revision
    adapter = None if condition == "M0" else run_dir / "adapter"
    if adapter is not None and not adapter.is_dir():
        raise FileNotFoundError(f"Completed adapter not found: {adapter}")
    identifier = "M0" if condition == "M0" else f"{condition}_seed_{seed}"
    return AgentSpec(
        id=identifier,
        condition=condition,
        training_seed=seed,
        run_dir=run_dir,
        manifest_path=manifest_path,
        adapter_path=adapter,
        model_name=str(model_name),
        model_revision=str(revision),
        tokenizer_revision=str(tokenizer_revision),
        manifest_sha256=sha256_file(manifest_path),
    )


def discover_agents(config: dict[str, Any], artifact_root: str | Path) -> list[AgentSpec]:
    root = Path(artifact_root)
    agents: list[AgentSpec] = []
    for condition in config["agents"]["conditions"]:
        if condition == "M0":
            agents.append(_agent_from_manifest("M0", None, root / "M0"))
            continue
        for seed in config["agents"]["seeds"]:
            agents.append(_agent_from_manifest(condition, int(seed), root / condition / f"seed_{int(seed)}"))
    identities = {(agent.model_name, agent.model_revision, agent.tokenizer_revision) for agent in agents}
    if len(identities) != 1:
        details = sorted((agent.id, agent.model_name, agent.model_revision, agent.tokenizer_revision) for agent in agents)
        raise ValueError(f"Tournament agents do not share one pinned base/tokenizer identity: {details}")
    return agents
