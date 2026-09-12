from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from safety_training.io import atomic_write_json, atomic_write_jsonl, read_jsonl

from .config import config_for_manifest, diffing_config_hash
from .registry import TargetSpec


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionRecorder:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.manifest_path = self.path / "manifest.json"
        self.observations_path = self.path / "observations.jsonl"
        self.events_path = self.path / "events.jsonl"
        self.report_path = self.path / "report.json"

    @classmethod
    def create(
        cls,
        root: str | Path,
        session_id: str,
        *,
        config: dict[str, Any],
        target_a: TargetSpec,
        target_b: TargetSpec,
        seed_prompt_id: str,
        seed_prompt: str,
        investigator_model: str,
        requested_pair: tuple[str, str],
    ) -> "SessionRecorder":
        if not session_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in session_id
        ):
            raise ValueError("session_id may contain only letters, digits, underscores, and hyphens")
        recorder = cls(Path(root) / session_id)
        if recorder.path.exists():
            raise FileExistsError(f"Refusing to overwrite existing diffing session: {recorder.path}")
        recorder.path.mkdir(parents=True)
        manifest = {
            "schema_version": "1.0",
            "session_id": session_id,
            "experiment_id": config["experiment_id"],
            "status": "running",
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "method_source": "Chughtai, Engels, and Nanda (2026), Building and evaluating model diffing agents",
            "method_source_url": "https://www.lesswrong.com/posts/qi4mNbZYAFDYwfRba/building-and-evaluating-model-diffing-agents",
            "config": config_for_manifest(config),
            "config_sha256": diffing_config_hash(config),
            "requested_pair": list(requested_pair),
            "blind_label_mapping": {"A": target_a.as_dict(), "B": target_b.as_dict()},
            "seed_prompt_id": seed_prompt_id,
            "seed_prompt": seed_prompt,
            "investigator_model": investigator_model,
            "send_messages_calls": 0,
            "api_response_ids": [],
            "report_is_agent_claim_only": True,
        }
        atomic_write_json(recorder.manifest_path, manifest)
        atomic_write_jsonl(recorder.observations_path, [])
        atomic_write_jsonl(recorder.events_path, [])
        return recorder

    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _update_manifest(self, **updates: Any) -> None:
        manifest = self.manifest()
        manifest.update(updates)
        manifest["updated_at_utc"] = utc_now()
        atomic_write_json(self.manifest_path, manifest)

    @staticmethod
    def _append_jsonl(path: Path, new_records: Iterable[dict[str, Any]]) -> None:
        existing = read_jsonl(path) if path.exists() else []
        atomic_write_jsonl(path, [*existing, *new_records])

    def add_api_response_id(self, response_id: str) -> None:
        manifest = self.manifest()
        ids = list(manifest.get("api_response_ids", []))
        ids.append(response_id)
        self._update_manifest(api_response_ids=ids)

    def record_visible_agent_text(self, response_id: str, text: str) -> None:
        self._append_jsonl(
            self.events_path,
            [{"type": "investigator_visible_text", "at_utc": utc_now(), "response_id": response_id, "text": text}],
        )

    def record_tool_error(self, tool_name: str, message: str) -> None:
        self._append_jsonl(
            self.events_path,
            [{"type": "tool_validation_error", "at_utc": utc_now(), "tool": tool_name, "message": message}],
        )

    def record_query(
        self,
        *,
        turn: int,
        phase: str,
        test_purpose: str,
        raw_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for prompt_index, result in enumerate(raw_results):
            prompt_id = f"t{turn:02d}_p{prompt_index:02d}"
            paired_samples: list[dict[str, Any]] = []
            answers_a = result["model_a"]
            answers_b = result["model_b"]
            if len(answers_a) != len(answers_b):
                raise ValueError(f"Target sample counts differ for {prompt_id}")
            for sample_index, (answer_a, answer_b) in enumerate(zip(answers_a, answers_b)):
                pair_id = f"{prompt_id}_s{sample_index:02d}"
                observation = {
                    "pair_id": pair_id,
                    "turn": turn,
                    "phase": phase,
                    "test_purpose": test_purpose,
                    "prompt_id": prompt_id,
                    "prompt": result["prompt"],
                    "sample_index": sample_index,
                    "sampling_seed": result["sampling_seed"],
                    "responses": {"A": answer_a, "B": answer_b},
                }
                observations.append(observation)
                paired_samples.append(
                    {"pair_id": pair_id, "sample_index": sample_index, "model_a": answer_a, "model_b": answer_b}
                )
            tool_results.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": result["prompt"],
                    "phase": phase,
                    "samples": paired_samples,
                }
            )
        self._append_jsonl(self.observations_path, observations)
        self._append_jsonl(
            self.events_path,
            [
                {
                    "type": "send_messages",
                    "at_utc": utc_now(),
                    "turn": turn,
                    "phase": phase,
                    "test_purpose": test_purpose,
                    "prompt_ids": [result["prompt_id"] for result in tool_results],
                }
            ],
        )
        self._update_manifest(send_messages_calls=turn)
        return tool_results

    def known_pair_ids(self) -> set[str]:
        return {row["pair_id"] for row in read_jsonl(self.observations_path)}

    def validation_prompt_ids_for_pairs(self, pair_ids: set[str]) -> set[str]:
        return {
            row["prompt_id"]
            for row in read_jsonl(self.observations_path)
            if row["pair_id"] in pair_ids and row["phase"] == "validate"
        }

    def finish(self, report: dict[str, Any]) -> None:
        atomic_write_json(self.report_path, report)
        self._append_jsonl(
            self.events_path,
            [{"type": "end_conversation", "at_utc": utc_now(), "result": report["result"]}],
        )
        self._update_manifest(status="completed", completed_at_utc=utc_now())

    def fail(self, error: BaseException) -> None:
        self._append_jsonl(
            self.events_path,
            [{"type": "failure", "at_utc": utc_now(), "error_type": type(error).__name__, "message": str(error)}],
        )
        self._update_manifest(status="failed", failed_at_utc=utc_now())
