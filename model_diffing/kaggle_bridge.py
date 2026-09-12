from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import subprocess
import sysconfig
import time
import uuid
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

import yaml

from safety_training.io import atomic_write_json, atomic_write_jsonl, read_jsonl, sha256_json

from .prompts import validate_final_report


WORKER_TEMPLATE = Path(__file__).with_name("kaggle_worker_template.py")
_BATCH_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_TERMINAL_FAILURES = ("error", "failed", "failure", "cancelled", "canceled")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _positive_int(value: Any, label: str, *, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if parsed < 1 or (maximum is not None and parsed > maximum):
        suffix = f" and at most {maximum}" if maximum is not None else ""
        raise ValueError(f"{label} must be positive{suffix}")
    return parsed


def _config_path(value: Any, *, relative_to: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path")
    expanded = Path(os.path.expandvars(value)).expanduser()
    return (relative_to / expanded).resolve() if not expanded.is_absolute() else expanded.resolve()


@dataclass(frozen=True)
class BridgeSettings:
    source_path: Path
    raw: dict[str, Any]
    session_id: str
    seed_prompt_id: str
    seed_prompt: str
    state_root: Path
    max_turns: int
    max_prompts_per_turn: int
    max_samples_per_prompt: int
    min_validation_prompts: int
    max_duration_seconds: int
    report_reserve_seconds: int
    alpha: float
    sampling: dict[str, Any]
    target_first: str
    target_second: str
    randomize_order: bool
    private_seed: int
    kaggle: dict[str, Any]
    worker: dict[str, Any]
    config_sha256: str

    @property
    def state_dir(self) -> Path:
        return self.state_root / self.session_id


def load_bridge_settings(path: str | Path) -> BridgeSettings:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = _mapping(raw, "bridge config")

    session = _mapping(raw.get("session"), "session")
    session_id = str(session.get("id", ""))
    if not _SESSION_ID.fullmatch(session_id):
        raise ValueError("session.id must contain only letters, digits, underscores, and hyphens")
    seed_prompt_id = str(session.get("seed_prompt_id", "custom"))
    seed_prompt = session.get("seed_prompt")
    if not isinstance(seed_prompt, str) or not seed_prompt.strip():
        raise ValueError("session.seed_prompt must be a non-empty string")
    state_root = _config_path(session.get("state_root"), relative_to=source.parent, label="session.state_root")

    method = _mapping(raw.get("method"), "method")
    max_turns = _positive_int(method.get("max_turns"), "method.max_turns", maximum=10)
    max_prompts = _positive_int(
        method.get("max_prompts_per_turn"), "method.max_prompts_per_turn", maximum=5
    )
    max_samples = _positive_int(
        method.get("max_samples_per_prompt"), "method.max_samples_per_prompt", maximum=5
    )
    min_validation = _positive_int(method.get("min_validation_prompts"), "method.min_validation_prompts")
    max_duration_minutes = _positive_int(
        method.get("max_duration_minutes", 120), "method.max_duration_minutes", maximum=120
    )
    report_reserve_minutes = _positive_int(
        method.get("report_reserve_minutes", 10),
        "method.report_reserve_minutes",
        maximum=max_duration_minutes,
    )
    if report_reserve_minutes >= max_duration_minutes:
        raise ValueError("method.report_reserve_minutes must be less than method.max_duration_minutes")
    alpha = float(method.get("alpha", 0.05))
    if not 0 < alpha < 1:
        raise ValueError("method.alpha must be in (0, 1)")

    sampling = _mapping(raw.get("sampling"), "sampling")
    if not sampling.get("do_sample"):
        raise ValueError("sampling.do_sample must be true for repeated black-box sampling")
    if float(sampling.get("temperature", 0)) <= 0:
        raise ValueError("sampling.temperature must be positive")
    top_p = float(sampling.get("top_p", 0))
    if not 0 < top_p <= 1:
        raise ValueError("sampling.top_p must be in (0, 1]")
    _positive_int(sampling.get("max_new_tokens"), "sampling.max_new_tokens")
    if not isinstance(sampling.get("master_seed"), int):
        raise ValueError("sampling.master_seed must be an integer")

    targets = _mapping(raw.get("targets"), "targets")
    first = targets.get("first")
    second = targets.get("second")
    if not isinstance(first, str) or not first.strip() or not isinstance(second, str) or not second.strip():
        raise ValueError("targets.first and targets.second must be non-empty target IDs")
    private_seed = targets.get("private_seed")
    if not isinstance(private_seed, int):
        raise ValueError("targets.private_seed must be an integer")

    kaggle = _mapping(raw.get("kaggle"), "kaggle")
    kernel_ref = kaggle.get("kernel_ref")
    if not isinstance(kernel_ref, str) or not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", kernel_ref):
        raise ValueError("kaggle.kernel_ref must have the form username/kernel-slug")
    _positive_int(kaggle.get("poll_seconds", 20), "kaggle.poll_seconds", maximum=60)
    _positive_int(kaggle.get("timeout_seconds", 3600), "kaggle.timeout_seconds")
    for key in ("dataset_sources", "kernel_sources", "model_sources", "competition_sources"):
        values = kaggle.get(key, [])
        if not isinstance(values, list) or any(not isinstance(item, str) or not item for item in values):
            raise ValueError(f"kaggle.{key} must be a list of non-empty strings")

    worker = _mapping(raw.get("worker"), "worker")
    artifact_root = worker.get("artifact_root")
    repo_root = worker.get("repo_root")
    if not isinstance(artifact_root, str) or not artifact_root.startswith("/kaggle/"):
        raise ValueError("worker.artifact_root must be an absolute /kaggle/... path")
    if not isinstance(repo_root, str) or not repo_root.startswith("/kaggle/"):
        raise ValueError("worker.repo_root must be an absolute /kaggle/... path")
    repository = _mapping(worker.get("repository"), "worker.repository")
    mode = repository.get("mode")
    if mode not in {"git", "path"}:
        raise ValueError("worker.repository.mode must be 'git' or 'path'")
    if mode == "git" and (not isinstance(repository.get("url"), str) or not repository["url"].strip()):
        raise ValueError("worker.repository.url is required in git mode")
    if mode == "git" and not repo_root.startswith("/kaggle/working/"):
        raise ValueError("worker.repo_root must be below /kaggle/working in git mode")

    return BridgeSettings(
        source_path=source,
        raw=raw,
        session_id=session_id,
        seed_prompt_id=seed_prompt_id,
        seed_prompt=seed_prompt.strip(),
        state_root=state_root,
        max_turns=max_turns,
        max_prompts_per_turn=max_prompts,
        max_samples_per_prompt=max_samples,
        min_validation_prompts=min_validation,
        max_duration_seconds=max_duration_minutes * 60,
        report_reserve_seconds=report_reserve_minutes * 60,
        alpha=alpha,
        sampling=dict(sampling),
        target_first=first,
        target_second=second,
        randomize_order=bool(targets.get("randomize_order", True)),
        private_seed=private_seed,
        kaggle=dict(kaggle),
        worker=dict(worker),
        config_sha256=sha256_json(raw),
    )


class BatchBackend(Protocol):
    def query(
        self,
        prompts: list[str],
        samples_per_model: int,
        *,
        turn: int,
        batch_id: str,
        sampling: dict[str, Any],
        target_a: str,
        target_b: str,
        timeout_seconds: int,
    ) -> list[dict[str, Any]]: ...


class KaggleCommandError(RuntimeError):
    pass


class KaggleCLI:
    def __init__(self, executable: str = "kaggle") -> None:
        if executable == "kaggle":
            executable_name = "kaggle.exe" if os.name == "nt" else "kaggle"
            environment_executable = Path(sysconfig.get_path("scripts")) / executable_name
            self.executable = str(environment_executable) if environment_executable.is_file() else executable
        else:
            self.executable = executable

    def run(self, *arguments: str, timeout: int = 300) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                [self.executable, *arguments],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError as error:
            raise KaggleCommandError(
                "Kaggle CLI was not found. Install the bridge extra and run `kaggle auth login`."
            ) from error
        except subprocess.TimeoutExpired as error:
            raise KaggleCommandError(f"Kaggle command timed out: {arguments[0] if arguments else ''}") from error
        if result.returncode:
            detail = (result.stderr or result.stdout or "unknown Kaggle CLI error").strip()
            raise KaggleCommandError(f"Kaggle command failed ({arguments[0] if arguments else ''}): {detail}")
        return result


class KaggleBatchBackend:
    def __init__(self, settings: BridgeSettings, *, cli: KaggleCLI | None = None) -> None:
        self.settings = settings
        self.cli = cli or KaggleCLI(str(settings.kaggle.get("executable", "kaggle")))

    def _request_payload(
        self,
        prompts: list[str],
        samples_per_model: int,
        *,
        turn: int,
        batch_id: str,
        sampling: dict[str, Any],
        target_a: str,
        target_b: str,
        request_id: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "request_id": request_id,
            "batch_id": batch_id,
            "turn": turn,
            "prompts": prompts,
            "samples_per_model": samples_per_model,
            "sampling": sampling,
            "target_a": target_a,
            "target_b": target_b,
            "worker": self.settings.worker,
        }

    def _stage_job(self, payload: dict[str, Any]) -> Path:
        request_id = str(payload["request_id"])
        batch_id = str(payload["batch_id"])
        turn = int(payload["turn"])
        job_dir = self.settings.state_dir / "jobs" / f"t{turn:02d}_{batch_id}_{request_id[:8]}"
        if job_dir.exists():
            raise FileExistsError(f"Refusing to overwrite staged Kaggle job: {job_dir}")
        job_dir.mkdir(parents=True)

        encoded = base64.b64encode(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
        template = WORKER_TEMPLATE.read_text(encoding="utf-8")
        marker = "__MODEL_DIFFING_REQUEST_B64__"
        if template.count(marker) != 1:
            raise RuntimeError("Kaggle worker template has an invalid request marker")
        (job_dir / "model_diffing_worker.py").write_text(template.replace(marker, encoded), encoding="utf-8")

        kaggle = self.settings.kaggle
        metadata = {
            "id": kaggle["kernel_ref"],
            "title": str(kaggle.get("kernel_title", "Private Model Diffing Worker")),
            "code_file": "model_diffing_worker.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true" if bool(kaggle.get("enable_internet", True)) else "false",
            "machine_shape": str(kaggle.get("machine_shape", "NvidiaTeslaT4")),
            "dataset_sources": list(kaggle.get("dataset_sources", [])),
            "competition_sources": list(kaggle.get("competition_sources", [])),
            "kernel_sources": list(kaggle.get("kernel_sources", [])),
            "model_sources": list(kaggle.get("model_sources", [])),
        }
        atomic_write_json(job_dir / "kernel-metadata.json", metadata)
        atomic_write_json(job_dir / "request.private.json", payload)
        return job_dir

    @staticmethod
    def _write_cli_log(job_dir: Path, name: str, result: subprocess.CompletedProcess[str]) -> None:
        (job_dir / name).write_text(
            f"returncode={result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}",
            encoding="utf-8",
        )

    def _run_private(
        self,
        job_dir: Path,
        log_name: str,
        *arguments: str,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = self.cli.run(*arguments, timeout=timeout)
        except KaggleCommandError as error:
            (job_dir / log_name).write_text(str(error), encoding="utf-8")
            raise RuntimeError(
                "Kaggle bridge command failed; inspect the private logs outside the blind workspace"
            ) from error
        self._write_cli_log(job_dir, log_name, result)
        return result

    @staticmethod
    def _is_complete(status: str) -> bool:
        lowered = status.lower()
        return "complete" in lowered or "success" in lowered

    @staticmethod
    def _is_failed(status: str) -> bool:
        lowered = status.lower()
        return any(marker in lowered for marker in _TERMINAL_FAILURES)

    @staticmethod
    def _remaining_timeout(deadline: float, maximum: int) -> int:
        remaining = math.ceil(deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError("The Kaggle query exceeded its allotted investigation time")
        return min(maximum, remaining)

    def _download_response(
        self, job_dir: Path, request_id: str, *, deadline: float
    ) -> list[dict[str, Any]] | None:
        output_dir = job_dir / "output"
        output_dir.mkdir(exist_ok=True)
        result = self._run_private(
            job_dir,
            "output.log",
            "kernels",
            "output",
            str(self.settings.kaggle["kernel_ref"]),
            "-p",
            str(output_dir),
            "-o",
            "--file-pattern",
            r"^model_diffing_response\.json$",
            timeout=self._remaining_timeout(deadline, 300),
        )
        response_path = output_dir / "model_diffing_response.json"
        if not response_path.is_file():
            return None
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if response.get("request_id") != request_id:
            return None
        if response.get("schema_version") != "1.0" or response.get("status") != "ok":
            raise RuntimeError("Kaggle worker returned an invalid or failed response")
        results = response.get("results")
        if not isinstance(results, list):
            raise RuntimeError("Kaggle worker response.results must be a list")
        return results

    @staticmethod
    def _validate_results(
        results: list[dict[str, Any]], prompts: list[str], samples_per_model: int
    ) -> list[dict[str, Any]]:
        if len(results) != len(prompts):
            raise RuntimeError("Kaggle worker returned the wrong number of prompt results")
        validated: list[dict[str, Any]] = []
        for prompt, result in zip(prompts, results):
            if not isinstance(result, dict) or result.get("prompt") != prompt:
                raise RuntimeError("Kaggle worker returned prompts out of order")
            answers_a = result.get("model_a")
            answers_b = result.get("model_b")
            if not isinstance(answers_a, list) or not isinstance(answers_b, list):
                raise RuntimeError("Kaggle worker responses must be lists")
            if len(answers_a) != samples_per_model or len(answers_b) != samples_per_model:
                raise RuntimeError("Kaggle worker returned the wrong sample count")
            if any(not isinstance(item, str) for item in [*answers_a, *answers_b]):
                raise RuntimeError("Kaggle worker returned a non-text response")
            if not isinstance(result.get("sampling_seed"), int):
                raise RuntimeError("Kaggle worker omitted the sampling seed")
            validated.append(
                {
                    "prompt": prompt,
                    "sampling_seed": result["sampling_seed"],
                    "model_a": answers_a,
                    "model_b": answers_b,
                }
            )
        return validated

    def query(
        self,
        prompts: list[str],
        samples_per_model: int,
        *,
        turn: int,
        batch_id: str,
        sampling: dict[str, Any],
        target_a: str,
        target_b: str,
        timeout_seconds: int,
    ) -> list[dict[str, Any]]:
        if timeout_seconds < 1:
            raise TimeoutError("No investigation time remains for another Kaggle query")
        deadline = time.monotonic() + timeout_seconds
        request_id = str(uuid.uuid4())
        payload = self._request_payload(
            prompts,
            samples_per_model,
            turn=turn,
            batch_id=batch_id,
            sampling=sampling,
            target_a=target_a,
            target_b=target_b,
            request_id=request_id,
        )
        job_dir = self._stage_job(payload)
        self._run_private(
            job_dir,
            "push.log",
            "kernels",
            "push",
            "-p",
            str(job_dir),
            timeout=self._remaining_timeout(deadline, 300),
        )

        poll_seconds = int(self.settings.kaggle.get("poll_seconds", 20))
        status_count = 0
        while time.monotonic() < deadline:
            status = self._run_private(
                job_dir,
                f"status_{status_count + 1:04d}.log",
                "kernels",
                "status",
                str(self.settings.kaggle["kernel_ref"]),
                timeout=self._remaining_timeout(deadline, 120),
            )
            status_count += 1
            status_text = f"{status.stdout}\n{status.stderr}"
            if self._is_failed(status_text):
                try:
                    logs = self.cli.run(
                        "kernels",
                        "logs",
                        str(self.settings.kaggle["kernel_ref"]),
                        timeout=self._remaining_timeout(deadline, 180),
                    )
                    self._write_cli_log(job_dir, "kernel_failure.log", logs)
                except KaggleCommandError as error:
                    (job_dir / "kernel_failure.log").write_text(str(error), encoding="utf-8")
                raise RuntimeError(
                    "Kaggle worker failed; inspect the private logs outside the blind workspace"
                )
            if self._is_complete(status_text):
                results = self._download_response(job_dir, request_id, deadline=deadline)
                if results is not None:
                    return self._validate_results(results, prompts, samples_per_model)
            time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
        raise TimeoutError(f"Kaggle worker did not return request {request_id} before the configured timeout")


class BridgeSession:
    def __init__(self, settings: BridgeSettings) -> None:
        self.settings = settings
        self.path = settings.state_dir
        self.manifest_path = self.path / "manifest.private.json"
        self.observations_path = self.path / "observations.jsonl"
        self.batches_path = self.path / "batches.private.jsonl"
        self.events_path = self.path / "events.jsonl"
        self.report_path = self.path / "report.json"
        self._initialize()

    def _initialize(self) -> None:
        if self.path.exists():
            if not self.manifest_path.is_file():
                raise FileExistsError(f"Bridge session directory has no manifest: {self.path}")
            manifest = self.manifest()
            if manifest.get("config_sha256") != self.settings.config_sha256:
                raise ValueError(
                    "Bridge config changed after this session started; use a new session.id to preserve provenance"
                )
            return

        self.path.mkdir(parents=True)
        first, second = self.settings.target_first, self.settings.target_second
        if self.settings.randomize_order:
            rng = random.Random(f"{self.settings.private_seed}:{self.settings.session_id}")
            if rng.randrange(2):
                first, second = second, first
        atomic_write_json(
            self.manifest_path,
            {
                "schema_version": "1.0",
                "session_id": self.settings.session_id,
                "status": "running",
                "created_at_utc": utc_now(),
                "updated_at_utc": utc_now(),
                "config_path": str(self.settings.source_path),
                "config_sha256": self.settings.config_sha256,
                "seed_prompt_id": self.settings.seed_prompt_id,
                "seed_prompt": self.settings.seed_prompt,
                "blind_label_mapping": {"A": first, "B": second},
                "send_messages_calls": 0,
            },
        )
        atomic_write_jsonl(self.observations_path, [])
        atomic_write_jsonl(self.batches_path, [])
        atomic_write_jsonl(self.events_path, [])

    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def update_manifest(self, **updates: Any) -> None:
        manifest = self.manifest()
        manifest.update(updates)
        manifest["updated_at_utc"] = utc_now()
        atomic_write_json(self.manifest_path, manifest)

    def remaining_seconds(self) -> int:
        manifest = self.manifest()
        created_at = datetime.fromisoformat(str(manifest["created_at_utc"]))
        elapsed = max(0.0, (datetime.now(timezone.utc) - created_at).total_seconds())
        return max(0, math.floor(self.settings.max_duration_seconds - elapsed))

    @staticmethod
    def append(path: Path, rows: list[dict[str, Any]]) -> None:
        existing = read_jsonl(path) if path.exists() else []
        atomic_write_jsonl(path, [*existing, *rows])

    def public_status(self) -> dict[str, Any]:
        manifest = self.manifest()
        observations = read_jsonl(self.observations_path)
        validation_prompts = {row["prompt"] for row in observations if row["phase"] == "validate"}
        calls = int(manifest.get("send_messages_calls", 0))
        remaining_seconds = self.remaining_seconds()
        return {
            "session_id": self.settings.session_id,
            "status": manifest["status"],
            "seed_prompt_id": self.settings.seed_prompt_id,
            "seed_prompt": self.settings.seed_prompt,
            "turns_used": calls,
            "turns_remaining": max(0, self.settings.max_turns - calls),
            "validation_prompts_observed": len(validation_prompts),
            "minimum_validation_prompts": self.settings.min_validation_prompts,
            "wall_clock_minutes_remaining": round(remaining_seconds / 60, 1),
            "query_minutes_remaining": round(
                max(0, remaining_seconds - self.settings.report_reserve_seconds) / 60, 1
            ),
            "report_reserve_minutes": round(self.settings.report_reserve_seconds / 60, 1),
        }


class KaggleDiffingController:
    def __init__(self, settings: BridgeSettings, *, backend: BatchBackend | None = None) -> None:
        self.settings = settings
        self.session = BridgeSession(settings)
        self.backend = backend or KaggleBatchBackend(settings)
        self._lock = Lock()

    def status(self) -> dict[str, Any]:
        return self.session.public_status()

    def _cached_batch(self, batch_id: str, input_hash: str) -> dict[str, Any] | None:
        for row in read_jsonl(self.session.batches_path):
            if row["batch_id"] != batch_id:
                continue
            if row["input_sha256"] != input_hash:
                raise ValueError(f"batch_id {batch_id!r} was already used with different arguments")
            result = dict(row["public_result"])
            result["cached"] = True
            return result
        return None

    def query_models(
        self,
        *,
        batch_id: str,
        prompts: list[str],
        samples_per_model: int,
        phase: str,
        test_purpose: str,
    ) -> dict[str, Any]:
        with self._lock:
            if not _BATCH_ID.fullmatch(batch_id):
                raise ValueError("batch_id must contain 1-64 letters, digits, underscores, or hyphens")
            if phase not in {"explore", "refine", "validate"}:
                raise ValueError("phase must be explore, refine, or validate")
            if not isinstance(test_purpose, str) or not test_purpose.strip():
                raise ValueError("test_purpose must be a non-empty string")
            if not isinstance(prompts, list) or not 1 <= len(prompts) <= self.settings.max_prompts_per_turn:
                raise ValueError(f"prompts must contain 1-{self.settings.max_prompts_per_turn} strings")
            if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
                raise ValueError("prompts must contain only non-empty strings")
            if len(set(prompts)) != len(prompts):
                raise ValueError("prompts in one batch must be distinct")
            if any(len(prompt) > 20_000 for prompt in prompts) or sum(map(len, prompts)) > 50_000:
                raise ValueError("prompt batch is too large")
            if isinstance(samples_per_model, bool) or not isinstance(samples_per_model, int):
                raise ValueError("samples_per_model must be an integer")
            if not 1 <= samples_per_model <= self.settings.max_samples_per_prompt:
                raise ValueError(f"samples_per_model must be 1-{self.settings.max_samples_per_prompt}")

            request_shape = {
                "prompts": prompts,
                "samples_per_model": int(samples_per_model),
                "phase": phase,
                "test_purpose": test_purpose.strip(),
            }
            input_hash = sha256_json(request_shape)
            cached = self._cached_batch(batch_id, input_hash)
            if cached is not None:
                return cached

            prior_observations = read_jsonl(self.session.observations_path)
            if phase == "validate":
                prior_prompts = {row["prompt"] for row in prior_observations}
                reused = [prompt for prompt in prompts if prompt in prior_prompts]
                if reused:
                    raise ValueError("validation prompts must be fresh and not used in an earlier query")

            manifest = self.session.manifest()
            if manifest["status"] != "running":
                raise RuntimeError("This investigation is already completed")
            query_seconds_remaining = (
                self.session.remaining_seconds() - self.settings.report_reserve_seconds
            )
            if query_seconds_remaining < 1:
                raise RuntimeError(
                    "The experimental time budget is exhausted; finish the investigation now"
                )
            turn = int(manifest.get("send_messages_calls", 0)) + 1
            if turn > self.settings.max_turns:
                raise RuntimeError("The investigation query budget is exhausted; finish the investigation")
            mapping = manifest["blind_label_mapping"]
            raw_results = self.backend.query(
                prompts,
                int(samples_per_model),
                turn=turn,
                batch_id=batch_id,
                sampling=self.settings.sampling,
                target_a=mapping["A"],
                target_b=mapping["B"],
                timeout_seconds=min(
                    int(self.settings.kaggle.get("timeout_seconds", 900)),
                    query_seconds_remaining,
                ),
            )

            observations: list[dict[str, Any]] = []
            public_results: list[dict[str, Any]] = []
            for prompt_index, result in enumerate(raw_results):
                prompt_id = f"t{turn:02d}_p{prompt_index:02d}"
                samples: list[dict[str, Any]] = []
                answers_a = result["model_a"]
                answers_b = result["model_b"]
                if len(answers_a) != len(answers_b):
                    raise RuntimeError(f"Target sample counts differ for {prompt_id}")
                for sample_index, (answer_a, answer_b) in enumerate(zip(answers_a, answers_b)):
                    pair_id = f"{prompt_id}_s{sample_index:02d}"
                    observations.append(
                        {
                            "pair_id": pair_id,
                            "turn": turn,
                            "phase": phase,
                            "test_purpose": test_purpose.strip(),
                            "prompt_id": prompt_id,
                            "prompt": result["prompt"],
                            "sample_index": sample_index,
                            "sampling_seed": result["sampling_seed"],
                            "responses": {"A": answer_a, "B": answer_b},
                        }
                    )
                    samples.append(
                        {
                            "pair_id": pair_id,
                            "sample_index": sample_index,
                            "model_a": answer_a,
                            "model_b": answer_b,
                        }
                    )
                public_results.append(
                    {
                        "prompt_id": prompt_id,
                        "prompt": result["prompt"],
                        "phase": phase,
                        "samples": samples,
                    }
                )

            public_result = {
                "session_id": self.settings.session_id,
                "batch_id": batch_id,
                "turn": turn,
                "phase": phase,
                "turns_remaining": self.settings.max_turns - turn,
                "wall_clock_minutes_remaining": round(self.session.remaining_seconds() / 60, 1),
                "results": public_results,
                "cached": False,
            }
            self.session.append(self.session.observations_path, observations)
            self.session.append(
                self.session.batches_path,
                [
                    {
                        "batch_id": batch_id,
                        "input_sha256": input_hash,
                        "at_utc": utc_now(),
                        "public_result": public_result,
                    }
                ],
            )
            self.session.append(
                self.session.events_path,
                [
                    {
                        "type": "query_models",
                        "at_utc": utc_now(),
                        "turn": turn,
                        "batch_id": batch_id,
                        "phase": phase,
                        "prompt_ids": [item["prompt_id"] for item in public_results],
                    }
                ],
            )
            self.session.update_manifest(send_messages_calls=turn)
            return public_result

    def finish_investigation(self, report: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            manifest = self.session.manifest()
            observations = read_jsonl(self.session.observations_path)
            known_pair_ids = {row["pair_id"] for row in observations}
            validated = validate_final_report(report, known_pair_ids)
            if manifest["status"] == "completed":
                existing = json.loads(self.session.report_path.read_text(encoding="utf-8"))
                if existing != validated:
                    raise RuntimeError("This investigation already has a different final report")
                return {"session_id": self.settings.session_id, "status": "completed", "report": existing}

            if validated["result"] == "difference_found":
                cited = set(validated["evidence_pair_ids"])
                validation_prompts = {
                    row["prompt"]
                    for row in observations
                    if row["pair_id"] in cited and row["phase"] == "validate"
                }
                if len(validation_prompts) < self.settings.min_validation_prompts:
                    raise ValueError(
                        "A finding must cite held-out evidence from at least "
                        f"{self.settings.min_validation_prompts} validation prompts; found "
                        f"{len(validation_prompts)}"
                    )
            atomic_write_json(self.session.report_path, validated)
            self.session.append(
                self.session.events_path,
                [{"type": "finish_investigation", "at_utc": utc_now(), "result": validated["result"]}],
            )
            self.session.update_manifest(status="completed", completed_at_utc=utc_now())
            return {"session_id": self.settings.session_id, "status": "completed", "report": validated}


def bridge_preflight(settings: BridgeSettings, *, online: bool = False) -> dict[str, Any]:
    cli = KaggleCLI(str(settings.kaggle.get("executable", "kaggle")))
    version = cli.run("--version", timeout=60)
    result: dict[str, Any] = {
        "status": "READY_OFFLINE",
        "session_id": settings.session_id,
        "state_root": str(settings.state_root),
        "worker_template": str(WORKER_TEMPLATE),
        "kaggle_cli": (version.stdout or version.stderr).strip(),
        "online_auth_checked": False,
    }
    if online:
        cli.run("kernels", "list", "--mine", "--page-size", "1", timeout=120)
        result["status"] = "READY"
        result["online_auth_checked"] = True
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Private Kaggle batch bridge for the model-diffing MCP server")
    parser.add_argument("--config", default=os.environ.get("MODEL_DIFF_KAGGLE_CONFIG"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Validate bridge config and Kaggle CLI")
    preflight.add_argument("--online", action="store_true", help="Also verify Kaggle authentication over the network")

    subparsers.add_parser("status", help="Show only investigator-safe session state")

    query = subparsers.add_parser("query", help="Submit one prompt batch without MCP (diagnostic use)")
    query.add_argument("--batch-id", required=True)
    query.add_argument("--prompt", action="append", required=True)
    query.add_argument("--samples-per-model", type=int, default=1)
    query.add_argument("--phase", choices=("explore", "refine", "validate"), required=True)
    query.add_argument("--test-purpose", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.config:
        raise SystemExit("Supply --config or set MODEL_DIFF_KAGGLE_CONFIG")
    settings = load_bridge_settings(args.config)
    if args.command == "preflight":
        output = bridge_preflight(settings, online=bool(args.online))
    else:
        controller = KaggleDiffingController(settings)
        if args.command == "status":
            output = controller.status()
        elif args.command == "query":
            output = controller.query_models(
                batch_id=args.batch_id,
                prompts=args.prompt,
                samples_per_model=args.samples_per_model,
                phase=args.phase,
                test_purpose=args.test_purpose,
            )
        else:  # pragma: no cover
            raise AssertionError(args.command)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
