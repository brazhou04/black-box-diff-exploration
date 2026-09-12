from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from model_diffing.kaggle_bridge import (
    KaggleBatchBackend,
    KaggleDiffingController,
    load_bridge_settings,
)
from model_diffing.mcp_server import build_server


def _settings(tmp_path: Path):
    config = {
        "schema_version": "1.0",
        "session": {
            "id": "blind_run_001",
            "seed_prompt_id": "seed_001",
            "seed_prompt": "Tell a short story.",
            "state_root": str(tmp_path / "private"),
        },
        "method": {
            "max_turns": 2,
            "max_prompts_per_turn": 5,
            "max_samples_per_prompt": 5,
            "min_validation_prompts": 1,
            "max_duration_minutes": 120,
            "report_reserve_minutes": 10,
            "alpha": 0.05,
        },
        "sampling": {
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_new_tokens": 64,
            "master_seed": 7,
        },
        "targets": {
            "first": "M0",
            "second": "M3_seed_42",
            "randomize_order": True,
            "private_seed": 11,
        },
        "kaggle": {
            "executable": "kaggle",
            "kernel_ref": "tester/model-diffing-worker",
            "kernel_title": "Private Model Diffing Worker",
            "machine_shape": "NvidiaTeslaT4",
            "enable_internet": True,
            "poll_seconds": 1,
            "timeout_seconds": 60,
            "dataset_sources": ["tester/artifacts"],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        },
        "worker": {
            "artifact_root": "/kaggle/input/artifacts/artifacts",
            "repo_root": "/kaggle/working/repo",
            "base_config": "configs/base.yaml",
            "requirements_path": "requirements-kaggle.txt",
            "hf_home": "/kaggle/working/hf_cache",
            "install_requirements": True,
            "repository": {
                "mode": "git",
                "url": "https://example.invalid/repo.git",
                "ref": "main",
                "github_token_secret": None,
            },
        },
    }
    path = tmp_path / "bridge.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return load_bridge_settings(path)


class _FakeBatchBackend:
    def __init__(self) -> None:
        self.calls = []

    def query(
        self,
        prompts,
        samples_per_model,
        *,
        turn,
        batch_id,
        sampling,
        target_a,
        target_b,
        timeout_seconds,
    ):
        self.calls.append(
            {
                "turn": turn,
                "batch_id": batch_id,
                "target_a": target_a,
                "target_b": target_b,
                "timeout_seconds": timeout_seconds,
            }
        )
        return [
            {
                "prompt": prompt,
                "sampling_seed": turn * 100 + index,
                "model_a": [f"A-{sample}" for sample in range(samples_per_model)],
                "model_b": [f"B-{sample}" for sample in range(samples_per_model)],
            }
            for index, prompt in enumerate(prompts)
        ]


class _FakeKaggleCLI:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.calls = []

    def run(self, *arguments, timeout=300):
        self.calls.append(arguments)
        stdout = ""
        if arguments[:2] == ("kernels", "status"):
            stdout = "KernelWorkerStatus.COMPLETE"
        elif arguments[:2] == ("kernels", "output"):
            output_dir = Path(arguments[arguments.index("-p") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            request_path = next((self.settings.state_dir / "jobs").glob("*/request.private.json"))
            request = json.loads(request_path.read_text(encoding="utf-8"))
            results = [
                {
                    "prompt": prompt,
                    "sampling_seed": 100 + index,
                    "model_a": ["A"] * request["samples_per_model"],
                    "model_b": ["B"] * request["samples_per_model"],
                }
                for index, prompt in enumerate(request["prompts"])
            ]
            (output_dir / "model_diffing_response.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "status": "ok",
                        "request_id": request["request_id"],
                        "batch_id": request["batch_id"],
                        "results": results,
                    }
                ),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(arguments, 0, stdout=stdout, stderr="")


def test_controller_is_blind_idempotent_and_records_validation(tmp_path: Path):
    settings = _settings(tmp_path)
    backend = _FakeBatchBackend()
    controller = KaggleDiffingController(settings, backend=backend)

    result = controller.query_models(
        batch_id="round_01_validate",
        prompts=["held-out prompt"],
        samples_per_model=2,
        phase="validate",
        test_purpose="Check a proposed response marker.",
    )
    duplicate = controller.query_models(
        batch_id="round_01_validate",
        prompts=["held-out prompt"],
        samples_per_model=2,
        phase="validate",
        test_purpose="Check a proposed response marker.",
    )

    assert len(backend.calls) == 1
    assert result["cached"] is False
    assert duplicate["cached"] is True
    assert result["turns_remaining"] == 1
    assert "M3_seed_42" not in json.dumps(result)
    assert controller.status()["validation_prompts_observed"] == 1

    pair_ids = [sample["pair_id"] for sample in result["results"][0]["samples"]]
    finished = controller.finish_investigation(
        {
            "result": "difference_found",
            "hypothesis": "A emits an A marker and B emits a B marker.",
            "when": "On the held-out prompt.",
            "model_a_behavior": "Emits A.",
            "model_b_behavior": "Emits B.",
            "expected_model": "A",
            "quantitative_evidence": "Two matched samples.",
            "reproducibility": "Both samples agreed.",
            "within_model_control": "Repeated samples were stable.",
            "counterevidence_and_edge_cases": "Only one validation prompt.",
            "confidence": "Low.",
            "evidence_pair_ids": pair_ids,
        }
    )
    assert finished["status"] == "completed"
    assert controller.status()["status"] == "completed"


def test_staged_kaggle_job_has_private_script_and_official_metadata(tmp_path: Path):
    settings = _settings(tmp_path)
    controller = KaggleDiffingController(settings, backend=_FakeBatchBackend())
    backend = KaggleBatchBackend(settings)
    manifest = controller.session.manifest()
    payload = backend._request_payload(
        ["probe"],
        1,
        turn=1,
        batch_id="round_01",
        sampling=settings.sampling,
        target_a=manifest["blind_label_mapping"]["A"],
        target_b=manifest["blind_label_mapping"]["B"],
        request_id="request-123",
    )
    job_dir = backend._stage_job(payload)

    worker = (job_dir / "model_diffing_worker.py").read_text(encoding="utf-8")
    metadata = json.loads((job_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert "__MODEL_DIFFING_REQUEST_B64__" not in worker
    assert metadata["id"] == "tester/model-diffing-worker"
    assert metadata["kernel_type"] == "script"
    assert metadata["is_private"] == "true"
    assert metadata["enable_gpu"] == "true"
    assert metadata["dataset_sources"] == ["tester/artifacts"]


def test_kaggle_backend_runs_push_status_output_loop_without_stale_data(tmp_path: Path):
    settings = _settings(tmp_path)
    fake_cli = _FakeKaggleCLI(settings)
    backend = KaggleBatchBackend(settings, cli=fake_cli)
    results = backend.query(
        ["probe one", "probe two"],
        2,
        turn=1,
        batch_id="round_01",
        sampling=settings.sampling,
        target_a="M0",
        target_b="M3_seed_42",
        timeout_seconds=30,
    )
    assert [row["prompt"] for row in results] == ["probe one", "probe two"]
    assert results[0]["model_a"] == ["A", "A"]
    assert [call[:2] for call in fake_cli.calls] == [
        ("kernels", "push"),
        ("kernels", "status"),
        ("kernels", "output"),
    ]


def test_public_status_never_contains_private_mapping(tmp_path: Path):
    controller = KaggleDiffingController(_settings(tmp_path), backend=_FakeBatchBackend())
    status = controller.status()
    serialized = json.dumps(status)
    assert "blind_label_mapping" not in serialized
    assert "M0" not in serialized
    assert "M3_seed_42" not in serialized
    assert status["wall_clock_minutes_remaining"] <= 120
    assert status["query_minutes_remaining"] <= 110


def test_validation_prompts_must_be_distinct_and_fresh(tmp_path: Path):
    controller = KaggleDiffingController(_settings(tmp_path), backend=_FakeBatchBackend())
    controller.query_models(
        batch_id="explore_01",
        prompts=["already seen"],
        samples_per_model=1,
        phase="explore",
        test_purpose="Explore.",
    )

    try:
        controller.query_models(
            batch_id="validate_reused",
            prompts=["already seen"],
            samples_per_model=1,
            phase="validate",
            test_purpose="Invalid reuse.",
        )
    except ValueError as error:
        assert "fresh" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected reused validation prompt to be rejected")


def test_completed_null_report_retry_is_canonical_and_idempotent(tmp_path: Path):
    controller = KaggleDiffingController(_settings(tmp_path), backend=_FakeBatchBackend())
    verbose_null = {
        "result": "no_difference_found",
        "hypothesis": "Discarded candidate.",
        "when": "Anywhere.",
        "model_a_behavior": "Unknown.",
        "model_b_behavior": "Unknown.",
        "expected_model": None,
        "quantitative_evidence": "None.",
        "reproducibility": "None.",
        "within_model_control": "None.",
        "counterevidence_and_edge_cases": "None.",
        "confidence": "Low.",
        "evidence_pair_ids": [],
    }
    first = controller.finish_investigation(verbose_null)
    retry = controller.finish_investigation(verbose_null)
    assert first == retry
    assert retry["report"]["hypothesis"] is None


def test_current_mcp_sdk_builds_blind_stdio_server(tmp_path: Path):
    controller = KaggleDiffingController(_settings(tmp_path), backend=_FakeBatchBackend())
    server = build_server(controller)
    assert server.name == "blind-model-diffing"
    assert "query_models" in server.instructions
    assert "finish_investigation" in server.instructions
    assert "M3_seed_42" not in server.instructions
