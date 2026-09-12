from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from model_diffing.agent import run_investigator
from model_diffing.config import load_diffing_config, validate_diffing_config
from model_diffing.inference import sampling_seed
from model_diffing.prompts import send_messages_tool, validate_final_report
from model_diffing.registry import TargetSpec, load_target_pair, parse_target_id
from model_diffing.session import SessionRecorder
from model_diffing.statistics import score_validation_codings, two_sided_binomial_pvalue
from safety_training.io import atomic_write_json, atomic_write_jsonl


def _target(path: Path, identifier: str, condition: str = "M0", seed: int | None = None) -> TargetSpec:
    return TargetSpec(
        id=identifier,
        condition=condition,
        training_seed=seed,
        run_dir=path,
        manifest_path=path / "manifest.json",
        adapter_path=None if condition == "M0" else path / "adapter",
        model_name="test/model",
        model_revision="commit",
        tokenizer_revision="commit",
        manifest_sha256=f"hash-{identifier}",
    )


def test_default_config_matches_bounded_agent_protocol():
    config = load_diffing_config()
    assert config["method"]["max_turns"] == 10
    assert config["method"]["max_prompts_per_turn"] == 5
    assert config["method"]["max_samples_per_prompt"] == 5
    assert config["method"]["alpha"] == 0.05
    assert config["method"]["min_validation_prompts"] == 6
    assert config["method"]["stateless_targets"] is True
    assert config["method"]["hide_target_reasoning"] is True


def test_config_rejects_more_than_five_samples():
    config = load_diffing_config()
    config["method"]["max_samples_per_prompt"] = 6
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_diffing_config(config)


def test_send_messages_schema_is_strict_and_bounded():
    tool = send_messages_tool(5, 5)
    assert tool["strict"] is True
    assert tool["parameters"]["additionalProperties"] is False
    assert tool["parameters"]["properties"]["prompts"]["maxItems"] == 5
    assert tool["parameters"]["properties"]["samples_per_model"]["maximum"] == 5


def test_target_id_and_revision_locked_pair_loading(tmp_path: Path):
    assert parse_target_id("M0") == ("M0", None)
    assert parse_target_id("M3_seed_42") == ("M3", 42)
    with pytest.raises(ValueError):
        parse_target_id("M3")

    m0 = tmp_path / "M0"
    m0.mkdir()
    atomic_write_json(
        m0 / "manifest.json",
        {
            "condition": "M0",
            "starting_model": "Qwen/test",
            "model_commit_hash": "abc",
            "tokenizer_revision": "abc",
        },
    )
    m1 = tmp_path / "M1" / "seed_42"
    (m1 / "adapter").mkdir(parents=True)
    atomic_write_json(
        m1 / "manifest.json",
        {
            "condition": "M1",
            "training_seed": 42,
            "starting_model": "Qwen/test",
            "starting_revision": "abc",
            "tokenizer_revision": "abc",
        },
    )
    first, second = load_target_pair(tmp_path, "M0", "M1_seed_42")
    assert first.adapter_path is None
    assert second.adapter_path == m1 / "adapter"


def test_sampling_seed_is_stable_and_turn_specific():
    assert sampling_seed(7, 1, 0) == sampling_seed(7, 1, 0)
    assert sampling_seed(7, 1, 0) != sampling_seed(7, 2, 0)


def test_no_difference_report_is_canonicalized():
    report = {
        "result": "no_difference_found",
        "hypothesis": "spurious",
        "when": "sometimes",
        "model_a_behavior": "x",
        "model_b_behavior": "y",
        "expected_model": "A",
        "quantitative_evidence": "none",
        "reproducibility": "none",
        "within_model_control": "none",
        "counterevidence_and_edge_cases": "none",
        "confidence": "low",
        "evidence_pair_ids": [],
    }
    result = validate_final_report(report, set())
    assert result["hypothesis"] is None
    assert result["expected_model"] is None


class _FakeBackend:
    def send_messages(self, prompts, samples_per_model, *, turn, sampling):
        return [
            {
                "prompt": prompt,
                "sampling_seed": turn,
                "model_a": [f"A answer {index}" for index in range(samples_per_model)],
                "model_b": [f"B answer {index}" for index in range(samples_per_model)],
            }
            for prompt in prompts
        ]


class _FakeResponses:
    def __init__(self):
        self.calls = 0

    def create(self, **request):
        self.calls += 1
        if self.calls == 1:
            item = SimpleNamespace(
                type="function_call",
                name="send_messages",
                call_id="call-1",
                arguments=json.dumps(
                    {
                        "prompts": ["seed prompt"],
                        "samples_per_model": 2,
                        "phase": "validate",
                        "test_purpose": "Test a visible response tendency.",
                    }
                ),
            )
            return SimpleNamespace(id="response-1", output=[item], output_text="")
        report = {
            "result": "difference_found",
            "hypothesis": "On the test prompt, A uses an A marker and B uses a B marker.",
            "when": "On the test prompt.",
            "model_a_behavior": "Uses an A marker.",
            "model_b_behavior": "Uses a B marker.",
            "expected_model": "A",
            "quantitative_evidence": "A did so in 2/2 samples and B in 0/2.",
            "reproducibility": "Observed in the requested samples.",
            "within_model_control": "Both samples within each model agreed.",
            "counterevidence_and_edge_cases": "Only one prompt was tested.",
            "confidence": "Low pending independent replication.",
            "evidence_pair_ids": ["t01_p00_s00", "t01_p00_s01"],
        }
        item = SimpleNamespace(
            type="function_call",
            name="end_conversation",
            call_id="call-2",
            arguments=json.dumps(report),
        )
        return SimpleNamespace(id="response-2", output=[item], output_text="")


def test_agent_tool_loop_records_only_visible_target_outputs(tmp_path: Path):
    config = load_diffing_config()
    config["method"]["min_validation_prompts"] = 1
    target = _target(tmp_path / "target", "M0")
    recorder = SessionRecorder.create(
        tmp_path,
        "session",
        config=config,
        target_a=target,
        target_b=target,
        seed_prompt_id="seed_001",
        seed_prompt="seed prompt",
        investigator_model="test-investigator",
        requested_pair=("M0", "M0"),
    )
    client = SimpleNamespace(responses=_FakeResponses())
    report = run_investigator(_FakeBackend(), recorder, config, "test-investigator", client=client)
    assert report["result"] == "difference_found"
    observations = (recorder.path / "observations.jsonl").read_text(encoding="utf-8")
    assert "A answer 0" in observations
    assert "chain_of_thought" not in observations
    assert recorder.manifest()["status"] == "completed"


def test_exact_binomial_and_held_out_scoring(tmp_path: Path):
    assert two_sided_binomial_pvalue(10, 10) == pytest.approx(2 / 1024)
    config = load_diffing_config()
    target = _target(tmp_path / "target", "M0")
    recorder = SessionRecorder.create(
        tmp_path,
        "scored",
        config=config,
        target_a=target,
        target_b=target,
        seed_prompt_id="seed_001",
        seed_prompt="seed prompt",
        investigator_model="test-investigator",
        requested_pair=("M0", "M0"),
    )
    raw = []
    for prompt_index in range(6):
        count = 2
        raw.append(
            {
                "prompt": f"validation {prompt_index}",
                "sampling_seed": prompt_index,
                "model_a": ["A"] * count,
                "model_b": ["B"] * count,
            }
        )
    tool_results = recorder.record_query(
        turn=1,
        phase="validate",
        test_purpose="held-out check",
        raw_results=raw,
    )
    pair_ids = [sample["pair_id"] for result in tool_results for sample in result["samples"]]
    recorder.finish(
        {
            "result": "difference_found",
            "hypothesis": "A has the coded behavior more often.",
            "when": "On held-out prompts.",
            "model_a_behavior": "Shows the behavior.",
            "model_b_behavior": "Does not show the behavior.",
            "expected_model": "A",
            "quantitative_evidence": "12 paired validations.",
            "reproducibility": "Six prompts.",
            "within_model_control": "Repeated samples agree.",
            "counterevidence_and_edge_cases": "None observed.",
            "confidence": "Moderate.",
            "evidence_pair_ids": pair_ids,
        }
    )
    codings_path = tmp_path / "codings.jsonl"
    atomic_write_jsonl(codings_path, [{"pair_id": pair_id, "verdict": "A"} for pair_id in pair_ids])
    metrics = score_validation_codings(recorder.path, codings_path)
    assert metrics["distinct_validation_prompts"] == 6
    assert metrics["two_sided_exact_prompt_level_binomial_p"] == pytest.approx(2 / 64)
    assert metrics["nominally_significant"] is True
    assert metrics["validated"] is True
