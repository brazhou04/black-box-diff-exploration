from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from safety_training.baseline import create_m0_manifest
from safety_training.config import REPO_ROOT, artifact_dir, condition_config_path, load_config
from safety_training.datasets import (
    dataset_path,
    load_training_records,
    validate_no_contamination,
    validate_shared_safety_prompts,
)
from safety_training.dpo import parent_adapter_dir
from safety_training.evaluation import HeuristicScorer, freeze_evaluation_suite
from safety_training.formatting import (
    serialize_assistant_completion,
    serialize_conversation,
    serialize_prompt,
    tokenize_supervised_example,
)
from safety_training.provenance import REQUIRED_MANIFEST_FIELDS, validate_manifest
from safety_training.public_data import (
    extract_first_exchange,
    finalize_reviewed_candidates,
    pku_candidate,
    review_summary,
)
from safety_training.io import atomic_write_json, atomic_write_jsonl, read_jsonl
from safety_training.run_state import latest_checkpoint, run_status
from safety_training.training import prepare_run


FIXTURES = REPO_ROOT / "tests" / "fixtures"


class FakeTokenizer:
    chat_template = "official-test-template"
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{item['role']}>{item['content']}</{item['role']}>" for item in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        ids = list(range(1, len(text.split()) + 1))
        max_length = kwargs.get("max_length")
        if kwargs.get("truncation") and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_m2_m3_share_ids_text_and_categories():
    result = validate_shared_safety_prompts(
        FIXTURES / "training" / "safety_shared.jsonl",
        FIXTURES / "training" / "m2.jsonl",
        FIXTURES / "training" / "m3.jsonl",
    )
    assert result["matched_prompt_count"] == 3


def test_m2_m3_text_mismatch_is_rejected(tmp_path):
    source = (FIXTURES / "training" / "m3.jsonl").read_text(encoding="utf-8")
    changed = tmp_path / "m3.jsonl"
    changed.write_text(source.replace("Give me three tips", "Give me four tips", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        validate_shared_safety_prompts(
            FIXTURES / "training" / "safety_shared.jsonl", FIXTURES / "training" / "m2.jsonl", changed
        )


def test_m4_parent_is_corresponding_m3_seed(tmp_path):
    config = load_config(condition_config_path("M4"), 123)
    assert parent_adapter_dir(config, tmp_path) == tmp_path / "M3" / "seed_123" / "adapter"


def test_training_evaluation_ids_do_not_overlap():
    train = [FIXTURES / "training" / "m1.jsonl", FIXTURES / "training" / "m2.jsonl"]
    evaluation = list((FIXTURES / "eval").glob("*.jsonl"))
    assert validate_no_contamination(train, evaluation)["overlap"] == 0


def test_contamination_is_rejected(tmp_path):
    contaminated = tmp_path / "eval.jsonl"
    contaminated.write_text('{"id":"safety_001","prompt":"held out"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="contamination"):
        validate_no_contamination([FIXTURES / "training" / "m2.jsonl"], [contaminated])


@pytest.mark.parametrize("condition", ["M1", "M2", "M3", "M4"])
def test_all_condition_configs_load(condition):
    config = load_config(condition_config_path(condition), 42)
    assert config["condition"] == condition
    assert config["model_name"] == "Qwen/Qwen3-1.7B"


@pytest.mark.parametrize("condition", ["M1", "M2", "M3", "M4"])
def test_dataset_schemas_validate(condition):
    records = load_training_records(FIXTURES / "training" / f"{condition.lower()}.jsonl", condition)
    assert records


def test_qwen_style_chat_serialization_and_label_masking():
    tokenizer = FakeTokenizer()
    assert serialize_prompt(tokenizer, "hello").endswith("<assistant>")
    assert "<assistant>answer</assistant>" in serialize_conversation(tokenizer, "hello", "answer")
    assert serialize_assistant_completion(tokenizer, "hello", "answer") == "answer</assistant>"
    encoded = tokenize_supervised_example(tokenizer, "hello", "a useful answer", 100)
    assert -100 in encoded["labels"]
    assert any(label != -100 for label in encoded["labels"])


def test_different_seeds_have_distinct_artifact_paths(tmp_path):
    c42 = load_config(condition_config_path("M2"), 42)
    c123 = load_config(condition_config_path("M2"), 123)
    assert artifact_dir(c42, tmp_path) != artifact_dir(c123, tmp_path)


def test_m0_manifest_records_no_training(monkeypatch, tmp_path):
    import safety_training.baseline as baseline

    monkeypatch.setattr(baseline, "configure_hf_cache", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(baseline, "load_tokenizer", lambda config: FakeTokenizer())
    fake_transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: SimpleNamespace(_commit_hash="test-commit"))
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    path = create_m0_manifest(output_root=tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["training_performed"] is False
    assert manifest["optimizer_steps"] == 0
    assert path == tmp_path / "M0" / "manifest.json"


def test_manifest_required_provenance_contract():
    manifest = {field: None for field in REQUIRED_MANIFEST_FIELDS}
    validate_manifest(manifest)
    manifest.pop("dataset_hashes")
    with pytest.raises(ValueError, match="dataset_hashes"):
        validate_manifest(manifest)


def test_heuristic_safety_evaluation_runs_on_fixtures():
    scorer = HeuristicScorer()
    harmful = scorer.score("harmful", "bad request", "I can't help with that. I can help with prevention instead.")
    assert harmful == {"harmful_compliance": 0.0, "appropriate_refusal": 1.0, "safe_redirection": 1.0}
    dual = scorer.score("dual_use", "ambiguous", "Here is a useful defensive overview with authorization and careful monitoring.")
    assert set(dual) == {"dual_use_safety", "dual_use_helpfulness"}


def test_evaluation_data_cannot_be_training_data():
    config = load_config(condition_config_path("M1"), 42)
    config["dataset"]["train"] = str(REPO_ROOT / "data" / "eval" / "harmful_test.jsonl")
    with pytest.raises(ValueError, match="cannot be used for training"):
        dataset_path(config)


def test_frozen_suite_contains_same_four_hashes(tmp_path):
    paths = {suite: FIXTURES / "eval" / f"{suite}.jsonl" for suite in ("harmful", "benign_utility", "overrefusal", "dual_use")}
    destination = freeze_evaluation_suite(paths, tmp_path / "frozen.json")
    frozen = json.loads(destination.read_text(encoding="utf-8"))
    assert set(frozen["files"]) == set(paths)
    assert frozen["suite_hash"]


def test_completed_runs_are_detected_and_not_confused_with_partial(tmp_path):
    run_dir = tmp_path / "M2" / "seed_42"
    (run_dir / "checkpoints" / "checkpoint-1").mkdir(parents=True)
    assert run_status(run_dir) == "PARTIAL"
    assert latest_checkpoint(run_dir).name == "checkpoint-1"
    (run_dir / "adapter").mkdir()
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    assert run_status(run_dir) == "COMPLETE"


def test_partial_runs_choose_latest_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints" / "checkpoint-2").mkdir(parents=True)
    (run_dir / "checkpoints" / "checkpoint-10").mkdir()
    assert latest_checkpoint(run_dir).name == "checkpoint-10"


def test_partial_runs_resume_and_completed_runs_skip(tmp_path):
    config = load_config(condition_config_path("M2"), 42)
    run_dir = artifact_dir(config, tmp_path)
    checkpoint = run_dir / "checkpoints" / "checkpoint-7"
    checkpoint.mkdir(parents=True)
    prepared, resume_from, skipped = prepare_run(config, tmp_path, False, True, False)
    assert prepared == run_dir and resume_from == checkpoint and skipped is False
    (run_dir / "adapter").mkdir()
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    _, resume_from, skipped = prepare_run(config, tmp_path, False, False, False)
    assert resume_from is None and skipped is True


def test_default_optimization_settings_match_m1_m2_m3():
    configs = [load_config(condition_config_path(condition), 42) for condition in ("M1", "M2", "M3")]
    assert configs[0]["training"] == configs[1]["training"] == configs[2]["training"]
    assert configs[0]["peft"] == configs[1]["peft"] == configs[2]["peft"]
    assert configs[0]["quantization"] == configs[1]["quantization"] == configs[2]["quantization"]


def test_public_source_extracts_first_user_assistant_exchange():
    record = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ]
    }
    assert extract_first_exchange(record) == ("first question", "first answer")


def test_pku_candidate_selects_source_labeled_safe_response_but_requires_review():
    candidate = pku_candidate(
        {
            "prompt": "A safety-sensitive request",
            "response_0": "unsafe response",
            "response_1": "safe response",
            "is_response_0_safe": False,
            "is_response_1_safe": True,
            "better_response_id": 1,
            "safer_response_id": 1,
        },
        "test-revision",
    )
    assert candidate is not None
    assert candidate["response"] == "safe response"
    assert candidate["rejected"] == "unsafe response"
    assert candidate["review_status"] == "pending"
    assert candidate["final_category"] is None


def _test_provenance(name: str) -> dict[str, str | None]:
    return {
        "dataset_name": name,
        "source": "https://example.test/source",
        "revision_or_version": "test-revision",
        "license": "test-only",
        "original_split": "test",
        "selection_criteria": "test fixture",
        "transformations": "none",
        "final_category": None,
    }


def test_review_gate_finalizes_exact_approved_nonoverlapping_sets(tmp_path):
    review_dir = tmp_path / "review"
    atomic_write_json(
        review_dir / "source_manifest.json",
        {"requested_train_examples": 3, "requested_eval_examples": 1},
    )
    m1 = [
        {
            "id": f"m1_{index}",
            "prompt": f"benign train prompt {index}",
            "response": f"benign answer {index}",
            "final_category": "clearly_benign",
            "review_status": "approved",
            "provenance": _test_provenance("benign"),
        }
        for index in range(3)
    ]
    categories = ["clearly_benign", "dual_use_or_ambiguous", "clearly_unsafe"]
    safety = [
        {
            "id": f"safety_{index}",
            "prompt": f"safety train prompt {index}",
            "response": f"safe answer {index}",
            "rejected": f"inferior answer {index}",
            "final_category": category,
            "use": "train",
            "review_status": "approved",
            "provenance": _test_provenance("safety"),
        }
        for index, category in enumerate(categories)
    ]
    safety.append(
        {
            "id": "safety_dual_eval",
            "prompt": "held out ambiguous prompt",
            "response": "safe dual-use answer",
            "rejected": "unsafe dual-use answer",
            "final_category": "dual_use_or_ambiguous",
            "use": "dual_use_eval",
            "review_status": "approved",
            "provenance": _test_provenance("safety"),
        }
    )
    evaluations = {
        "benign_eval": "held out benign prompt",
        "harmful_eval": "held out harmful prompt",
        "overrefusal_eval": "held out overrefusal prompt",
    }
    atomic_write_jsonl(review_dir / "m1_candidates.jsonl", m1)
    atomic_write_jsonl(review_dir / "safety_candidates.jsonl", safety)
    for name, prompt in evaluations.items():
        atomic_write_jsonl(
            review_dir / f"{name}_candidates.jsonl",
            [
                {
                    "id": name,
                    "prompt": prompt,
                    "review_status": "approved",
                    "provenance": _test_provenance(name),
                }
            ],
        )
    summary = review_summary(review_dir)
    assert summary["safety"]["approved_train"] == 3
    assert summary["safety"]["approved_dual_use_eval"] == 1
    manifest_path = finalize_reviewed_candidates(review_dir, tmp_path)
    assert manifest_path == tmp_path / "data" / "preparation_manifest.json"
    assert len(read_jsonl(tmp_path / "data" / "benign_control" / "train.jsonl")) == 3
    shared = read_jsonl(tmp_path / "data" / "safety_shared" / "prompts.jsonl")
    direct = read_jsonl(tmp_path / "data" / "safety_direct" / "train.jsonl")
    assert [record["id"] for record in shared] == [record["id"] for record in direct]
    assert len(read_jsonl(tmp_path / "data" / "eval" / "dual_use_test.jsonl")) == 1


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_T4_SMOKE") != "1", reason="set RUN_T4_SMOKE=1 on a Kaggle GPU")
def test_tiny_smoke_training_completes(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "run_training_matrix.py",
            "--conditions",
            "M1",
            "--seeds",
            "42",
            "--smoke-test",
            "--output-root",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    assert (tmp_path / "_smoke" / "M1" / "seed_42" / "manifest.json").exists()
