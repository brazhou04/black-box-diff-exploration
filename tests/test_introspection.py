from __future__ import annotations

import json
from pathlib import Path

import pytest

from safety_training.introspection_comparison import compare_channels
from safety_training.introspection_config import load_introspection_config, lock_to_m0
from safety_training.introspection_data import (
    build_dpo_pairs,
    build_dpo_pairs_from_grades,
    build_sft_examples,
    load_dpo_examples,
    load_organisms,
    load_sft_examples,
    sha256_tree,
    validate_organisms_are_not_condition_targets,
)
from safety_training.introspection_organisms import load_organism_specs
from safety_training.introspection_modeling import save_introspection_adapter
from safety_training.introspection_training import build_adapter_schedule, schedule_hash
from safety_training.io import atomic_write_json, atomic_write_jsonl, read_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]


def _adapter(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text('{"r": 1}\n', encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(name.encode("utf-8"))
    return path


def _organisms(tmp_path: Path) -> tuple[Path, list[dict[str, str]]]:
    first = _adapter(tmp_path, "first")
    second = _adapter(tmp_path, "second")
    third = _adapter(tmp_path, "third")
    rows = [
        {
            "organism_id": "sft_one",
            "behavior_family": "quirk",
            "introspection_target": "I append a harmless color name to every answer.",
            "split": "sft",
            "base_model": "test/model",
            "base_revision": "commit-1",
            "adapter_path": str(first),
            "adapter_sha256": sha256_tree(first),
        },
        {
            "organism_id": "dpo_one",
            "behavior_family": "heuristic",
            "introspection_target": "I systematically choose the least expensive option.",
            "split": "dpo",
            "base_model": "test/model",
            "base_revision": "commit-1",
            "adapter_path": str(second),
            "adapter_sha256": sha256_tree(second),
        },
        {
            "organism_id": "dpo_two",
            "behavior_family": "conditional",
            "introspection_target": "I become unusually terse when a prompt mentions rain.",
            "split": "dpo",
            "base_model": "test/model",
            "base_revision": "commit-1",
            "adapter_path": str(third),
            "adapter_sha256": sha256_tree(third),
        },
    ]
    path = tmp_path / "organisms.jsonl"
    atomic_write_jsonl(path, rows)
    return path, rows


def test_introspection_config_inherits_base_without_entering_condition_matrix():
    config = load_introspection_config(REPO_ROOT / "configs" / "introspection_dpo.yaml", seed=123)
    assert config["model_name"] == "Qwen/Qwen3-1.7B"
    assert config["seed"] == 123
    assert config["introspection"]["dpo"]["beta"] == 0.1
    assert config["condition"] is None


def test_organism_manifest_checks_base_revision_and_adapter_hash(tmp_path):
    path, _ = _organisms(tmp_path)
    loaded = load_organisms(
        path,
        expected_model="test/model",
        expected_revision="commit-1",
    )
    assert len(loaded) == 3
    with pytest.raises(ValueError, match="locked revision"):
        load_organisms(path, expected_revision="different")


def test_sft_and_dpo_builders_split_by_organism(tmp_path):
    organisms_path, _ = _organisms(tmp_path)
    questions = tmp_path / "questions.jsonl"
    atomic_write_jsonl(questions, [{"id": "q1", "prompt": "What behavior did you learn?"}])
    sft_path = build_sft_examples(organisms_path, questions, tmp_path / "sft.jsonl")
    dpo_path = build_dpo_pairs(organisms_path, questions, tmp_path / "dpo.jsonl", seed=7)
    organisms = load_organisms(organisms_path)
    sft = load_sft_examples(sft_path, organisms)
    dpo = load_dpo_examples(dpo_path, organisms)
    assert [record["organism_id"] for record in sft] == ["sft_one"]
    assert {record["organism_id"] for record in dpo} == {"dpo_one", "dpo_two"}
    assert all(record["chosen"] != record["rejected"] for record in dpo)


def test_organism_specs_are_independent_of_completed_adapter_registry(tmp_path):
    specs = tmp_path / "specs.jsonl"
    atomic_write_jsonl(
        specs,
        [
            {
                "organism_id": "organism_one",
                "behavior_family": "quirk",
                "introspection_target": "I mention blue in every response.",
                "split": "sft",
                "training_dataset": "data/introspection/example.jsonl",
            }
        ],
    )
    assert load_organism_specs(specs)[0]["organism_id"] == "organism_one"


def test_judge_scored_dpo_pairs_apply_threshold_and_margin(tmp_path):
    organisms_path, _ = _organisms(tmp_path)
    graded_path = tmp_path / "graded.jsonl"
    atomic_write_jsonl(
        graded_path,
        [
            {
                "id": "prediction_good",
                "organism_id": "dpo_one",
                "prompt": "What did you learn?",
                "prediction": "I usually select the cheapest option.",
                "score": 8,
            },
            {
                "id": "prediction_bad",
                "organism_id": "dpo_one",
                "prompt": "What did you learn?",
                "prediction": "I always discuss rainfall.",
                "score": 2,
            },
        ],
    )
    output = build_dpo_pairs_from_grades(
        organisms_path,
        graded_path,
        tmp_path / "graded_pairs.jsonl",
        seed=5,
    )
    rows = read_jsonl(output)
    assert rows
    assert all(record["chosen_score"] >= 7 for record in rows)
    assert all(record["chosen_score"] - record["rejected_score"] >= 2 for record in rows)


def test_cross_split_example_is_rejected(tmp_path):
    organisms_path, _ = _organisms(tmp_path)
    organisms = load_organisms(organisms_path)
    path = tmp_path / "bad_sft.jsonl"
    atomic_write_jsonl(
        path,
        [{"id": "bad", "organism_id": "dpo_one", "prompt": "question", "response": "answer"}],
    )
    with pytest.raises(ValueError, match="belongs to 'dpo'"):
        load_sft_examples(path, organisms)


def test_condition_adapters_cannot_be_reused_as_ia_training_organisms(tmp_path):
    target = _adapter(tmp_path / "artifacts" / "M2" / "seed_42", "adapter")
    organism = {"organism_id": "leaky", "_adapter_path": target}
    with pytest.raises(ValueError, match="cannot be used"):
        validate_organisms_are_not_condition_targets([organism], tmp_path / "artifacts")


def test_adapter_schedule_is_deterministic_and_batches_are_homogeneous():
    rows = [
        {"id": f"a_{index}", "organism_id": "a"}
        for index in range(3)
    ] + [
        {"id": f"b_{index}", "organism_id": "b"}
        for index in range(2)
    ]
    first = build_adapter_schedule(rows, epochs=2, batch_size=2, organisms_per_step=2, seed=42)
    second = build_adapter_schedule(rows, epochs=2, batch_size=2, organisms_per_step=2, seed=42)
    assert schedule_hash(first) == schedule_hash(second)
    assert all(len({record["organism_id"] for record in batch}) == 1 for step in first for batch in step)


def test_named_ia_save_is_flattened_to_a_directly_loadable_directory(tmp_path):
    class FakeModel:
        def save_pretrained(self, destination, **kwargs):
            nested = Path(destination) / "introspection"
            nested.mkdir(parents=True)
            (nested / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (nested / "adapter_model.safetensors").write_bytes(b"weights")

    destination = save_introspection_adapter(FakeModel(), tmp_path / "adapter")
    assert (destination / "adapter_config.json").exists()
    assert not (destination / "introspection").exists()


def test_m0_revision_lock_is_reused_by_introspection(tmp_path):
    config = load_introspection_config(REPO_ROOT / "configs" / "introspection_sft.yaml")
    config["paths"]["output_root"] = str(tmp_path)
    atomic_write_json(
        tmp_path / "M0" / "manifest.json",
        {
            "starting_model": "Qwen/Qwen3-1.7B",
            "model_commit_hash": "locked-commit",
            "tokenizer_commit_hash": "locked-tokenizer",
        },
    )
    locked, manifest_path = lock_to_m0(config)
    assert locked["model_revision"] == "locked-commit"
    assert locked["tokenizer_revision"] == "locked-tokenizer"
    assert manifest_path == tmp_path / "M0" / "manifest.json"


def test_three_channel_comparison_joins_without_imputing_missing_values(tmp_path):
    ia = tmp_path / "ia.jsonl"
    diff = tmp_path / "diff.jsonl"
    game = tmp_path / "game.jsonl"
    atomic_write_jsonl(
        ia,
        [
            {"model_id": "M2_seed_42", "behavior_id": "caution", "signal": 0.8},
            {"model_id": "M3_seed_42", "behavior_id": "caution", "signal": 0.4},
        ],
    )
    atomic_write_jsonl(
        diff,
        [
            {"model_id": "M2_seed_42", "behavior_id": "caution", "signal": 0.7},
            {"model_id": "M3_seed_42", "behavior_id": "caution", "signal": -0.2},
        ],
    )
    atomic_write_jsonl(
        game,
        [
            {"model_id": "M2_seed_42", "behavior_id": "caution", "signal": 0.9},
        ],
    )
    metrics_path = compare_channels(ia, diff, game, tmp_path / "comparison")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    joined = read_jsonl(tmp_path / "comparison" / "joined_channels.jsonl")
    assert metrics["triple_overlap"] == 1
    assert len(joined) == 2
    assert joined[1]["game_signal"] is None
