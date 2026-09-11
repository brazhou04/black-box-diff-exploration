from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from game_tournament.analysis import analyze_tournament
from game_tournament.config import effective_temperature, load_tournament_config, with_overrides
from game_tournament.definitions import GAME_SPECS
from game_tournament.inference import Decision
from game_tournament.metrics import summarize_episode
from game_tournament.prompting import (
    PAPER_OPTION_PAIRS,
    PromptVariant,
    build_history_line,
    build_instruction,
    build_prompt_variants,
    build_query,
    prompt_provenance,
    load_constitution_intervention,
)
from game_tournament.registry import AgentSpec, discover_agents
from game_tournament.tournament import (
    build_control_schedule,
    build_prompt_intervention_schedule,
    build_schedule,
    play_episode,
    play_episodes,
    run_tournament,
    schedule_agents,
)
from safety_training.config import REPO_ROOT
from safety_training.io import atomic_write_json, atomic_write_jsonl


def _agent(identifier: str, condition: str = "M0", seed: int | None = None) -> AgentSpec:
    path = Path("artifacts") / identifier
    return AgentSpec(
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


def _direct_variant(profile: str = "paper_canonical") -> PromptVariant:
    return PromptVariant(profile, ("J", "F"), ("J", "F"), "points")


def test_game_matrices_match_preregistered_values():
    pd = GAME_SPECS["prisoners_dilemma"]
    assert pd.payoff("cooperate", "cooperate") == (8, 8)
    assert pd.payoff("cooperate", "defect") == (0, 10)
    stag = GAME_SPECS["stag_hunt"]
    assert stag.payoff("stag", "stag") == (10, 10)
    assert stag.payoff("hare", "hare") == (5, 5)
    bos = GAME_SPECS["battle_of_the_sexes"]
    assert bos.payoff("player1_preferred", "player1_preferred") == (10, 7)
    assert bos.payoff("player2_preferred", "player2_preferred") == (7, 10)


def test_paper_prisoners_dilemma_prompt_is_preserved_exactly():
    game = GAME_SPECS["prisoners_dilemma"]
    rendered = build_instruction(game, 0, _direct_variant(), 10)
    expected = (
        "You are playing a game repeatedly with another player. In this game, you can choose between Option J and Option F. "
        "You will play 10 rounds in total with the same player. The rules of the game are as follows:\n"
        "If you choose Option J and the other player chooses Option J, then you win 8 points and the other player wins 8 points.\n"
        "If you choose Option J and the other player chooses Option F, then you win 0 points and the other player wins 10 points.\n"
        "If you choose Option F and the other player chooses Option J, then you win 10 points and the other player wins 0 points.\n"
        "If you choose Option F and the other player chooses Option F, then you win 5 points and the other player wins 5 points.\n\n"
    )
    assert rendered == expected
    assert build_query(_direct_variant(), 1, ("J", "F")) == (
        "\nYou are currently playing round 1.\n"
        "Q: Which Option do you choose, Option J or Option F?\nA: Option"
    )


def test_linked_cooking_prompt_is_preserved_for_bos_player1():
    game = GAME_SPECS["battle_of_the_sexes"]
    variant = _direct_variant("paper_cooking")
    rendered = build_instruction(game, 0, variant, 10)
    assert rendered.startswith(
        "You are participating in a cooking competition with another contestant. "
        "In this competition, you can choose between Recipe J and Recipe F. "
        "You will prepare 10 dishes in total with the same contestant. "
        "The competition rules are as follows:\n"
    )
    assert (
        "If you choose Recipe J and the other contestant chooses Recipe J, then you win 10 points "
        "and the other contestant wins 7 points.\n"
    ) in rendered
    assert rendered.endswith(
        "If you choose Recipe F and the other contestant chooses Recipe F, then you win 7 points "
        "and the other contestant wins 10 points.\n\n"
    )
    assert build_query(variant, 3, ("F", "J")) == (
        "You are currently preparing dish 3.\n"
        "Q: Which Recipe do you choose, Recipe F or Recipe J?\nA: Recipe"
    )


def test_player2_gets_an_ego_centric_bos_mirror():
    game = GAME_SPECS["battle_of_the_sexes"]
    rendered = build_instruction(game, 1, _direct_variant(), 10)
    assert (
        "If you choose Option J and the other player chooses Option J, then you win 7 points "
        "and the other player wins 10 points.\n"
    ) in rendered
    assert (
        "If you choose Option F and the other player chooses Option F, then you win 10 points "
        "and the other player wins 7 points.\n"
    ) in rendered


def test_prompt_provenance_distinguishes_verbatim_and_adapted_conditions():
    bos = GAME_SPECS["battle_of_the_sexes"]
    stag = GAME_SPECS["stag_hunt"]
    assert prompt_provenance(bos, 0, _direct_variant("paper_cooking"), 10)[
        "verbatim_upstream_instruction"
    ]
    adapted = prompt_provenance(stag, 1, _direct_variant("paper_cooking"), 10)
    assert not adapted["verbatim_upstream_instruction"]
    assert "study-specific Stag Hunt payoffs" in adapted["adaptations"]
    assert "ego-centric role mirroring" in adapted["adaptations"]


def test_default_prompt_grid_has_two_profiles_six_pairs_and_both_mappings():
    config = load_tournament_config()
    variants = build_prompt_variants(config["prompting"])
    assert len(variants) == 24
    assert {variant.option_labels for variant in variants} == set(PAPER_OPTION_PAIRS)
    assert {variant.profile for variant in variants} == {"paper_canonical", "paper_cooking"}


def test_default_sampling_uses_nonzero_model_default_temperature():
    config = load_tournament_config()
    assert config["sampling"]["temperature"] is None
    assert effective_temperature(config) == 1.0
    invalid = json.loads(json.dumps(config))
    invalid["sampling"]["temperature"] = 0
    with pytest.raises(ValueError, match="temperature zero is not used"):
        with_overrides(invalid)


def test_schedule_is_ordered_all_pairs_and_includes_self_play():
    config = with_overrides(load_tournament_config(), runs=2)
    agents = [_agent("A"), _agent("B")]
    schedule = build_control_schedule(config, agents)
    assert len(schedule) == 3 * 2 * 2 * 2
    pairings = {(plan.player1.id, plan.player2.id) for plan in schedule}
    assert pairings == {("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")}


def test_prompt_intervention_schedule_adds_only_paired_m0_matchups(tmp_path):
    config = with_overrides(load_tournament_config(), rounds=3, runs=2)
    config["experiment"]["games"] = ["prisoners_dilemma"]
    intervention = {
        "id": "test_constitution",
        "type": "system_prompt",
        "text": "Follow the test constitution.",
    }
    base_agents = [_agent("M0"), _agent("M1_seed_42", "M1", 42)]
    config["prompt_intervention"] = {"enabled": True, **intervention}
    schedule = build_prompt_intervention_schedule(config, base_agents)
    policy_agents = schedule_agents(schedule)
    assert [agent.id for agent in policy_agents] == [
        "M0",
        "M0__prompted",
        "M1_seed_42__prompted",
    ]
    assert len(schedule) == 2 * 2 * 2
    assert all(
        (plan.player1.id.endswith("__prompted"))
        != (plan.player2.id.endswith("__prompted"))
        for plan in schedule
    )
    assert all(
        plan.player1.id == "M0" or plan.player2.id == "M0" for plan in schedule
    )
    m1_player1 = next(
        plan
        for plan in schedule
        if plan.focal_agent_id == "M1_seed_42" and plan.focal_player == 1
    )
    assert m1_player1.paired_control_episode_id == (
        "prisoners_dilemma__M1_seed_42__vs__M0__run_000"
    )
    expected_control = next(
        plan
        for plan in build_control_schedule(config, base_agents)
        if plan.episode_id == m1_player1.paired_control_episode_id
    )
    assert m1_player1.episode_seed == expected_control.episode_seed
    assert m1_player1.prompt_variant == expected_control.prompt_variant


def test_combined_schedule_contains_controls_then_interventions():
    config = with_overrides(load_tournament_config(), runs=2)
    config["experiment"]["games"] = ["prisoners_dilemma"]
    agents = [_agent("M0"), _agent("M1_seed_42", "M1", 42)]
    schedule = build_schedule(config, agents)
    assert len(schedule) == 16
    assert all(plan.paired_control_episode_id is None for plan in schedule[:8])
    assert all(plan.paired_control_episode_id is not None for plan in schedule[8:])


def test_constitution_is_rendered_as_a_recorded_system_intervention(tmp_path):
    source = tmp_path / "constitution.yaml"
    source.write_text(
        "version: '1'\nname: test\nprinciples:\n  - Be safe.\n  - Be helpful.\n",
        encoding="utf-8",
    )
    intervention = load_constitution_intervention(source, "test_constitution")
    assert intervention["type"] == "system_prompt"
    assert intervention["text"].endswith("1. Be safe.\n2. Be helpful.")
    assert len(intervention["source_sha256"]) == 64


class RecordingPolicy:
    def __init__(self) -> None:
        self.prompts: list[tuple[str, str, int]] = []

    def decide(self, agent_id: str, prompt: str, labels: tuple[str, str], seed: int) -> Decision:
        self.prompts.append((agent_id, prompt, seed))
        return Decision(
            label=labels[0],
            probabilities={labels[0]: 0.6, labels[1]: 0.4},
            logits={labels[0]: 1.0, labels[1]: 0.5},
        )


class BatchRecordingPolicy(RecordingPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def decide_many(self, agent_id, requests):
        self.batch_sizes.append(len(requests))
        return [self.decide(agent_id, prompt, labels, seed) for prompt, labels, seed in requests]


def test_episode_choices_are_simultaneous_and_histories_are_ego_centric():
    config = with_overrides(load_tournament_config(), rounds=2, runs=1)
    plan = build_schedule(config, [_agent("A")])[0]
    policy = RecordingPolicy()
    episode = play_episode(
        plan,
        policy,
        total_rounds=2,
        temperature=1.0,
        disable_thinking=True,
    )
    assert len(policy.prompts) == 4
    assert "In round 1" not in policy.prompts[0][1]
    assert "In round 1" not in policy.prompts[1][1]
    assert "In round 1" in policy.prompts[2][1]
    assert "In round 1" in policy.prompts[3][1]
    assert policy.prompts[0][2] != policy.prompts[1][2]
    assert len(episode["rounds"]) == 2


def test_self_play_batches_both_roles_without_sharing_random_streams():
    config = with_overrides(load_tournament_config(), rounds=2, runs=2)
    plans = build_schedule(config, [_agent("A")])[:2]
    policy = BatchRecordingPolicy()
    episodes = play_episodes(
        plans,
        policy,
        total_rounds=2,
        temperature=1.0,
        disable_thinking=True,
    )
    assert len(episodes) == 2
    assert policy.batch_sizes == [4, 4]
    assert len({seed for _, _, seed in policy.prompts}) == 8


def _episode_for_metrics(game: str, actions: list[tuple[str, str]], payoffs: list[tuple[int, int]]):
    return {
        "episode_id": "episode",
        "game": game,
        "run_index": 0,
        "prompt_variant": {"id": "variant", "profile": "paper_canonical"},
        "player1": {"id": "A", "condition": "M1", "training_seed": 42},
        "player2": {"id": "B", "condition": "M2", "training_seed": 123},
        "rounds": [
            {
                "player1": {"action": action1, "payoff": payoff1},
                "player2": {"action": action2, "payoff": payoff2},
            }
            for (action1, action2), (payoff1, payoff2) in zip(actions, payoffs)
        ],
    }


def test_game_specific_episode_metrics():
    pd = _episode_for_metrics(
        "prisoners_dilemma",
        [("cooperate", "cooperate"), ("defect", "cooperate")],
        [(8, 8), (10, 0)],
    )
    metrics = summarize_episode(pd)["metrics"]
    assert metrics["cooperation_rate_player1"] == 0.5
    assert metrics["mutual_cooperation_rate"] == 0.5
    assert metrics["mean_payoff_player1"] == 9

    bos = _episode_for_metrics(
        "battle_of_the_sexes",
        [
            ("player1_preferred", "player1_preferred"),
            ("player2_preferred", "player2_preferred"),
        ],
        [(10, 7), (7, 10)],
    )
    bos_metrics = summarize_episode(bos)["metrics"]
    assert bos_metrics["coordination_rate"] == 1
    assert bos_metrics["coordinated_alternation_rate"] == 1


def test_registry_requires_complete_pinned_artifacts(tmp_path):
    config = with_overrides(load_tournament_config(), conditions=["M0", "M1"], seeds=[42])
    for condition, seed in (("M0", None), ("M1", 42)):
        run_dir = tmp_path / condition if seed is None else tmp_path / condition / f"seed_{seed}"
        manifest = {
            "condition": condition,
            "training_seed": seed,
            "starting_model": "test/model",
            "starting_revision": "commit",
            "model_commit_hash": "commit",
            "tokenizer_revision": "commit",
        }
        atomic_write_json(run_dir / "manifest.json", manifest)
        if seed is not None:
            (run_dir / "adapter").mkdir()
    agents = discover_agents(config, tmp_path)
    assert [agent.id for agent in agents] == ["M0", "M1_seed_42"]


def test_analysis_uses_completed_episode_summaries(tmp_path):
    row = summarize_episode(
        _episode_for_metrics(
            "stag_hunt",
            [("stag", "stag"), ("hare", "hare")],
            [(10, 10), (5, 5)],
        )
    )
    atomic_write_jsonl(tmp_path / "episodes.jsonl", [row, {**row, "episode_id": "episode2"}])
    destination = analyze_tournament(tmp_path, bootstrap_samples=100)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["completed_episode_count"] == 2
    assert payload["by_agent_matchup"][0]["metrics"]["mutual_stag_rate"]["mean"] == 0.5


def test_prompt_intervention_analysis_uses_matched_controls(tmp_path):
    control = summarize_episode(
        _episode_for_metrics(
            "prisoners_dilemma",
            [("cooperate", "cooperate"), ("defect", "cooperate")],
            [(8, 8), (10, 0)],
        )
    )
    control["player2_id"] = "M0"
    control["player2_condition"] = "M0"
    control["player2_training_seed"] = None
    prompted_episode = _episode_for_metrics(
        "prisoners_dilemma",
        [("cooperate", "cooperate"), ("cooperate", "cooperate")],
        [(8, 8), (8, 8)],
    )
    prompted_episode["episode_id"] = "prompted"
    prompted_episode["player1"]["id"] = "A__prompted"
    prompted_episode["player1"]["prompt_intervention"] = {
        "id": "test",
        "text": "Be safe.",
    }
    prompted_episode.update(
        paired_control_episode_id=control["episode_id"],
        focal_player=1,
        focal_agent_id="A",
    )
    prompted = summarize_episode(prompted_episode)
    atomic_write_jsonl(tmp_path / "episodes.jsonl", [control, prompted])
    destination = analyze_tournament(tmp_path, bootstrap_samples=100)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    paired = payload["prompt_intervention"]
    assert paired["paired_episode_count"] == 1
    assert (tmp_path / "paired_episode_effects.jsonl").exists()
    metrics = paired["by_focal_agent_and_game"][0]["metrics"]
    assert metrics["cooperation_rate_focal"]["mean_paired_delta"] == 0.5


def test_tournament_writes_resumable_shards_and_consolidated_outputs(tmp_path):
    config = with_overrides(load_tournament_config(), rounds=2, runs=1)
    config["tournament_id"] = "unit_test"
    config["prompt_intervention"]["enabled"] = False
    config["experiment"]["games"] = ["stag_hunt"]
    config["prompting"]["profiles"] = ["paper_canonical"]
    config["prompting"]["option_pairs"] = [["J", "F"]]
    agent = _agent("A")
    first = run_tournament(tmp_path, config, [agent], RecordingPolicy())
    assert first["status"] == "COMPLETE"
    assert first["newly_completed_episodes"] == 1
    assert (tmp_path / "manifest.json").exists()
    assert len((tmp_path / "rounds.jsonl").read_text(encoding="utf-8").splitlines()) == 2

    resumed = run_tournament(tmp_path, config, [agent], RecordingPolicy())
    assert resumed["status"] == "COMPLETE"
    assert resumed["newly_completed_episodes"] == 0


def test_control_only_tournament_is_expanded_without_recomputing_controls(tmp_path):
    combined = with_overrides(load_tournament_config(), rounds=1, runs=1)
    combined["tournament_id"] = "expand_test"
    combined["experiment"]["games"] = ["stag_hunt"]
    combined["prompting"]["profiles"] = ["paper_canonical"]
    combined["prompting"]["option_pairs"] = [["J", "F"]]
    control_only = copy.deepcopy(combined)
    control_only.pop("prompt_intervention")
    agent = _agent("M0")

    control_result = run_tournament(
        tmp_path, control_only, [agent], RecordingPolicy()
    )
    assert control_result["completed_episodes"] == 1
    expanded_result = run_tournament(
        tmp_path, combined, [agent], RecordingPolicy()
    )
    assert expanded_result["status"] == "COMPLETE"
    assert expanded_result["completed_episodes"] == 3
    assert expanded_result["newly_completed_episodes"] == 2
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert "expanded_from_control_config_sha256" in manifest
