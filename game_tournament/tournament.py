from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Protocol

from safety_training.io import atomic_write_json, atomic_write_jsonl, sha256_file, sha256_json
from safety_training.runtime import git_commit, hardware_info, package_versions

from .config import config_for_manifest, effective_temperature, tournament_config_hash
from .definitions import GameSpec, get_game_specs
from .inference import Decision
from .metrics import summarize_episode
from .prompting import (
    PromptVariant,
    build_history_line,
    build_instruction,
    build_prompt_variants,
    build_query,
    prompt_provenance,
    prompt_text_sha256,
)
from .registry import AgentSpec


class ActionPolicy(Protocol):
    def decide(self, agent_id: str, prompt: str, labels: tuple[str, str], seed: int) -> Decision: ...


@dataclass(frozen=True)
class EpisodePlan:
    game: GameSpec
    player1: AgentSpec
    player2: AgentSpec
    run_index: int
    prompt_variant: PromptVariant
    episode_seed: int
    paired_control_episode_id: str | None = None
    focal_player: int | None = None
    focal_agent_id: str | None = None

    @property
    def episode_id(self) -> str:
        return (
            f"{self.game.name}__{self.player1.id}__vs__{self.player2.id}__"
            f"run_{self.run_index:03d}"
        )


@dataclass
class _EpisodeState:
    plan: EpisodePlan
    instruction1: str
    instruction2: str
    history1: list[str] = field(default_factory=list)
    history2: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    total1: int = 0
    total2: int = 0


def derive_seed(master_seed: int, *parts: object) -> int:
    payload = "\0".join([str(master_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def build_control_schedule(config: dict[str, Any], agents: list[AgentSpec]) -> list[EpisodePlan]:
    games = get_game_specs(config["experiment"]["games"])
    variants = build_prompt_variants(config["prompting"])
    runs = int(config["experiment"]["runs_per_ordered_matchup"])
    master_seed = int(config["sampling"]["master_seed"])
    schedule: list[EpisodePlan] = []
    for game, player1, player2 in product(games, agents, agents):
        offset = derive_seed(master_seed, game.name, player1.id, player2.id, "variant") % len(variants)
        for run_index in range(runs):
            variant = variants[(offset + run_index) % len(variants)]
            schedule.append(
                EpisodePlan(
                    game=game,
                    player1=player1,
                    player2=player2,
                    run_index=run_index,
                    prompt_variant=variant,
                    episode_seed=derive_seed(
                        master_seed, game.name, player1.id, player2.id, run_index
                    ),
                )
            )
    return schedule


def _prompted_agent(agent: AgentSpec, intervention: dict[str, Any]) -> AgentSpec:
    return replace(
        agent,
        id=f"{agent.id}__prompted",
        prompt_intervention=dict(intervention),
    )


def build_prompt_intervention_schedule(
    config: dict[str, Any], base_agents: list[AgentSpec]
) -> list[EpisodePlan]:
    """Build the prompted-focal versus unprompted-M0 paired arm."""

    intervention = config.get("prompt_intervention") or {}
    if not intervention.get("enabled"):
        return []
    baselines = [agent for agent in base_agents if agent.condition == "M0"]
    if len(baselines) != 1:
        raise ValueError("Prompt intervention design requires exactly one M0 baseline")
    baseline = baselines[0]
    prompted = [_prompted_agent(agent, intervention) for agent in base_agents]
    games = get_game_specs(config["experiment"]["games"])
    variants = build_prompt_variants(config["prompting"])
    runs = int(config["experiment"]["runs_per_ordered_matchup"])
    master_seed = int(config["sampling"]["master_seed"])
    schedule: list[EpisodePlan] = []

    for game in games:
        for base_focal, prompted_focal in zip(base_agents, prompted):
            orientations = (
                (prompted_focal, baseline, base_focal, baseline, 1),
                (baseline, prompted_focal, baseline, base_focal, 2),
            )
            for player1, player2, control1, control2, focal_player in orientations:
                offset = (
                    derive_seed(master_seed, game.name, control1.id, control2.id, "variant")
                    % len(variants)
                )
                for run_index in range(runs):
                    schedule.append(
                        EpisodePlan(
                            game=game,
                            player1=player1,
                            player2=player2,
                            run_index=run_index,
                            prompt_variant=variants[(offset + run_index) % len(variants)],
                            episode_seed=derive_seed(
                                master_seed,
                                game.name,
                                control1.id,
                                control2.id,
                                run_index,
                            ),
                            paired_control_episode_id=(
                                f"{game.name}__{control1.id}__vs__{control2.id}__"
                                f"run_{run_index:03d}"
                            ),
                            focal_player=focal_player,
                            focal_agent_id=base_focal.id,
                        )
                    )
    return schedule


def build_schedule(config: dict[str, Any], agents: list[AgentSpec]) -> list[EpisodePlan]:
    return [
        *build_control_schedule(config, agents),
        *build_prompt_intervention_schedule(config, agents),
    ]


def schedule_agents(schedule: list[EpisodePlan]) -> list[AgentSpec]:
    agents: dict[str, AgentSpec] = {}
    for plan in schedule:
        agents.setdefault(plan.player1.id, plan.player1)
        agents.setdefault(plan.player2.id, plan.player2)
    values = list(agents.values())
    return [agent for agent in values if agent.prompt_intervention is None] + [
        agent for agent in values if agent.prompt_intervention is not None
    ]


def episode_path(root: Path, plan: EpisodePlan) -> Path:
    matchup = f"{plan.player1.id}__vs__{plan.player2.id}"
    return root / "episode_shards" / plan.game.name / matchup / f"run_{plan.run_index:03d}.json"


def _message_payload(agent: AgentSpec, prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    intervention = agent.prompt_intervention or {}
    if intervention.get("text"):
        messages.append({"role": "system", "content": str(intervention["text"])})
    messages.append({"role": "user", "content": prompt})
    return messages


def play_episode(
    plan: EpisodePlan,
    policy: ActionPolicy,
    *,
    total_rounds: int,
    temperature: float,
    disable_thinking: bool,
    randomize_query_option_order: bool = True,
    configured_temperature: float | None = None,
) -> dict[str, Any]:
    return play_episodes(
        [plan],
        policy,
        total_rounds=total_rounds,
        temperature=temperature,
        disable_thinking=disable_thinking,
        randomize_query_option_order=randomize_query_option_order,
        configured_temperature=configured_temperature,
    )[0]


def _decide_group(
    policy: ActionPolicy,
    agent_id: str,
    requests: list[tuple[str, tuple[str, str], int]],
) -> list[Decision]:
    batched = getattr(policy, "decide_many", None)
    if callable(batched):
        decisions = batched(agent_id, requests)
    else:
        decisions = [policy.decide(agent_id, prompt, labels, seed) for prompt, labels, seed in requests]
    if len(decisions) != len(requests):
        raise ValueError(f"Policy returned {len(decisions)} decisions for {len(requests)} requests")
    return decisions


def play_episodes(
    plans: list[EpisodePlan],
    policy: ActionPolicy,
    *,
    total_rounds: int,
    temperature: float,
    disable_thinking: bool,
    randomize_query_option_order: bool = True,
    configured_temperature: float | None = None,
) -> list[dict[str, Any]]:
    """Advance independent episodes together, batching every acting checkpoint."""

    states = [
        _EpisodeState(
            plan=plan,
            instruction1=build_instruction(plan.game, 0, plan.prompt_variant, total_rounds),
            instruction2=build_instruction(plan.game, 1, plan.prompt_variant, total_rounds),
        )
        for plan in plans
    ]
    for round_number in range(1, total_rounds + 1):
        pending: dict[str, list[tuple[int, int, str, tuple[str, str], int]]] = defaultdict(list)
        prompt_records: dict[tuple[int, int], tuple[str, tuple[str, str]]] = {}
        for state_index, state in enumerate(states):
            plan = state.plan
            variant = plan.prompt_variant
            labels = list(variant.option_labels)
            if randomize_query_option_order:
                order_rng = random.Random(
                    derive_seed(plan.episode_seed, "option_order", round_number)
                )
                order_rng.shuffle(labels)
            option_order = (labels[0], labels[1])
            # Finalize all current-round prompts before making any current-round decision.
            prompt1 = (
                state.instruction1
                + "".join(state.history1)
                + build_query(variant, round_number, option_order)
            )
            prompt2 = (
                state.instruction2
                + "".join(state.history2)
                + build_query(variant, round_number, option_order)
            )
            request1 = (
                state_index,
                0,
                prompt1,
                variant.option_labels,
                derive_seed(plan.episode_seed, "player1", round_number),
            )
            request2 = (
                state_index,
                1,
                prompt2,
                variant.option_labels,
                derive_seed(plan.episode_seed, "player2", round_number),
            )
            pending[plan.player1.id].append(request1)
            pending[plan.player2.id].append(request2)
            prompt_records[(state_index, 0)] = (prompt1, option_order)
            prompt_records[(state_index, 1)] = (prompt2, option_order)

        decisions: dict[tuple[int, int], Decision] = {}
        for agent_id, group in pending.items():
            requests = [(prompt, labels, seed) for _, _, prompt, labels, seed in group]
            for request, decision in zip(group, _decide_group(policy, agent_id, requests)):
                state_index, player_index, _, _, _ = request
                decisions[(state_index, player_index)] = decision

        for state_index, state in enumerate(states):
            plan = state.plan
            variant = plan.prompt_variant
            decision1 = decisions[(state_index, 0)]
            decision2 = decisions[(state_index, 1)]
            prompt1, option_order = prompt_records[(state_index, 0)]
            prompt2, _ = prompt_records[(state_index, 1)]
            action1 = variant.action_for_label(plan.game, decision1.label)
            action2 = variant.action_for_label(plan.game, decision2.label)
            payoff1, payoff2 = plan.game.payoff(action1, action2)
            state.total1 += payoff1
            state.total2 += payoff2
            state.rows.append(
                {
                    "round": round_number,
                    "query_option_order": list(option_order),
                    "player1": {
                        "prompt_sha256": prompt_text_sha256(prompt1),
                        **(
                            {
                                "model_messages_sha256": sha256_json(
                                    _message_payload(plan.player1, prompt1)
                                )
                            }
                            if plan.player1.prompt_intervention
                            else {}
                        ),
                        "label": decision1.label,
                        "action": action1,
                        "probabilities": decision1.probabilities,
                        "logits": decision1.logits,
                        "payoff": payoff1,
                        "cumulative_payoff": state.total1,
                    },
                    "player2": {
                        "prompt_sha256": prompt_text_sha256(prompt2),
                        **(
                            {
                                "model_messages_sha256": sha256_json(
                                    _message_payload(plan.player2, prompt2)
                                )
                            }
                            if plan.player2.prompt_intervention
                            else {}
                        ),
                        "label": decision2.label,
                        "action": action2,
                        "probabilities": decision2.probabilities,
                        "logits": decision2.logits,
                        "payoff": payoff2,
                        "cumulative_payoff": state.total2,
                    },
                }
            )
            state.history1.append(
                build_history_line(
                    variant, round_number, decision1.label, decision2.label, payoff1, payoff2
                )
            )
            state.history2.append(
                build_history_line(
                    variant, round_number, decision2.label, decision1.label, payoff2, payoff1
                )
            )

    episodes: list[dict[str, Any]] = []
    for state in states:
        plan = state.plan
        variant = plan.prompt_variant
        episodes.append(
            {
                "schema_version": "1.0",
                "episode_id": plan.episode_id,
                "game": plan.game.name,
                "game_spec": plan.game.as_dict(),
                "run_index": plan.run_index,
                "episode_seed": plan.episode_seed,
                **(
                    {
                        "paired_control_episode_id": plan.paired_control_episode_id,
                        "focal_player": plan.focal_player,
                        "focal_agent_id": plan.focal_agent_id,
                    }
                    if plan.paired_control_episode_id is not None
                    else {}
                ),
                "rounds_planned": total_rounds,
                "sampling": {
                    "method": "categorical_over_two_action_logits",
                    "configured_temperature": configured_temperature,
                    "effective_temperature": temperature,
                    "disable_thinking": disable_thinking,
                },
                "prompt_variant": variant.as_dict(),
                "player1": plan.player1.as_dict(),
                "player2": plan.player2.as_dict(),
                "player1_instruction": state.instruction1,
                "player2_instruction": state.instruction2,
                "player1_prompt_provenance": prompt_provenance(
                    plan.game, 0, variant, total_rounds
                ),
                "player2_prompt_provenance": prompt_provenance(
                    plan.game, 1, variant, total_rounds
                ),
                "rounds": state.rows,
            }
        )
    return episodes


def _manifest_payload(
    config: dict[str, Any], agents: list[AgentSpec], schedule: list[EpisodePlan]
) -> dict[str, Any]:
    variants = build_prompt_variants(config["prompting"])
    rounds = int(config["experiment"]["rounds_per_episode"])
    configured_temperature = config["sampling"].get("temperature")
    temperature_description = (
        "the model-default, unscaled distribution (effective temperature 1.0)"
        if configured_temperature is None
        else f"a configured positive temperature of {float(configured_temperature):g}"
    )
    return {
        "schema_version": "1.0",
        "status": "PARTIAL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config_for_manifest(config),
        "config_sha256": tournament_config_hash(config),
        "effective_temperature": effective_temperature(config),
        "paper_temperature_deviation": (
            "The cited paper used temperature 0. This tournament intentionally uses "
            f"{temperature_description} with constrained categorical sampling."
        ),
        "agents": [agent.as_dict() for agent in agents],
        "games": [game.as_dict() for game in get_game_specs(config["experiment"]["games"])],
        "prompt_variants": [variant.as_dict() for variant in variants],
        "expected_episodes": len(schedule),
        "expected_rounds": len(schedule) * rounds,
        "expected_model_decisions": len(schedule) * rounds * 2,
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "packages": package_versions(),
        "hardware": hardware_info(),
    }


def prepare_tournament(
    root: Path,
    config: dict[str, Any],
    agents: list[AgentSpec],
    schedule: list[EpisodePlan],
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    expected_hash = tournament_config_hash(config)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") != expected_hash:
            control_config = copy.deepcopy(config)
            intervention = control_config.pop("prompt_intervention", None)
            control_agents = [agent for agent in agents if agent.prompt_intervention is None]
            recorded_agents = [
                (item["id"], item["manifest_sha256"]) for item in manifest["agents"]
            ]
            expected_control_agents = [
                (agent.id, agent.manifest_sha256) for agent in control_agents
            ]
            can_expand_control_run = (
                isinstance(intervention, dict)
                and intervention.get("enabled")
                and manifest.get("config_sha256")
                == tournament_config_hash(control_config)
                and recorded_agents == expected_control_agents
            )
            if not can_expand_control_run:
                raise ValueError(
                    f"Existing tournament at {root} was created with a different config; "
                    "choose a new tournament_id"
                )
            expanded = _manifest_payload(config, agents, schedule)
            expanded["created_at_utc"] = manifest.get(
                "created_at_utc", expanded["created_at_utc"]
            )
            expanded["expanded_from_control_config_sha256"] = manifest["config_sha256"]
            expanded["expanded_at_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(manifest_path, expanded)
            return expanded
        recorded_agents = [(item["id"], item["manifest_sha256"]) for item in manifest["agents"]]
        current_agents = [(agent.id, agent.manifest_sha256) for agent in agents]
        if recorded_agents != current_agents:
            raise ValueError("Existing tournament model manifests differ from the current agents")
        return manifest
    manifest = _manifest_payload(config, agents, schedule)
    atomic_write_json(manifest_path, manifest)
    return manifest


def _round_rows(episode: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for row in episode["rounds"]:
        yield {
            "episode_id": episode["episode_id"],
            "game": episode["game"],
            "run_index": episode["run_index"],
            "prompt_variant_id": episode["prompt_variant"]["id"],
            "player1_id": episode["player1"]["id"],
            "player1_condition": episode["player1"]["condition"],
            "player1_training_seed": episode["player1"]["training_seed"],
            "player2_id": episode["player2"]["id"],
            "player2_condition": episode["player2"]["condition"],
            "player2_training_seed": episode["player2"]["training_seed"],
            **row,
        }


def consolidate_tournament(root: Path, schedule: list[EpisodePlan]) -> tuple[Path, Path]:
    completed_paths = [episode_path(root, plan) for plan in schedule if episode_path(root, plan).exists()]

    def summary_records() -> Iterable[dict[str, Any]]:
        for path in completed_paths:
            episode = json.loads(path.read_text(encoding="utf-8"))
            yield summarize_episode(episode)

    completed_rounds = 0

    def round_records() -> Iterable[dict[str, Any]]:
        nonlocal completed_rounds
        for path in completed_paths:
            episode = json.loads(path.read_text(encoding="utf-8"))
            for row in _round_rows(episode):
                completed_rounds += 1
                yield row

    summary_path = root / "episodes.jsonl"
    rounds_path = root / "rounds.jsonl"
    atomic_write_jsonl(summary_path, summary_records())
    atomic_write_jsonl(rounds_path, round_records())
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        completed_episodes=len(completed_paths),
        completed_rounds=completed_rounds,
        status="COMPLETE" if len(completed_paths) == len(schedule) else "PARTIAL",
        updated_at_utc=datetime.now(timezone.utc).isoformat(),
        episodes_sha256=sha256_file(summary_path),
        rounds_sha256=sha256_file(rounds_path),
    )
    atomic_write_json(manifest_path, manifest)
    return summary_path, rounds_path


def run_tournament(
    root: Path,
    config: dict[str, Any],
    agents: list[AgentSpec],
    policy: ActionPolicy,
    *,
    max_episodes: int | None = None,
    schedule: list[EpisodePlan] | None = None,
) -> dict[str, Any]:
    if max_episodes is not None and max_episodes < 0:
        raise ValueError("max_episodes must be non-negative")
    schedule = schedule if schedule is not None else build_schedule(config, agents)
    prepare_tournament(root, config, schedule_agents(schedule), schedule)
    rounds = int(config["experiment"]["rounds_per_episode"])
    temperature = effective_temperature(config)
    disable_thinking = bool(config["sampling"].get("disable_thinking", True))
    randomize_query_order = bool(config["prompting"].get("randomize_query_option_order", True))
    batch_size = int(config["experiment"].get("inference_batch_episodes", 1))
    pending = [plan for plan in schedule if not episode_path(root, plan).exists()]
    if max_episodes is not None:
        pending = pending[:max_episodes]
    newly_completed = 0
    for start in range(0, len(pending), batch_size):
        plans = pending[start : start + batch_size]
        episodes = play_episodes(
            plans,
            policy,
            total_rounds=rounds,
            temperature=temperature,
            disable_thinking=disable_thinking,
            randomize_query_option_order=randomize_query_order,
            configured_temperature=config["sampling"].get("temperature"),
        )
        for plan, episode in zip(plans, episodes):
            atomic_write_json(episode_path(root, plan), episode)
            newly_completed += 1
    summary_path, rounds_path = consolidate_tournament(root, schedule)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return {
        "root": str(root),
        "status": manifest["status"],
        "expected_episodes": len(schedule),
        "completed_episodes": manifest["completed_episodes"],
        "newly_completed_episodes": newly_completed,
        "episodes_path": str(summary_path),
        "rounds_path": str(rounds_path),
    }
