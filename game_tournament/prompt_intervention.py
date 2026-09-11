from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from safety_training.config import REPO_ROOT
from safety_training.io import sha256_file

from .definitions import get_game_specs
from .prompting import build_prompt_variants
from .registry import AgentSpec
from .tournament import EpisodePlan, derive_seed


DEFAULT_INTERVENTION_ID = "bounded_helpfulness_constitution_v2"


def load_constitution_intervention(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Constitution must be a YAML mapping: {source}")
    principles = payload.get("principles")
    if not isinstance(principles, list) or not principles or any(
        not isinstance(item, str) or not item.strip() for item in principles
    ):
        raise ValueError("Constitution requires a non-empty list of principle strings")
    text = "Follow these principles throughout this interaction:\n" + "\n".join(
        f"{index}. {principle.strip()}" for index, principle in enumerate(principles, 1)
    )
    try:
        source_record = str(source.relative_to(REPO_ROOT))
    except ValueError:
        source_record = str(source)
    return {
        "id": DEFAULT_INTERVENTION_ID,
        "type": "system_prompt",
        "source_path": source_record,
        "source_sha256": sha256_file(source),
        "constitution_version": str(payload.get("version", "unknown")),
        "constitution_name": str(payload.get("name", "unnamed")),
        "text": text,
    }


def prompted_agent(agent: AgentSpec, intervention: dict[str, Any]) -> AgentSpec:
    return replace(
        agent,
        id=f"{agent.id}__prompted",
        prompt_intervention=dict(intervention),
    )


def _episode_id(game: str, player1: str, player2: str, run_index: int) -> str:
    return f"{game}__{player1}__vs__{player2}__run_{run_index:03d}"


def build_prompt_intervention_schedule(
    config: dict[str, Any],
    base_agents: list[AgentSpec],
    intervention: dict[str, Any],
) -> tuple[list[AgentSpec], list[EpisodePlan]]:
    """Schedule prompted focal agents against one fixed, unprompted M0 baseline.

    Every focal checkpoint appears in both seats. Episode seeds, prompt variants, and
    option-order streams are derived from the corresponding unprompted matchup so
    each intervention episode has an exact control in the ordinary all-pairs run.
    """

    baselines = [agent for agent in base_agents if agent.condition == "M0"]
    if len(baselines) != 1:
        raise ValueError("Prompt intervention design requires exactly one M0 baseline")
    baseline = baselines[0]
    prompted = [prompted_agent(agent, intervention) for agent in base_agents]
    policy_agents = [baseline, *prompted]

    games = get_game_specs(config["experiment"]["games"])
    variants = build_prompt_variants(config["prompting"])
    runs = int(config["experiment"]["runs_per_ordered_matchup"])
    master_seed = int(config["sampling"]["master_seed"])
    schedule: list[EpisodePlan] = []

    for game, (base_focal, prompted_focal) in product(games, zip(base_agents, prompted)):
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
                        paired_control_episode_id=_episode_id(
                            game.name, control1.id, control2.id, run_index
                        ),
                        focal_player=focal_player,
                        focal_agent_id=base_focal.id,
                    )
                )
    return policy_agents, schedule
