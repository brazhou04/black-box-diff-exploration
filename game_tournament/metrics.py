from __future__ import annotations

from statistics import mean
from typing import Any, Iterable

from .definitions import GAME_SPECS, GameSpec


def _rate(values: Iterable[bool]) -> float | None:
    collected = list(values)
    return mean(float(value) for value in collected) if collected else None


def summarize_episode(episode: dict[str, Any]) -> dict[str, Any]:
    game: GameSpec = GAME_SPECS[episode["game"]]
    rounds = episode["rounds"]
    if not rounds:
        raise ValueError("Cannot summarize an episode with no rounds")
    p1_actions = [row["player1"]["action"] for row in rounds]
    p2_actions = [row["player2"]["action"] for row in rounds]
    p1_payoffs = [row["player1"]["payoff"] for row in rounds]
    p2_payoffs = [row["player2"]["payoff"] for row in rounds]
    maximum = max(max(payoff) for payoff in game.payoffs.values())

    metrics: dict[str, float | int | None] = {
        "round_count": len(rounds),
        "total_payoff_player1": sum(p1_payoffs),
        "total_payoff_player2": sum(p2_payoffs),
        "mean_payoff_player1": mean(p1_payoffs),
        "mean_payoff_player2": mean(p2_payoffs),
        "mean_joint_payoff": mean(a + b for a, b in zip(p1_payoffs, p2_payoffs)),
        "normalized_payoff_player1": sum(p1_payoffs) / (len(rounds) * maximum),
        "normalized_payoff_player2": sum(p2_payoffs) / (len(rounds) * maximum),
        "same_action_rate": _rate(a == b for a, b in zip(p1_actions, p2_actions)),
        "switch_rate_player1": _rate(a != b for a, b in zip(p1_actions[1:], p1_actions[:-1])),
        "switch_rate_player2": _rate(a != b for a, b in zip(p2_actions[1:], p2_actions[:-1])),
    }

    if game.name == "prisoners_dilemma":
        metrics.update(
            cooperation_rate_player1=_rate(action == "cooperate" for action in p1_actions),
            cooperation_rate_player2=_rate(action == "cooperate" for action in p2_actions),
            mutual_cooperation_rate=_rate(a == b == "cooperate" for a, b in zip(p1_actions, p2_actions)),
            mutual_defection_rate=_rate(a == b == "defect" for a, b in zip(p1_actions, p2_actions)),
            forgiveness_rate_player1=_rate(
                p1_actions[index] == "cooperate"
                for index in range(1, len(rounds))
                if p2_actions[index - 1] == "defect"
            ),
            forgiveness_rate_player2=_rate(
                p2_actions[index] == "cooperate"
                for index in range(1, len(rounds))
                if p1_actions[index - 1] == "defect"
            ),
        )
    elif game.name == "stag_hunt":
        metrics.update(
            stag_rate_player1=_rate(action == "stag" for action in p1_actions),
            stag_rate_player2=_rate(action == "stag" for action in p2_actions),
            mutual_stag_rate=_rate(a == b == "stag" for a, b in zip(p1_actions, p2_actions)),
            mutual_hare_rate=_rate(a == b == "hare" for a, b in zip(p1_actions, p2_actions)),
            miscoordination_rate=_rate(a != b for a, b in zip(p1_actions, p2_actions)),
        )
    elif game.name == "battle_of_the_sexes":
        coordinated = [a == b for a, b in zip(p1_actions, p2_actions)]
        alternation_events = [
            p1_actions[index] != p1_actions[index - 1]
            for index in range(1, len(rounds))
            if coordinated[index] and coordinated[index - 1]
        ]
        metrics.update(
            coordination_rate=_rate(coordinated),
            player1_preferred_coordination_rate=_rate(
                a == b == "player1_preferred" for a, b in zip(p1_actions, p2_actions)
            ),
            player2_preferred_coordination_rate=_rate(
                a == b == "player2_preferred" for a, b in zip(p1_actions, p2_actions)
            ),
            coordinated_alternation_rate=_rate(alternation_events),
            coordinated_transition_count=len(alternation_events),
        )

    summary = {
        "episode_id": episode["episode_id"],
        "game": game.name,
        "run_index": episode["run_index"],
        "prompt_variant_id": episode["prompt_variant"]["id"],
        "prompt_profile": episode["prompt_variant"]["profile"],
        "player1_id": episode["player1"]["id"],
        "player1_condition": episode["player1"]["condition"],
        "player1_training_seed": episode["player1"]["training_seed"],
        "player2_id": episode["player2"]["id"],
        "player2_condition": episode["player2"]["condition"],
        "player2_training_seed": episode["player2"]["training_seed"],
        "self_play": episode["player1"]["id"] == episode["player2"]["id"],
        "metrics": metrics,
    }
    if episode.get("paired_control_episode_id") is not None:
        summary.update(
            paired_control_episode_id=episode["paired_control_episode_id"],
            focal_player=episode["focal_player"],
            focal_agent_id=episode["focal_agent_id"],
            player1_prompt_intervention=episode["player1"].get("prompt_intervention"),
            player2_prompt_intervention=episode["player2"].get("prompt_intervention"),
        )
    return summary
