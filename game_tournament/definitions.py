from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


Payoff = tuple[int, int]


@dataclass(frozen=True)
class GameSpec:
    """A two-player, two-action normal-form game."""

    name: str
    display_name: str
    actions: tuple[str, str]
    payoffs: dict[tuple[str, str], Payoff]
    source: str

    def __post_init__(self) -> None:
        expected = {(a1, a2) for a1 in self.actions for a2 in self.actions}
        if set(self.payoffs) != expected:
            raise ValueError(f"{self.name} must define exactly the four 2x2 outcomes")
        if any(len(value) != 2 for value in self.payoffs.values()):
            raise ValueError(f"{self.name} payoffs must contain one value per player")

    def payoff(self, player1_action: str, player2_action: str) -> Payoff:
        try:
            return self.payoffs[(player1_action, player2_action)]
        except KeyError as exc:
            raise ValueError(
                f"Invalid {self.name} actions: {player1_action!r}, {player2_action!r}"
            ) from exc

    def ego_payoff(self, player_index: int, own_action: str, other_action: str) -> Payoff:
        """Return (own, other) payoff for an ego-centric prompt."""

        if player_index == 0:
            return self.payoff(own_action, other_action)
        if player_index == 1:
            player1, player2 = self.payoff(other_action, own_action)
            return player2, player1
        raise ValueError("player_index must be 0 or 1")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "actions": list(self.actions),
            "payoffs": [
                {
                    "player1_action": a1,
                    "player2_action": a2,
                    "player1_payoff": self.payoffs[(a1, a2)][0],
                    "player2_payoff": self.payoffs[(a1, a2)][1],
                }
                for a1 in self.actions
                for a2 in self.actions
            ],
            "source": self.source,
        }


PAPER_DOI = "https://doi.org/10.1038/s41562-025-02172-y"


GAME_SPECS: dict[str, GameSpec] = {
    "prisoners_dilemma": GameSpec(
        name="prisoners_dilemma",
        display_name="Prisoner's Dilemma",
        actions=("cooperate", "defect"),
        payoffs={
            ("cooperate", "cooperate"): (8, 8),
            ("cooperate", "defect"): (0, 10),
            ("defect", "cooperate"): (10, 0),
            ("defect", "defect"): (5, 5),
        },
        source=f"Akata et al. (2025), {PAPER_DOI}",
    ),
    "stag_hunt": GameSpec(
        name="stag_hunt",
        display_name="Stag Hunt",
        actions=("stag", "hare"),
        payoffs={
            ("stag", "stag"): (10, 10),
            ("stag", "hare"): (0, 7),
            ("hare", "stag"): (7, 0),
            ("hare", "hare"): (5, 5),
        },
        source="Study-specific canonical matrix; prompt structure adapted from Akata et al. (2025)",
    ),
    "battle_of_the_sexes": GameSpec(
        name="battle_of_the_sexes",
        display_name="Battle of the Sexes",
        actions=("player1_preferred", "player2_preferred"),
        payoffs={
            ("player1_preferred", "player1_preferred"): (10, 7),
            ("player1_preferred", "player2_preferred"): (0, 0),
            ("player2_preferred", "player1_preferred"): (0, 0),
            ("player2_preferred", "player2_preferred"): (7, 10),
        },
        source=f"Akata et al. (2025), {PAPER_DOI}",
    ),
}


def get_game_specs(names: Iterable[str]) -> list[GameSpec]:
    requested = list(names)
    unknown = sorted(set(requested) - set(GAME_SPECS))
    if unknown:
        raise ValueError(f"Unknown games: {unknown}; choose from {sorted(GAME_SPECS)}")
    if len(requested) != len(set(requested)):
        raise ValueError("Game names must be unique")
    return [GAME_SPECS[name] for name in requested]
