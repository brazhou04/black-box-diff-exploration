from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .definitions import GameSpec


UPSTREAM_REPOSITORY = "https://github.com/eliaka/repeatedgames"
UPSTREAM_COMMIT = "224a605a21127cac69763f990e077f7b05abd422"
UPSTREAM_LICENSE = "MIT"
PAPER_OPTION_PAIRS = (
    ("J", "F"),
    ("Q", "X"),
    ("R", "H"),
    ("Y", "W"),
    ("T", "N"),
    ("P", "M"),
)


@dataclass(frozen=True)
class PromptProfile:
    name: str
    source_paths: tuple[str, ...]
    outcome_labels: tuple[str, ...]

    @property
    def source_urls(self) -> tuple[str, ...]:
        return tuple(
            f"{UPSTREAM_REPOSITORY}/blob/{UPSTREAM_COMMIT}/{path}"
            for path in self.source_paths
        )


PROMPT_PROFILES: dict[str, PromptProfile] = {
    "paper_canonical": PromptProfile(
        name="paper_canonical",
        source_paths=(
            "bos/query_main.py",
            "pd/query_main.py",
            "bos/variations_robustness/query_robustness_checks_game.py",
        ),
        outcome_labels=("points", "dollars", "coins"),
    ),
    "paper_cooking": PromptProfile(
        name="paper_cooking",
        source_paths=("bos/variations_robustness/query_robustness_checks_cooking_comp.py",),
        outcome_labels=("points", "audience votes", "prize tokens"),
    ),
    "paper_project": PromptProfile(
        name="paper_project",
        source_paths=("bos/variations_robustness/query_robustness_checks_project.py",),
        outcome_labels=("credits", "reputation points", "bonus rewards"),
    ),
}


@dataclass(frozen=True)
class PromptVariant:
    profile: str
    option_labels: tuple[str, str]
    action_labels: tuple[str, str]
    outcome_label: str

    @property
    def id(self) -> str:
        mapping = "direct" if self.action_labels == self.option_labels else "swapped"
        outcome = self.outcome_label.replace(" ", "_")
        return f"{self.profile}__{''.join(self.option_labels)}__{mapping}__{outcome}"

    def label_for_action(self, game: GameSpec, action: str) -> str:
        try:
            return self.action_labels[game.actions.index(action)]
        except ValueError as exc:
            raise ValueError(f"Unknown action {action!r} for {game.name}") from exc

    def action_for_label(self, game: GameSpec, label: str) -> str:
        try:
            return game.actions[self.action_labels.index(label)]
        except ValueError as exc:
            raise ValueError(f"Unknown option label {label!r} for variant {self.id}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "profile": self.profile,
            "option_labels": list(self.option_labels),
            "action_labels": list(self.action_labels),
            "outcome_label": self.outcome_label,
            "source": prompt_source_record(self.profile),
        }


def prompt_source_record(profile_name: str, game_name: str | None = None) -> dict[str, Any]:
    try:
        profile = PROMPT_PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt profile {profile_name!r}") from exc
    paths = profile.source_paths
    if profile_name == "paper_canonical" and game_name == "prisoners_dilemma":
        paths = ("pd/query_main.py",)
    elif profile_name == "paper_canonical" and game_name == "battle_of_the_sexes":
        paths = (
            "bos/query_main.py",
            "bos/variations_robustness/query_robustness_checks_game.py",
        )
    return {
        "repository": UPSTREAM_REPOSITORY,
        "commit": UPSTREAM_COMMIT,
        "paths": list(paths),
        "urls": [f"{UPSTREAM_REPOSITORY}/blob/{UPSTREAM_COMMIT}/{path}" for path in paths],
        "license": UPSTREAM_LICENSE,
    }


def build_prompt_variants(config: dict[str, Any]) -> list[PromptVariant]:
    names = list(config.get("profiles", ["paper_canonical", "paper_cooking"]))
    unknown = sorted(set(names) - set(PROMPT_PROFILES))
    if unknown:
        raise ValueError(f"Unknown prompt profiles: {unknown}")
    raw_pairs = config.get("option_pairs", [list(pair) for pair in PAPER_OPTION_PAIRS])
    pairs: list[tuple[str, str]] = []
    for raw in raw_pairs:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError("Each option pair must contain exactly two labels")
        pair = (str(raw[0]), str(raw[1]))
        if pair[0] == pair[1] or not all(label.isalpha() and len(label) == 1 for label in pair):
            raise ValueError(f"Option labels must be distinct single letters: {pair}")
        pairs.append(pair)
    if len(pairs) != len(set(pairs)):
        raise ValueError("Option pairs must be unique")

    selected_outcomes = config.get("outcome_labels", {})
    counterbalance = bool(config.get("counterbalance_action_mapping", True))
    variants: list[PromptVariant] = []
    for pair in pairs:
        for name in names:
            profile = PROMPT_PROFILES[name]
            outcomes = tuple(selected_outcomes.get(name, profile.outcome_labels[:1]))
            if not outcomes:
                raise ValueError(f"At least one outcome label is required for {name}")
            invalid = sorted(set(outcomes) - set(profile.outcome_labels))
            if invalid:
                raise ValueError(f"Outcome labels not published for {name}: {invalid}")
            mappings = (pair, (pair[1], pair[0])) if counterbalance else (pair,)
            for mapping in mappings:
                for outcome in outcomes:
                    variants.append(PromptVariant(name, pair, mapping, str(outcome)))
    return variants


def _label_action_map(game: GameSpec, variant: PromptVariant) -> dict[str, str]:
    return dict(zip(variant.action_labels, game.actions))


def _rule_lines(game: GameSpec, player_index: int, variant: PromptVariant, choice_word: str) -> str:
    label_to_action = _label_action_map(game, variant)
    lines: list[str] = []
    for own_label in variant.option_labels:
        for other_label in variant.option_labels:
            own, other = game.ego_payoff(
                player_index,
                label_to_action[own_label],
                label_to_action[other_label],
            )
            lines.append(
                f"If you choose {choice_word} {own_label} and the other {('player' if variant.profile == 'paper_canonical' else 'contestant' if variant.profile == 'paper_cooking' else 'colleague')} "
                f"chooses {choice_word} {other_label}, then you win {own} {variant.outcome_label} "
                f"and the other {('player' if variant.profile == 'paper_canonical' else 'contestant' if variant.profile == 'paper_cooking' else 'colleague')} wins {other} {variant.outcome_label}.\n"
            )
    return "".join(lines)


def build_instruction(
    game: GameSpec,
    player_index: int,
    variant: PromptVariant,
    total_rounds: int,
) -> str:
    first, second = variant.option_labels
    if variant.profile == "paper_canonical":
        opening = (
            f"You are playing a game repeatedly with another player. In this game, you can choose between Option {first} and Option {second}. "
            f"You will play {total_rounds} rounds in total with the same player. "
            "The rules of the game are as follows:\n"
        )
        return opening + _rule_lines(game, player_index, variant, "Option") + "\n"
    if variant.profile == "paper_cooking":
        opening = (
            f"You are participating in a cooking competition with another contestant. In this competition, you can choose between Recipe {first} and Recipe {second}. "
            f"You will prepare {total_rounds} dishes in total with the same contestant. "
            "The competition rules are as follows:\n"
        )
        return opening + _rule_lines(game, player_index, variant, "Recipe") + "\n"
    if variant.profile == "paper_project":
        opening = (
            f"You are working on a collaborative project with another colleague. In this project, you can choose between Approach {first} and Approach {second}. "
            f"You will work on {total_rounds} tasks in total with the same colleague. "
            "The guidelines of the project are as follows:\n"
        )
        return opening + _rule_lines(game, player_index, variant, "Approach") + "\n"
    raise ValueError(f"Unknown prompt profile {variant.profile!r}")


def build_query(variant: PromptVariant, round_number: int, option_order: tuple[str, str]) -> str:
    first, second = option_order
    if set(option_order) != set(variant.option_labels):
        raise ValueError("Query option order must contain the variant's two option labels")
    if variant.profile == "paper_canonical":
        return (
            f"\nYou are currently playing round {round_number}.\n"
            f"Q: Which Option do you choose, Option {first} or Option {second}?\nA: Option"
        )
    if variant.profile == "paper_cooking":
        return (
            f"You are currently preparing dish {round_number}.\n"
            f"Q: Which Recipe do you choose, Recipe {first} or Recipe {second}?\nA: Recipe"
        )
    if variant.profile == "paper_project":
        return (
            f"You are currently working on task {round_number}.\n"
            f"Q: Which Approach do you choose, Approach {first} or Approach {second}?\nA: Approach"
        )
    raise ValueError(f"Unknown prompt profile {variant.profile!r}")


def build_history_line(
    variant: PromptVariant,
    round_number: int,
    own_label: str,
    other_label: str,
    own_payoff: int,
    other_payoff: int,
) -> str:
    outcome = variant.outcome_label
    if variant.profile == "paper_canonical":
        return (
            f"In round {round_number}, you chose Option {own_label} and the other player chose Option {other_label}. "
            f"Thus, you won {own_payoff} {outcome} and the other player won {other_payoff} {outcome}.\n"
        )
    if variant.profile == "paper_cooking":
        return (
            f"In dish {round_number}, you chose Recipe {own_label} and the other contestant chose Recipe {other_label}. "
            f"Thus, you won {own_payoff} {outcome} and the other contestant won {other_payoff} {outcome}.\n"
        )
    if variant.profile == "paper_project":
        # "In project" is retained from the published source rather than silently corrected.
        return (
            f"In project {round_number}, you chose Approach {own_label} and the other colleague chose Approach {other_label}. "
            f"Thus, you won {own_payoff} {outcome} and the other colleague won {other_payoff} {outcome}.\n"
        )
    raise ValueError(f"Unknown prompt profile {variant.profile!r}")


def prompt_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def has_verbatim_upstream_instruction(
    game: GameSpec,
    player_index: int,
    variant: PromptVariant,
    total_rounds: int,
) -> bool:
    """Whether the rendered instruction matches a role present in a cited source script."""

    direct = variant.action_labels == variant.option_labels
    if total_rounds != 10 or not direct:
        return False
    if variant.profile == "paper_canonical":
        if game.name == "prisoners_dilemma":
            return variant.option_labels == ("J", "F") and variant.outcome_label == "points"
        if game.name == "battle_of_the_sexes":
            published_player1 = (
                player_index == 0
                and variant.option_labels in PAPER_OPTION_PAIRS
                and variant.outcome_label in PROMPT_PROFILES["paper_canonical"].outcome_labels
            )
            published_player2 = (
                player_index == 1
                and variant.option_labels == ("J", "F")
                and variant.outcome_label == "points"
            )
            return published_player1 or published_player2
        return False
    if variant.profile in {"paper_cooking", "paper_project"}:
        return game.name == "battle_of_the_sexes" and player_index == 0
    return False


def prompt_provenance(
    game: GameSpec,
    player_index: int,
    variant: PromptVariant,
    total_rounds: int,
) -> dict[str, Any]:
    source = prompt_source_record(variant.profile, game.name)
    verbatim = has_verbatim_upstream_instruction(game, player_index, variant, total_rounds)
    role_mirroring_is_adapted = (
        player_index == 1
        and variant.profile != "paper_canonical"
    )
    return {
        **source,
        "verbatim_upstream_instruction": verbatim,
        "adaptations": []
        if verbatim
        else [
            item
            for item, applies in (
                ("ego-centric role mirroring", role_mirroring_is_adapted),
                ("study-specific Stag Hunt payoffs", game.name == "stag_hunt"),
                ("counterbalanced action-to-label mapping", variant.action_labels != variant.option_labels),
                ("configured playing horizon", total_rounds != 10),
                ("paper cover story applied to another game", game.name != "battle_of_the_sexes" and variant.profile != "paper_canonical"),
                (
                    "neutral option-label substitution",
                    variant.profile == "paper_canonical"
                    and game.name == "prisoners_dilemma"
                    and variant.option_labels != ("J", "F"),
                ),
                (
                    "role-specific neutral-label variant not present upstream",
                    variant.profile == "paper_canonical"
                    and game.name == "battle_of_the_sexes"
                    and player_index == 1
                    and variant.option_labels != ("J", "F"),
                ),
            )
            if applies
        ],
    }
