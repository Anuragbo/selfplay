"""Scripted Kuhn Poker opponents.

Each policy maps (card, first_to_act, legal_actions, rng) to an action string such as "[bet]". ScriptedAgent wraps a
policy as a TextArena-style agent: it reads the same text observation a language model sees, so it can stand in
for a fixed opponent during training and evaluation.
"""

import random
import re
from collections.abc import Callable

J, Q, K = 0, 1, 2
CARD_RANKS = {"J": J, "Q": Q, "K": K}

Policy = Callable[[int, bool, set, random.Random], str]


def equilibrium(card: int, first: bool, legal: set, rng: random.Random, alpha: float = 1 / 6) -> str:
    """Textbook Nash equilibrium of standard Kuhn Poker, alpha in [0, 1/3]."""
    facing_bet = "call" in legal
    if first and not facing_bet:
        p_bet = {J: alpha, Q: 0.0, K: 3 * alpha}[card]
        return "[bet]" if rng.random() < p_bet else "[check]"
    if first:
        p_call = {J: 0.0, Q: alpha + 1 / 3, K: 1.0}[card]
        return "[call]" if rng.random() < p_call else "[fold]"
    if not facing_bet:
        p_bet = {J: 1 / 3, Q: 0.0, K: 1.0}[card]
        return "[bet]" if rng.random() < p_bet else "[check]"
    p_call = {J: 0.0, Q: 1 / 3, K: 1.0}[card]
    return "[call]" if rng.random() < p_call else "[fold]"


def never_fold(card: int, first: bool, legal: set, rng: random.Random) -> str:
    """Bet whenever possible; call every bet."""
    return "[bet]" if "bet" in legal else "[call]"


def calling_station(card: int, first: bool, legal: set, rng: random.Random) -> str:
    """Check whenever possible; call every bet."""
    return "[check]" if "check" in legal else "[call]"


def uniform_random(card: int, first: bool, legal: set, rng: random.Random) -> str:
    return f"[{rng.choice(sorted(legal))}]"


POLICIES: dict[str, Policy] = {
    "equilibrium": equilibrium,
    "never_fold": never_fold,
    "calling_station": calling_station,
    "random": uniform_random,
}

SCRIPTED_PREFIX = "scripted:"

_HAND_START = "Starting round "
_CARD = re.compile(r"Your card is: ([JQK])")
_OPENER = re.compile(r"Your card is: [JQK]\nYour available actions are:")
_LEGAL = re.compile(r"Your available actions are: ((?:\[\w+\](?:, )?)+)")


def parse_observation(observation: str) -> tuple[int, bool, set]:
    """Current card, whether this player opened the current hand, and the legal actions, from the text history."""
    hand = observation[observation.rindex(_HAND_START):]
    card = CARD_RANKS[_CARD.search(hand).group(1)]
    legal = set(re.findall(r"\[(\w+)\]", _LEGAL.findall(observation)[-1]))
    return card, bool(_OPENER.search(hand)), legal


class ScriptedAgent:
    """Callable agent: observation text in, action text out."""

    def __init__(self, policy: str, seed: int | None = None):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; choose from {sorted(POLICIES)}")
        self.name, self.policy, self.rng = policy, POLICIES[policy], random.Random(seed)

    def __call__(self, observation: str) -> str:
        card, first, legal = parse_observation(observation)
        return self.policy(card, first, legal, self.rng)


def is_scripted(name: str | None) -> bool:
    return bool(name) and name.startswith(SCRIPTED_PREFIX)


def make_scripted(name: str, seed: int | None = None) -> ScriptedAgent:
    """Build an agent from a name such as "scripted:never_fold"."""
    return ScriptedAgent(name.removeprefix(SCRIPTED_PREFIX), seed=seed)
