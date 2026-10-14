"""Characterize Kuhn Poker environment variants with scripted policies.

Variants:
  textarena        TextArena (PyPI) KuhnPoker, 3 hands per match.
  textarena_fixed  Same, with standard stakes: a bet or call adds one chip to the pot
                   (the change made on TextArena's GitHub main branch).
  spiral           SPIRAL's KuhnPoker-v1, 5 hands per match. Requires textarena==0.6.4
                   and the path to third_party/spiral_kuhn_env.py.

For each variant the script reports whether a called bet changes the stakes, how the
match winner is decided, win/draw/loss rates for policy match-ups (seats alternate),
and the number of decisions per match.
"""

import argparse
import importlib.util
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

J, Q, K = 0, 1, 2
ACTION = re.compile(r"\[(Check|Bet|Fold|Call)\]", re.IGNORECASE)

Policy = Callable[[int, bool, set, random.Random], str]


# Scripted policies: (card, first_to_act, legal_actions, rng) -> action string.

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

MATCHUPS = [
    ("equilibrium", "equilibrium"),
    ("never_fold", "equilibrium"),
    ("random", "equilibrium"),
    ("random", "never_fold"),
    ("calling_station", "equilibrium"),
    ("never_fold", "never_fold"),
]


def make_env_factory(variant: str, spiral_env: Path | None = None) -> Callable:
    if variant in ("textarena", "textarena_fixed"):
        from textarena.envs.KuhnPoker.env import KuhnPokerEnv

        if variant == "textarena":
            return lambda: KuhnPokerEnv(max_rounds=3)

        class KuhnPokerFixedStakes(KuhnPokerEnv):
            def step(self, action):
                match = ACTION.search(action.strip())
                state = self.state.game_state
                move = match.group(1).lower() if match else None
                if move in ("bet", "call") and move in state["current_legal_action_tree"]:
                    state["player_chips"][self.state.current_player_id] -= 1
                    state["pot"] += 1
                return super().step(action)

        return lambda: KuhnPokerFixedStakes(max_rounds=3)

    if variant == "spiral":
        if spiral_env is None or not spiral_env.is_file():
            raise SystemExit("--spiral-env must point to third_party/spiral_kuhn_env.py")
        spec = importlib.util.spec_from_file_location("spiral_kuhn_env", spiral_env)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return lambda: module.KuhnPokerEnv(ante=1, max_rounds=5)

    raise SystemExit(f"unknown variant: {variant}")


@dataclass
class MatchResult:
    rewards: dict
    decisions: int
    final_chips: dict
    hand_winners: list


def _hand_winner(before: dict, after: dict) -> int | None:
    """Winner of one hand from chip totals; an ante already paid by both players cancels out."""
    diff = (after[0] - before[0]) - (after[1] - before[1])
    return 0 if diff > 0 else 1 if diff < 0 else None


def play_match(make_env: Callable, policy0: Policy, policy1: Policy, seed: int,
               forced_actions: list[str] | None = None) -> MatchResult:
    env = make_env()
    env.reset(num_players=2, seed=seed)
    rng = random.Random(10_000 + seed)
    policies = {0: policy0, 1: policy1}
    forced = list(forced_actions or [])
    decisions, hand_winners = 0, []
    current_hand, chips_at_hand_start = None, None

    while True:
        pid, _ = env.get_observation()
        state = env.state.game_state
        if state["current_round"] != current_hand:
            if current_hand is not None:
                hand_winners.append(_hand_winner(chips_at_hand_start, state["player_chips"]))
            current_hand, chips_at_hand_start = state["current_round"], dict(state["player_chips"])
        legal = set(state["current_legal_action_tree"].keys())
        first_to_act = state["starting_player"] == pid
        action = forced.pop(0) if forced else policies[pid](state["player_cards"][pid], first_to_act, legal, rng)
        done, _ = env.step(action)
        decisions += 1
        if done:
            break

    final_chips = dict(env.state.game_state["player_chips"])
    hand_winners.append(_hand_winner(chips_at_hand_start, final_chips))
    return MatchResult(dict(env.state.rewards or {}), decisions, final_chips, hand_winners)


def _outcome(rewards: dict, pid: int) -> str:
    r = rewards.get(pid, 0)
    return "win" if r > 0 else "loss" if r < 0 else "draw"


def _leader(values: dict) -> int | None:
    return 0 if values.get(0, 0) > values.get(1, 0) else 1 if values.get(1, 0) > values.get(0, 0) else None


def payoff_check(make_env: Callable) -> dict:
    """Play the same deal twice, differing only in the first hand (bet-call vs check-check)."""
    bet_call = play_match(make_env, calling_station, calling_station, seed=3, forced_actions=["[bet]", "[call]"])
    check_check = play_match(make_env, calling_station, calling_station, seed=3, forced_actions=["[check]", "[check]"])
    return {
        "bet_call_final_chips": bet_call.final_chips,
        "check_check_final_chips": check_check.final_chips,
        "called_bet_changes_stakes": bet_call.final_chips != check_check.final_chips,
    }


def run_matchups(make_env: Callable, n_games: int) -> dict:
    matchups, decisions = {}, []
    rule = Counter()
    for a, b in MATCHUPS:
        tally, by_seat = Counter(), defaultdict(Counter)
        for i in range(n_games):
            seat_a = i % 2
            p0, p1 = (POLICIES[a], POLICIES[b]) if seat_a == 0 else (POLICIES[b], POLICIES[a])
            result = play_match(make_env, p0, p1, seed=i)
            outcome = _outcome(result.rewards, seat_a)
            tally[outcome] += 1
            by_seat[f"{a}_as_player{seat_a}"][outcome] += 1
            decisions.append(result.decisions)

            winner = _leader(result.rewards)
            rule["agrees_with_chip_leader"] += winner == _leader(result.final_chips)
            rule["agrees_with_last_hand"] += winner == result.hand_winners[-1]
            rule["games"] += 1

        matchups[f"{a} vs {b}"] = {
            "a_win": tally["win"] / n_games,
            "a_draw": tally["draw"] / n_games,
            "a_loss": tally["loss"] / n_games,
            "by_seat": {seat: {k: v / sum(c.values()) for k, v in c.items()} for seat, c in by_seat.items()},
        }

    return {
        "matchups": matchups,
        "winner_rule": {
            "agrees_with_chip_leader": rule["agrees_with_chip_leader"] / rule["games"],
            "agrees_with_last_hand": rule["agrees_with_last_hand"] / rule["games"],
            "games": rule["games"],
        },
        "decisions_per_match": {
            "mean": sum(decisions) / len(decisions),
            "min": min(decisions),
            "max": max(decisions),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variant", choices=["textarena", "textarena_fixed", "spiral"])
    parser.add_argument("output", type=Path, help="output JSON path")
    parser.add_argument("--spiral-env", type=Path, help="path to third_party/spiral_kuhn_env.py")
    parser.add_argument("--games", type=int, default=10_000, help="matches per match-up (default: 10000)")
    args = parser.parse_args()

    make_env = make_env_factory(args.variant, args.spiral_env)
    report = {"impl": args.variant, "n_games_per_matchup": args.games, "payoff_check": payoff_check(make_env)}
    report.update(run_matchups(make_env, args.games))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"variant: {args.variant}  called bet changes stakes: {report['payoff_check']['called_bet_changes_stakes']}")
    for name, m in report["matchups"].items():
        print(f"  {name:32s} win {m['a_win']:.3f}  draw {m['a_draw']:.3f}  loss {m['a_loss']:.3f}")
    print(f"  winner rule: {report['winner_rule']}")
    print(f"  decisions per match: {report['decisions_per_match']}")


if __name__ == "__main__":
    main()
