import random

import pytest

from selfplay.opponents import (
    POLICIES,
    J,
    K,
    Q,
    ScriptedAgent,
    is_scripted,
    make_scripted,
    parse_observation,
)

LEGAL_STATES = [{"check", "bet"}, {"fold", "call"}]


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_policies_only_return_legal_actions(name):
    rng = random.Random(0)
    for card in (J, Q, K):
        for first in (True, False):
            for legal in LEGAL_STATES:
                for _ in range(50):
                    assert POLICIES[name](card, first, legal, rng).strip("[]") in legal


def test_equilibrium_frequencies():
    rng, n, policy = random.Random(1), 60_000, POLICIES["equilibrium"]
    opener_bets_king = sum(policy(K, True, {"check", "bet"}, rng) == "[bet]" for _ in range(n)) / n
    responder_calls_queen = sum(policy(Q, False, {"fold", "call"}, rng) == "[call]" for _ in range(n)) / n
    assert opener_bets_king == pytest.approx(0.5, abs=0.01)  # 3 * alpha with alpha = 1/6
    assert responder_calls_queen == pytest.approx(1 / 3, abs=0.01)


def test_never_fold_and_calling_station():
    rng = random.Random(0)
    assert POLICIES["never_fold"](J, True, {"check", "bet"}, rng) == "[bet]"
    assert POLICIES["never_fold"](J, False, {"fold", "call"}, rng) == "[call]"
    assert POLICIES["calling_station"](K, True, {"check", "bet"}, rng) == "[check]"
    assert POLICIES["calling_station"](J, False, {"fold", "call"}, rng) == "[call]"


def test_names():
    assert is_scripted("scripted:never_fold") and not is_scripted("google/gemini-2.0-flash-lite-001")
    assert not is_scripted(None)
    assert make_scripted("scripted:random", seed=0).name == "random"
    with pytest.raises(ValueError):
        ScriptedAgent("always_raise")


textarena = pytest.importorskip("textarena")


def test_parsed_observation_matches_game_state():
    """Card, opener and legal actions read from the text agree with the true state at every decision."""
    from selfplay.envs import SPIRAL_KUHN_ID, register_envs

    register_envs()
    for seed in range(200):
        env = textarena.make(SPIRAL_KUHN_ID)
        env.reset(num_players=2, seed=seed)
        rng, done = random.Random(seed), False
        while not done:
            pid, obs = env.get_observation()
            gs = env.state.game_state
            legal = set(gs["current_legal_action_tree"])
            truth = (gs["player_cards"][pid], gs["starting_player"] == pid, legal)
            assert parse_observation(obs) == truth
            done, _ = env.step(f"[{rng.choice(sorted(legal))}]")


def test_scripted_agents_play_complete_matches():
    from selfplay.envs import SPIRAL_KUHN_ID, register_envs

    register_envs()
    wins = 0
    for seed in range(400):
        env = textarena.make(SPIRAL_KUHN_ID)
        env.reset(num_players=2, seed=seed)
        env.state.error_allowance = 0  # as UnstableBaselines runs games
        agents = {seed % 2: make_scripted("scripted:never_fold"), 1 - seed % 2: make_scripted("scripted:random", seed)}
        done = False
        while not done:
            pid, obs = env.get_observation()
            done, _ = env.step(agents[pid](obs))
        rewards, game_info = env.close()
        assert not any(info["invalid_move"] for info in game_info.values())
        wins += rewards[seed % 2] == 1
    assert wins / 400 > 0.6  # never fold beats random (69% in results/kuhn_spiral.json)
