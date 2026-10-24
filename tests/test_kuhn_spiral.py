import json
from pathlib import Path

import pytest

textarena = pytest.importorskip("textarena")

from kuhn_env_explore import POLICIES, payoff_check, play_match, run_matchups  # noqa: E402
from selfplay.envs import SPIRAL_KUHN_ID, register_envs  # noqa: E402
from selfplay.envs.kuhn_spiral import SpiralKuhnPokerEnv  # noqa: E402
from spiral_golden_transcripts import record  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def make_env():
    return SpiralKuhnPokerEnv(ante=1, max_rounds=5)


def test_transcripts_match_spiral_environment():
    """Every observation and reward matches SPIRAL's own env (textarena 0.6.4), invalid moves included."""
    golden = json.loads((ROOT / "tests" / "data" / "spiral_kuhn_golden.json").read_text(encoding="utf-8"))
    assert record(make_env, len(golden)) == golden


def test_reproduces_checkin1_spiral_results():
    """Same 60,000 scripted matches as results/kuhn_spiral.json, bit for bit."""
    expected = json.loads((ROOT / "results" / "kuhn_spiral.json").read_text(encoding="utf-8"))
    report = {"impl": "spiral", "n_games_per_matchup": 10_000, "payoff_check": payoff_check(make_env)}
    report.update(run_matchups(make_env, 10_000))
    assert json.loads(json.dumps(report)) == expected


def test_final_hand_decides_the_match():
    for seed in range(200):
        result = play_match(make_env, POLICIES["random"], POLICIES["equilibrium"], seed=seed)
        assert len(result.hand_winners) == 5
        assert result.rewards[result.hand_winners[-1]] == 1
        assert sum(result.rewards.values()) == 0


def test_called_bet_does_not_change_stakes():
    assert payoff_check(make_env)["called_bet_changes_stakes"] is False


def test_second_invalid_move_loses_with_zero_for_opponent():
    env = make_env()
    env.reset(num_players=2, seed=0)
    pid, _ = env.get_observation()
    assert env.step("no action")[0] is False  # first invalid move is forgiven; same player retries
    assert env.get_observation()[0] == pid
    assert env.step("still no action")[0] is True
    assert env.state.rewards == {pid: -1, 1 - pid: 0}


def test_registered_with_textarena():
    register_envs()
    register_envs()  # idempotent
    env = textarena.make(f"{SPIRAL_KUHN_ID}-raw")
    env.reset(num_players=2, seed=1)
    pid, obs = env.get_observation()
    assert pid == 1  # player 1 opens the first hand
    assert any("Starting round 1 out of 5 rounds." in message for _, message, _ in obs)
