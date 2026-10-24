import random

import pytest

from selfplay.advantages import RoleAdvantage, SharedBaseline, make_estimator


def test_first_update_matches_formula():
    est = RoleAdvantage(tau=0.1)
    assert est(1.0, pid=0, env_id="kuhn") == pytest.approx(1.0 - 0.1)  # b = 0.9 * 0 + 0.1 * 1


def test_role_baselines_are_separate_per_role_and_env():
    est = RoleAdvantage(tau=0.5)
    est(1.0, pid=0, env_id="a")
    est(-1.0, pid=1, env_id="a")
    est(1.0, pid=0, env_id="b")
    assert est.baselines == {"a": {0: 0.5, 1: -0.5}, "b": {0: 0.5}}


def test_shared_baseline_is_shared_across_roles():
    est = SharedBaseline(tau=0.5)
    est(1.0, pid=0, env_id="a")
    assert est(-1.0, pid=1, env_id="a") == pytest.approx(-1.0 - (0.5 * 0.5 + 0.5 * -1.0))


def _simulate(est, edge: float, n: int = 20_000):
    """Zero-sum game where player 0 wins with probability 0.5 + edge. Returns mean advantage per role."""
    rng, sums = random.Random(0), {0: 0.0, 1: 0.0}
    for _ in range(n):
        r0 = 1.0 if rng.random() < 0.5 + edge else -1.0
        sums[0] += est(r0, 0, "kuhn")
        sums[1] += est(-r0, 1, "kuhn")
    return {pid: s / n for pid, s in sums.items()}


def test_rae_removes_a_seat_advantage():
    """With a 10-point seat edge, RAE advantages average about 0 per role; a shared baseline keeps the edge."""
    rae = _simulate(RoleAdvantage(tau=0.01), edge=0.1)
    shared = _simulate(SharedBaseline(tau=0.01), edge=0.1)
    assert abs(rae[0]) < 0.03 and abs(rae[1]) < 0.03
    assert shared[0] == pytest.approx(0.2, abs=0.03) and shared[1] == pytest.approx(-0.2, abs=0.03)


def test_batch_normalization_is_finite_when_rewards_are_equal():
    from types import SimpleNamespace

    from selfplay.advantages import NormalizeRewardsByEnv

    steps = [SimpleNamespace(env_id="a", reward=r) for r in (1.0, 1.0, 1.0)] + \
            [SimpleNamespace(env_id="b", reward=r) for r in (0.0, 2.0)]
    NormalizeRewardsByEnv()(steps)
    assert [s.reward for s in steps[:3]] == [0.0, 0.0, 0.0]
    assert [s.reward for s in steps[3:]] == pytest.approx([-1.0, 1.0])


def test_make_estimator():
    assert isinstance(make_estimator("rae"), RoleAdvantage)
    assert make_estimator("shared", tau=0.2).tau == 0.2
    with pytest.raises(ValueError):
        make_estimator("gae")
