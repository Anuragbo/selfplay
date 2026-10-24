"""Advantage estimators for the ablation: role-conditioned (RAE) vs. a shared global baseline.

Both are final-reward transforms with UnstableBaselines' calling convention, (reward, pid, env_id) -> advantage,
so they drop into unstable.reward_transformations.ComposeFinalRewardTransforms without importing it here.

Each keeps an exponential moving average of the final match reward, b <- (1 - tau) * b + tau * R, and returns
R - b. As in UnstableBaselines' RoleAdvantageByEnvFormatter, the baseline is updated with R before subtracting.

- RoleAdvantage: one baseline per (environment, player role). This is SPIRAL's RAE: in a game where one seat has
  an edge, the edge is absorbed by that seat's baseline instead of being credited to the policy.
- SharedBaseline: one baseline per environment, shared by both roles. In a zero-sum game the mean reward over
  both seats is 0, so this baseline stays near 0 and the advantage is close to the raw reward.

UnstableBaselines' own RoleAdvantageFormatter is not used: its constructor stores a float but __call__ indexes it
by player id, so it fails on the first call.
"""

from collections import defaultdict


class RoleAdvantage:
    """EMA baseline per (env_id, pid)."""

    def __init__(self, tau: float = 0.001, initial: float = 0.0):
        self.tau, self.initial = tau, initial
        self.baselines: dict[str | None, dict[int, float]] = defaultdict(dict)

    def __call__(self, reward: float, pid: int, env_id: str | None = None) -> float:
        env = self.baselines[env_id]
        env[pid] = (1 - self.tau) * env.get(pid, self.initial) + self.tau * reward
        return reward - env[pid]


class SharedBaseline:
    """EMA baseline per env_id, the same for every role."""

    def __init__(self, tau: float = 0.001, initial: float = 0.0):
        self.tau, self.initial = tau, initial
        self.baselines: dict[str | None, float] = {}

    def __call__(self, reward: float, pid: int, env_id: str | None = None) -> float:
        baseline = (1 - self.tau) * self.baselines.get(env_id, self.initial) + self.tau * reward
        self.baselines[env_id] = baseline
        return reward - baseline


class NormalizeRewardsByEnv:
    """Per-environment z-score of step rewards within a training batch (a sampling transform).

    Same as UnstableBaselines' NormalizeRewardsByEnv(z_score=True), except that there the epsilon is added after
    dividing, (r - mean) / std + 1e-8, so a batch whose rewards are all equal produces NaN advantages.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, steps: list, env_id: str | None = None) -> list:
        buckets = defaultdict(list)
        for step in steps:
            buckets[step.env_id].append(step)
        for env_steps in buckets.values():
            rewards = [s.reward for s in env_steps]
            mean = sum(rewards) / len(rewards)
            std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
            for s in env_steps:
                s.reward = float((s.reward - mean) / (std + self.eps))
        return steps


ESTIMATORS = {"rae": RoleAdvantage, "shared": SharedBaseline}


def make_estimator(name: str, tau: float = 0.001):
    if name not in ESTIMATORS:
        raise ValueError(f"unknown estimator {name!r}; choose from {sorted(ESTIMATORS)}")
    return ESTIMATORS[name](tau=tau)
