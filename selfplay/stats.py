"""Paired statistics over problems: bootstrap CI, exact McNemar test and minimum detectable effect.

Inputs are per-problem scores for two models on the same problems, in the same order: 0/1 correctness, or the
fraction of samples correct when several samples are drawn per problem. Pooling across benchmarks means
concatenating their per-problem vectors; `strata` keeps the bootstrap resampling within each benchmark.
"""

import math
from collections.abc import Sequence
from statistics import NormalDist

import numpy as np


def _z(p: float) -> float:
    return NormalDist().inv_cdf(p)


def _resampled_sums(values: np.ndarray, n_boot: int, rng: np.random.Generator, block: int = 1000) -> np.ndarray:
    """Sum of each of n_boot resamples (with replacement) of values, drawn in blocks to bound memory."""
    sums = np.empty(n_boot)
    for start in range(0, n_boot, block):
        size = min(block, n_boot - start)
        sums[start:start + size] = values[rng.integers(0, len(values), size=(size, len(values)))].sum(axis=1)
    return sums


def paired_bootstrap(a: Sequence[float], b: Sequence[float], *, n_boot: int = 10_000, alpha: float = 0.05,
                     seed: int = 0, strata: Sequence | None = None) -> dict:
    """Mean of b - a with a percentile bootstrap CI; problems are resampled with replacement."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("a and b must be non-empty 1-D sequences of equal length")
    diff, rng = b - a, np.random.default_rng(seed)
    if strata is None:
        boots = _resampled_sums(diff, n_boot, rng)
    else:
        strata = np.asarray(strata)
        if strata.shape != diff.shape:
            raise ValueError("strata must have the same length as a and b")
        boots = sum(_resampled_sums(diff[strata == s], n_boot, rng) for s in np.unique(strata))
    boots = boots / len(diff)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": len(diff),
            "n_boot": n_boot, "alpha": alpha}


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> dict:
    """Exact two-sided McNemar test on paired 0/1 outcomes (binomial test on the discordant pairs)."""
    if len(a) != len(b):
        raise ValueError("a and b must have equal length")
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = only_a + only_b
    if n == 0:
        p = 1.0
    else:
        k = min(only_a, only_b)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"only_a_correct": only_a, "only_b_correct": only_b, "discordant": n, "p_value": p}


def ci_half_width(discordance: float, n: int, alpha: float = 0.05) -> float:
    """Approximate 95% half-width of a paired accuracy difference near zero effect: z * sqrt(d / n).

    d is the fraction of problems on which the two models disagree. This is the resolution quoted in the project
    notes (MATH-500, d = 8-16%: about 2.5-3.5 points).
    """
    return _z(1 - alpha / 2) * math.sqrt(discordance / n)


def minimum_detectable_effect(discordance: float, n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest true difference detected with the given power by a two-sided paired test: (z_a + z_b) sqrt(d/n)."""
    return (_z(1 - alpha / 2) + _z(power)) * math.sqrt(discordance / n)


def discordance(a: Sequence[int], b: Sequence[int]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("a and b must be non-empty and of equal length")
    return sum(bool(x) != bool(y) for x, y in zip(a, b, strict=True)) / len(a)


def split_half_discordance(samples: Sequence[Sequence[int]]) -> float:
    """Discordance between two independent draws of the same model, from per-problem lists of >= 2 samples.

    Used before training to estimate d, and so the detectable effect, from base-model samples alone: the first
    and second sample of each problem act as two "models" with the same true accuracy.
    """
    pairs = [(s[0], s[1]) for s in samples if len(s) >= 2]
    if not pairs:
        raise ValueError("need at least one problem with two samples")
    return sum(bool(x) != bool(y) for x, y in pairs) / len(pairs)
