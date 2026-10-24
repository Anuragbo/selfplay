import random

import pytest

from selfplay.stats import (
    ci_half_width,
    discordance,
    mcnemar_exact,
    minimum_detectable_effect,
    paired_bootstrap,
    split_half_discordance,
)


def test_half_width_matches_project_notes():
    """PROJECT_NOTES section 4: MATH-500 alone resolves about +-2.5 to 3.5 points for d = 8-16%."""
    assert ci_half_width(0.08, 500) == pytest.approx(0.0248, abs=5e-4)
    assert ci_half_width(0.16, 500) == pytest.approx(0.0351, abs=5e-4)
    assert ci_half_width(0.114, 1517) == pytest.approx(0.017, abs=5e-4)  # pooled suite, about +-1.7


def test_mde_is_larger_than_half_width():
    assert minimum_detectable_effect(0.1, 500) == pytest.approx(ci_half_width(0.1, 500) * (1.96 + 0.8416) / 1.96,
                                                                 rel=1e-3)


def test_mcnemar_exact_known_values():
    assert mcnemar_exact([1] * 10, [0] * 10)["p_value"] == pytest.approx(2 / 2 ** 10)
    assert mcnemar_exact([1, 0, 1], [1, 0, 1])["p_value"] == 1.0
    r = mcnemar_exact([1, 1, 0, 0, 1], [0, 1, 1, 1, 1])
    assert (r["only_a_correct"], r["only_b_correct"]) == (1, 2)
    assert r["p_value"] == pytest.approx(1.0)


def test_bootstrap_ci_covers_true_difference():
    rng = random.Random(0)
    a = [int(rng.random() < 0.70) for _ in range(2000)]
    b = [x if rng.random() < 0.9 else int(rng.random() < 0.75) for x in a]
    r = paired_bootstrap(a, b, n_boot=2000)
    assert r["ci_low"] < r["mean_diff"] < r["ci_high"]
    assert r["ci_high"] - r["ci_low"] < 0.05


def test_bootstrap_identical_models_has_zero_width():
    r = paired_bootstrap([1, 0, 1, 1], [1, 0, 1, 1], n_boot=500)
    assert r["mean_diff"] == r["ci_low"] == r["ci_high"] == 0.0


def test_stratified_bootstrap_runs_and_is_reproducible():
    a, b = [1, 0, 1, 0, 1, 1], [1, 1, 1, 0, 0, 1]
    strata = ["MATH-500"] * 3 + ["AIME 2024"] * 3
    r1 = paired_bootstrap(a, b, n_boot=1000, strata=strata, seed=3)
    r2 = paired_bootstrap(a, b, n_boot=1000, strata=strata, seed=3)
    assert r1 == r2 and r1["mean_diff"] == 0.0


def test_discordance():
    assert discordance([1, 0, 1, 0], [1, 1, 0, 0]) == 0.5
    assert split_half_discordance([[1, 0, 1], [1, 1], [0, 0]]) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        paired_bootstrap([1, 0], [1])
