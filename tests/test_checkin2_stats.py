import json
from pathlib import Path

import pytest

from checkin2_stats import accuracy_interval, load_correct, paired_summary

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_correctness_strings_match_reported_accuracy():
    for file in ("eval-base-qwen3-1.7b.json", "eval-base-qwen3-4b.json"):
        data = json.loads((RESULTS / file).read_text(encoding="utf-8"))
        correct = load_correct(RESULTS / file)
        reported = next(r["value"] for r in data["results"] if r["metric"] == "accuracy")
        assert len(correct) == 500 and sum(correct) / 500 == pytest.approx(reported)


def test_paired_summary_and_interval():
    a, b = [1, 0, 0, 1] * 25, [1, 1, 0, 1] * 25
    summary = paired_summary(a, b, n_boot=500)
    assert summary["mcnemar"]["only_b_correct"] == 25 and summary["mcnemar"]["only_a_correct"] == 0
    assert summary["bootstrap"]["mean_diff"] == pytest.approx(0.25)
    acc = accuracy_interval([1] * 70 + [0] * 30)
    assert acc["ci_low"] < 0.7 < acc["ci_high"]


def test_saved_stats_are_reproducible():
    saved = json.loads((RESULTS / "checkin2_stats.json").read_text(encoding="utf-8"))
    small = load_correct(RESULTS / "eval-base-qwen3-1.7b.json")
    large = load_correct(RESULTS / "eval-base-qwen3-4b.json")
    assert paired_summary(small, large) == saved["difference_4b_minus_1_7b"]
