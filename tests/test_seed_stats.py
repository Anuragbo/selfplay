import json

import numpy as np
import pytest

from results_index import build, to_markdown
from seed_stats import analyse, hierarchical_ci, load_scores, t_interval

H100 = {"gpu": "NVIDIA H100 PCIe"}


def _eval(path, run_id, bits, *, estimator=None, seed=0, step=30, versions=H100, extra=None):
    correct = {"MATH-500": bits, **(extra or {})}
    rows = [{"run_id": run_id, "metric": "accuracy", "benchmark": b, "value": np.mean([int(c) for c in v.split(",")]),
             "model": "Qwen/Qwen3-1.7B-Base", "estimator": estimator, "seed": seed, "step": step}
            for b, v in correct.items()]
    path.write_text(json.dumps({"run_id": run_id, "results": rows, "correct": correct, "versions": versions}))


def _train(path, run_id, estimator, seed, tokens, versions=H100):
    rows = [{"step": s, "completion_tokens_per_turn": tokens} for s in (1, 2)]
    path.write_text(json.dumps({"run_id": run_id, "label": run_id, "versions": versions, "metrics": rows,
                                "config": {"model": "Qwen/Qwen3-1.7B-Base", "estimator": estimator, "seed": seed,
                                           "steps": 2}}))


@pytest.fixture
def results(tmp_path):
    _eval(tmp_path / "eval-base-qwen3-1.7b-hpc.json", "eval-base-qwen3-1.7b-hpc", "0,0,1,1", estimator="none", step=0)
    runs = [("rae", 1, "1,0,1,1", 200), ("rae", 2, "1,1,1,1", 30), ("shared", 1, "0,0,0,1", 200)]
    for est, seed, bits, tokens in runs:
        run = f"selfplay_qwen3-1.7b_{est}_s{seed}"
        _eval(tmp_path / f"eval-{run}-step30.json", f"eval-{run}-step30", bits, estimator=est, seed=seed)
        _train(tmp_path / f"train-{run}.json", run, est, seed, tokens)
    _eval(tmp_path / "eval-pilot_qwen3-1.7b_rae_s1-step30.json", "eval-pilot_qwen3-1.7b_rae_s1-step30", "0,0,1,0",
          estimator="rae", seed=1)
    # same run on another GPU must not be mixed in, and has no matching base
    _eval(tmp_path / "eval-pilot_qwen3-1.7b_rae_s0-step30.json", "eval-pilot_qwen3-1.7b_rae_s0-step30", "1,1,1,1",
          estimator="rae", seed=0, versions={"gpu": "Tesla T4"})
    return tmp_path


def test_index_labels_hardware_arm_and_collapse(results):
    entries = {e["file"]: e for e in build(results)}
    assert entries["eval-base-qwen3-1.7b-hpc.json"]["arm"] == "base model"
    assert entries["eval-pilot_qwen3-1.7b_rae_s0-step30.json"]["source"].startswith("Colab T4")
    run = entries["eval-selfplay_qwen3-1.7b_rae_s2-step30.json"]
    assert (run["arm"], run["estimator"], run["seed"], run["gpu"]) == ("self-play", "rae", 2, "NVIDIA H100 PCIe")
    assert "collapse" in entries["train-selfplay_qwen3-1.7b_rae_s2.json"]["note"]
    assert "collapse" not in entries["train-selfplay_qwen3-1.7b_rae_s1.json"]["note"]
    assert "## SJSU HPC H100" in to_markdown(list(entries.values()))


def test_index_reads_hardware_from_filename_suffix(tmp_path):
    _eval(tmp_path / "eval-pilot_qwen3-1.7b_rae_s0-step30-rtx5080.json", "x", "1", estimator="rae", versions={})
    assert build(tmp_path)[0]["gpu"] == "NVIDIA GeForce RTX 5080"


def test_arms_pool_seeds_against_same_gpu_base(results):
    out = analyse(results, "NVIDIA H100 PCIe", "", "MATH-500")
    arms = {(a["arm"], a["estimator"]): a for a in out["primary"]["arms"]}
    rae = arms[("self-play", "rae")]
    assert rae["n_seeds"] == 2 and rae["mean_change"] == pytest.approx((0.25 + 0.5) / 2)
    assert rae["seeds_positive"] == 2 and rae["hierarchical_ci"][0] <= rae["mean_change"] <= rae["hierarchical_ci"][1]
    assert arms[("pilot vs never_fold", "rae")]["n_seeds"] == 1  # the T4 seed-0 run is excluded
    assert out["primary"]["factors"]["self-play minus pilot"]["n_runs"] == [3, 1]
    excl = {(a["arm"], a["estimator"]): a for a in out["excluding_collapsed"]["arms"]}
    assert excl[("self-play", "rae")]["n_seeds"] == 1 and excl[("self-play", "rae")]["mean_change"] == 0.25


def test_suffix_selects_suite_files_and_pools_benchmarks(results):
    _eval(results / "eval-base-qwen3-1.7b-hpc-suite.json", "eval-base-qwen3-1.7b-hpc-suite", "0,1",
          estimator="none", step=0, extra={"AMC 2023": "0,0"})
    _eval(results / "eval-selfplay_qwen3-1.7b_rae_s1-step30-suite.json", "eval-selfplay_qwen3-1.7b_rae_s1-step30-suite",
          "1,1", estimator="rae", seed=1, extra={"AMC 2023": "1,0"})
    out = analyse(results, "NVIDIA H100 PCIe", "-suite", "pooled")
    assert out["n_problems"] == 4 and [r["file"] for r in out["runs"]] == [
        "eval-selfplay_qwen3-1.7b_rae_s1-step30-suite.json"]
    assert out["runs"][0]["change"] == pytest.approx(0.5)
    # the MATH-500-only analysis ignores suite files
    math500 = analyse(results, "NVIDIA H100 PCIe", "", "MATH-500")
    assert all(not r["file"].endswith("-suite.json") for r in math500["runs"])


def test_load_scores_strata_and_intervals(results):
    scores, strata = load_scores(results / "eval-base-qwen3-1.7b-hpc.json", "MATH-500")
    assert list(scores) == [0, 0, 1, 1] and set(strata) == {0}
    deltas = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    lo, hi = hierarchical_ci(deltas, strata, n_boot=2000)
    assert lo <= deltas.mean() <= hi
    assert t_interval([1.0]) is None and t_interval([1.0, 1.0]) == (1.0, 1.0)


def test_kl_runs_form_their_own_arm_and_stay_out_of_the_design_pool(results):
    run = "selfplay-kl_qwen3-1.7b_rae_s1"
    _eval(results / f"eval-{run}-step30.json", f"eval-{run}-step30", "1,1,1,1", estimator="rae", seed=1)
    _eval(results / f"eval-{run}-step60.json", f"eval-{run}-step60", "0,0,0,0", estimator="rae", seed=1, step=60)
    out = analyse(results, "NVIDIA H100 PCIe", "", "MATH-500")
    arms = {(a["arm"], a["estimator"]): a for a in out["primary"]["arms"]}
    assert arms[("self-play + KL", "rae")]["n_seeds"] == 1  # step 60 is not mixed with step 30
    assert out["primary"]["factors"]["self-play + KL minus self-play"]["n_runs"] == [1, 3]
    assert out["primary"]["factors"]["RAE minus shared"]["n_runs"] == [3, 1]
    assert out["primary"]["all_trained_runs"]["n_seeds"] == 4
