"""Pool the trained-vs-base math change over seeds, per arm (pilot / self-play x RAE / shared) and per factor.

A single run's paired bootstrap only covers problem sampling; seeds of one arm disagree by more than that, so the
arm-level interval here also resamples seeds (hierarchical bootstrap: seeds, then problems, with base and trained
answers kept paired). A t interval over the per-seed changes is reported as well; with 2-3 seeds it is very wide,
which is the honest answer at this sample size. Only runs from one GPU type are compared with that GPU's base run.

    python explore/seed_stats.py results results/seed_stats.json                      # MATH-500, H100
    python explore/seed_stats.py results out.json --suffix -suite --benchmark pooled   # full suite, pooled

Runs whose final training reply length fell below results_index.COLLAPSE_TOKENS are flagged; the "excluding
collapsed runs" numbers are a post-hoc sensitivity check, not the primary result.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from results_index import COLLAPSE_TOKENS, build

ARMS = [("self-play", "rae"), ("self-play", "shared"), ("pilot vs never_fold", "rae"),
        ("pilot vs never_fold", "shared"), ("self-play + KL", "rae"), ("self-play + KL", "shared")]
FACTORS = {"self-play minus pilot": ("arm", "self-play", "pilot vs never_fold"),
           "RAE minus shared": ("estimator", "rae", "shared"),
           "self-play + KL minus self-play": ("arm", "self-play + KL", "self-play")}
DESIGN_ARMS = {"self-play", "pilot vs never_fold"}  # the original 2x2: used by the estimator contrast and the pool
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def _matches_suffix(name: str, suffix: str) -> bool:
    return name.endswith(f"{suffix}.json") and (suffix or not name.endswith("-suite.json"))


def load_scores(path: Path, benchmark: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-problem 0/1 scores and a stratum (benchmark) label per problem; 'pooled' concatenates all benchmarks."""
    correct = json.loads(path.read_text(encoding="utf-8"))["correct"]
    names = sorted(correct) if benchmark == "pooled" else [benchmark]
    scores = [np.array([float(c.split("/")[0]) for c in correct[b].split(",")]) for b in names]
    return np.concatenate(scores), np.concatenate([np.full(len(s), i) for i, s in enumerate(scores)])


def collect(results: Path, gpu: str, suffix: str, benchmark: str, step: int = 30) -> tuple[dict, list[dict]]:
    entries = build(results)
    evals = [e for e in entries if e["kind"] == "eval" and e["gpu"] == gpu and _matches_suffix(e["file"], suffix)]
    base = [e for e in evals if e["arm"] == "base model" and e["model"] == "Qwen/Qwen3-1.7B-Base"]
    if len(base) != 1:
        raise SystemExit(f"expected one 1.7B base eval on {gpu} with suffix {suffix!r}, found {len(base)}")
    tokens = {(e["run_id"], e["gpu"]): e["final_tokens_per_turn"] for e in entries if e["kind"] == "train"}
    runs = []
    for e in evals:
        if e["arm"] == "base model" or e["step"] != step:
            continue
        run_id = e["run_id"].removeprefix("eval-").removesuffix(suffix).removesuffix(f"-step{step}")
        final = tokens.get((run_id, gpu))
        runs.append({**e, "train_run_id": run_id, "final_tokens_per_turn": final,
                     "collapsed": final is not None and final < COLLAPSE_TOKENS})
    return base[0], runs


def _stratified_index(strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([rng.choice(np.flatnonzero(strata == s), size=(strata == s).sum())
                           for s in np.unique(strata)])


def hierarchical_ci(deltas: np.ndarray, strata: np.ndarray, *, n_boot: int = 10_000, seed: int = 0) -> tuple:
    """95% percentile interval of the mean change, resampling seeds (rows) and then problems (columns)."""
    rng = np.random.default_rng(seed)
    n_seeds = deltas.shape[0]
    est = np.empty(n_boot)
    for i in range(n_boot):
        rows = rng.integers(0, n_seeds, n_seeds)
        est[i] = deltas[rows][:, _stratified_index(strata, rng)].mean()
    return float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))


def t_interval(values: list[float]) -> tuple | None:
    if len(values) < 2:
        return None
    mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
    half = T95[len(values) - 1] * sd / math.sqrt(len(values))
    return mean - half, mean + half


def summarize_group(runs: list[dict], base_scores: np.ndarray, strata: np.ndarray, paths: Path,
                    benchmark: str) -> dict:
    deltas = np.stack([load_scores(paths / r["file"], benchmark)[0] - base_scores for r in runs])
    per_seed = deltas.mean(axis=1)
    lo, hi = hierarchical_ci(deltas, strata)
    t_ci = t_interval(list(per_seed))
    return {"n_seeds": len(runs), "mean_change": float(per_seed.mean()),
            "sd_across_seeds": float(np.std(per_seed, ddof=1)) if len(runs) > 1 else None,
            "hierarchical_ci": [lo, hi], "t_ci_over_seeds": list(t_ci) if t_ci else None,
            "seeds_positive": int((per_seed > 0).sum())}


def contrast(a: list[dict], b: list[dict], base_scores: np.ndarray, strata: np.ndarray, paths: Path,
             benchmark: str, *, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Mean change of group a minus group b, resampling seeds within each group and problems jointly."""
    da = np.stack([load_scores(paths / r["file"], benchmark)[0] - base_scores for r in a])
    db = np.stack([load_scores(paths / r["file"], benchmark)[0] - base_scores for r in b])
    rng = np.random.default_rng(seed)
    est = np.empty(n_boot)
    for i in range(n_boot):
        cols = _stratified_index(strata, rng)
        est[i] = (da[rng.integers(0, len(a), len(a))][:, cols].mean()
                  - db[rng.integers(0, len(b), len(b))][:, cols].mean())
    return {"difference": float(da.mean() - db.mean()), "hierarchical_ci": [float(np.quantile(est, 0.025)),
            float(np.quantile(est, 0.975))], "n_runs": [len(a), len(b)]}


def analyse(results: Path, gpu: str, suffix: str, benchmark: str) -> dict:
    base, runs = collect(results, gpu, suffix, benchmark)
    base_scores, strata = load_scores(results / base["file"], benchmark)

    def block(selected: list[dict]) -> dict:
        arms = []
        for arm, estimator in ARMS:
            group = sorted((r for r in selected if r["arm"] == arm and r["estimator"] == estimator),
                           key=lambda r: r["seed"])
            if group:
                arms.append({"arm": arm, "estimator": estimator,
                             **summarize_group(group, base_scores, strata, results, benchmark)})
        factors = {}
        design = [r for r in selected if r["arm"] in DESIGN_ARMS]
        for label, (key, left, right) in FACTORS.items():
            pool = selected if key == "arm" else design
            a, b = [r for r in pool if r[key] == left], [r for r in pool if r[key] == right]
            if a and b:
                factors[label] = contrast(a, b, base_scores, strata, results, benchmark)
        overall = summarize_group(design, base_scores, strata, results, benchmark) if design else None
        return {"arms": arms, "factors": factors, "all_trained_runs": overall}

    per_run = []
    for r in sorted(runs, key=lambda r: (r["arm"], r["estimator"], r["seed"])):
        scores = load_scores(results / r["file"], benchmark)[0]
        per_run.append({"file": r["file"], "arm": r["arm"], "estimator": r["estimator"], "seed": r["seed"],
                        "accuracy": float(scores.mean()), "change": float((scores - base_scores).mean()),
                        "final_tokens_per_turn": r["final_tokens_per_turn"], "collapsed": r["collapsed"]})
    return {"gpu": gpu, "benchmark": benchmark, "suffix": suffix, "base_file": base["file"],
            "base_accuracy": float(base_scores.mean()), "n_problems": len(base_scores), "runs": per_run,
            "primary": block(runs), "excluding_collapsed": block([r for r in runs if not r["collapsed"]]),
            "collapse_threshold_tokens_per_turn": COLLAPSE_TOKENS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--gpu", default="NVIDIA H100 PCIe")
    parser.add_argument("--suffix", default="", help="eval file set, e.g. -suite for the six-benchmark evals")
    parser.add_argument("--benchmark", default="MATH-500", help="one benchmark name, or 'pooled'")
    args = parser.parse_args()
    out = analyse(args.results, args.gpu, args.suffix, args.benchmark)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for label, key in (("all runs", "primary"), ("excluding collapsed", "excluding_collapsed")):
        print(f"--- {label} (base {100 * out['base_accuracy']:.1f}%, {out['benchmark']}, {out['gpu']})")
        for a in out[key]["arms"]:
            lo, hi = a["hierarchical_ci"]
            print(f"{a['arm']:>20} {a['estimator']:<6} n={a['n_seeds']} change {100 * a['mean_change']:+.1f} "
                  f"[{100 * lo:+.1f}, {100 * hi:+.1f}]  positive {a['seeds_positive']}/{a['n_seeds']}")
        for name, f in out[key]["factors"].items():
            lo, hi = f["hierarchical_ci"]
            print(f"{name:>27}: {100 * f['difference']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")


if __name__ == "__main__":
    main()
