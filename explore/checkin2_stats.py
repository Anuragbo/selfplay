"""Paired comparison of the base models on MATH-500 (Check-in 2), from the evaluation results in results/.

Reads results/eval-base-qwen3-1.7b.json and results/eval-base-qwen3-4b.json, whose "correct" strings hold per-problem
correctness in the same problem order, and writes the per-model accuracy with a 95% interval, the paired bootstrap
interval and exact McNemar test for the difference, and the discordance between the two models.

    python explore/checkin2_stats.py results results/checkin2_stats.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

MODELS = {"Qwen3-1.7B-Base": "eval-base-qwen3-1.7b.json", "Qwen3-4B-Base": "eval-base-qwen3-4b.json"}
BENCHMARK = "MATH-500"


def load_correct(path: Path, benchmark: str = BENCHMARK) -> list[int]:
    """Per-problem correctness (first sample) from an evaluation result file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [int(entry[0]) for entry in data["correct"][benchmark].split(",")]


def accuracy_interval(correct: list[int], z: float = 1.96) -> dict:
    """Accuracy with a normal-approximation 95% interval."""
    p = sum(correct) / len(correct)
    half = z * math.sqrt(p * (1 - p) / len(correct))
    return {"accuracy": p, "ci_low": p - half, "ci_high": p + half, "n": len(correct)}


def paired_summary(a: list[int], b: list[int], n_boot: int = 10_000, seed: int = 0) -> dict:
    """Difference b - a: paired bootstrap interval, exact McNemar test, discordance and the implied half-width."""
    from selfplay.stats import ci_half_width, discordance, mcnemar_exact, paired_bootstrap

    d = discordance(a, b)
    return {"bootstrap": paired_bootstrap(a, b, n_boot=n_boot, seed=seed), "mcnemar": mcnemar_exact(a, b),
            "discordance": d, "ci_half_width_at_this_discordance": ci_half_width(d, len(a))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="directory with the evaluation result JSON files")
    parser.add_argument("output", type=Path, help="output JSON path")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for selfplay.stats when run as a script
    correct = {name: load_correct(args.results / file) for name, file in MODELS.items()}
    small, large = correct.values()
    report = {
        "benchmark": BENCHMARK,
        "sources": {name: f"results/{file}" for name, file in MODELS.items()},
        "accuracy": {name: accuracy_interval(c) for name, c in correct.items()},
        "difference_4b_minus_1_7b": paired_summary(small, large),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for name, acc in report["accuracy"].items():
        print(f"{name:16s} {100 * acc['accuracy']:.1f}%  [{100 * acc['ci_low']:.1f}, {100 * acc['ci_high']:.1f}]")
    diff = report["difference_4b_minus_1_7b"]
    boot = diff["bootstrap"]
    print(f"4B - 1.7B: {100 * boot['mean_diff']:+.1f} points "
          f"[{100 * boot['ci_low']:+.1f}, {100 * boot['ci_high']:+.1f}], McNemar p = {diff['mcnemar']['p_value']:.2g}")


if __name__ == "__main__":
    main()
