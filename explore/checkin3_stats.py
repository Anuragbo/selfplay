"""Summarize the pilot training runs (Check-in 3) from the files in results/.

Reads results/train-*.json (per-step training metrics), results/kuhn-base-*.json and, when present,
results/eval-pilot_*.json (MATH-500 on a trained adapter). Reports, per run, the invalid-move loss rate at the start and
end of training with 95% intervals over matches, and the paired MATH-500 change against the untrained base model.

    python explore/checkin3_stats.py results results/checkin3_stats.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

WINDOW = 5
BASE_EVAL = {"Qwen/Qwen3-1.7B-Base": "eval-base-qwen3-1.7b.json", "Qwen/Qwen3-4B-Base": "eval-base-qwen3-4b.json"}


def window_rate(rows: list[dict], key: str) -> dict:
    """Match-weighted mean of a per-step rate, with a normal-approximation 95% interval."""
    n = sum(r["matches"] for r in rows)
    p = sum(r[key] * r["matches"] for r in rows) / n
    half = 1.96 * math.sqrt(p * (1 - p) / n)
    return {"rate": p, "ci_low": max(0.0, p - half), "ci_high": min(1.0, p + half), "matches": n}


def summarize_run(run: dict, window: int = WINDOW) -> dict:
    rows = run["metrics"]
    first, last = rows[:window], rows[-window:]
    seconds = [r["rollout_seconds"] + r["update_seconds"] for r in rows]
    summary = {"run_id": run["run_id"], "steps_completed": rows[-1]["step"], "window_steps": window,
               "seconds_per_step": sum(seconds) / len(seconds), "total_hours": sum(seconds) / 3600}
    for key in ("invalid_move_loss_rate", "win_rate"):
        summary[key] = {"first": window_rate(first, key), "last": window_rate(last, key)}
    for key in ("format_error_rate", "truncated_rate", "completion_tokens_per_turn"):
        summary[key] = {"first": sum(r[key] for r in first) / len(first), "last": sum(r[key] for r in last) / len(last)}
    return summary


def load_correct(path: Path, benchmark: str = "MATH-500") -> list[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [int(entry[0]) for entry in data["correct"][benchmark].split(",")]


def answered_accuracy(metrics: dict) -> float:
    """Accuracy over replies with a boxed answer; grading is strict, so a reply without one is always wrong."""
    return metrics["accuracy"] / (1 - metrics["no_answer_rate"])


def transfer_summary(results: Path, eval_path: Path) -> dict:
    """Trained adapter vs. its base model on the same problems."""
    from selfplay.stats import discordance, mcnemar_exact, paired_bootstrap

    data = json.loads(eval_path.read_text(encoding="utf-8"))
    row = next(r for r in data["results"] if r["metric"] == "accuracy")
    base, trained = load_correct(results / BASE_EVAL[row["model"]]), load_correct(eval_path)
    metrics = {r["metric"]: r["value"] for r in data["results"]}
    base_data = json.loads((results / BASE_EVAL[row["model"]]).read_text(encoding="utf-8"))
    base_metrics = {r["metric"]: r["value"] for r in base_data["results"]}
    change = discordance(base, trained)
    return {"run_id": data["run_id"], "model": row["model"], "step": row["step"], "base": base_metrics,
            "trained": metrics, "bootstrap": paired_bootstrap(base, trained), "mcnemar": mcnemar_exact(base, trained),
            "discordance": change, "detectable_effect": 1.96 * math.sqrt(change / len(base)),
            "accuracy_when_answered": {"base": answered_accuracy(base_metrics),
                                       "trained": answered_accuracy(metrics)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="directory with the result JSON files")
    parser.add_argument("output", type=Path, help="output JSON path")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for selfplay.stats when run as a script
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.results.glob("train-*.json"))]
    report = {"training": [summarize_run(r) for r in runs],
              "transfer": [transfer_summary(args.results, p) for p in sorted(args.results.glob("eval-pilot_*.json"))]}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for t in report["training"]:
        rate = t["invalid_move_loss_rate"]
        print(f"{t['run_id']}: {t['steps_completed']} steps, invalid-move losses "
              f"{100 * rate['first']['rate']:.0f}% -> {100 * rate['last']['rate']:.0f}%")
    for t in report["transfer"]:
        boot = t["bootstrap"]
        print(f"{t['run_id']}: MATH-500 {100 * t['base']['accuracy']:.1f} -> {100 * t['trained']['accuracy']:.1f}, "
              f"diff {100 * boot['mean_diff']:+.1f} [{100 * boot['ci_low']:+.1f}, {100 * boot['ci_high']:+.1f}]")


if __name__ == "__main__":
    main()
