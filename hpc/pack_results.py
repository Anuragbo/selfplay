"""Pack finished HPC runs into the files committed under results/ (same format as the existing ones).

    python hpc/pack_results.py train <run_id> "<label>"     -> results/train-<run_id>.json
    python hpc/pack_results.py eval <eval_id>                -> results/<eval_id>.json
    python hpc/pack_results.py kuhn <run_id> "<notes>"       -> results/<run_id>.json   (hpc/kuhn_base.sbatch output)

Reads checkpoints/<run_id>/ and outputs/<eval_id>/. Training metrics are cut at the newest complete checkpoint, so
the file matches what a resumed run would continue from.
"""

import argparse
import json
from pathlib import Path

from selfplay.evaluate import _read_jsonl, summarize
from selfplay.train_single import COMPLETE_MARKER, METRICS_FILE


def pack_train(run_id: str, label: str, checkpoints: Path) -> dict:
    d = checkpoints / run_id
    steps = sorted(int(s.name.split("_")[1]) for s in d.glob("step_*") if (s / COMPLETE_MARKER).exists())
    if not steps:
        raise SystemExit(f"no complete checkpoint in {d}")
    rows = [json.loads(line) for line in (d / METRICS_FILE).read_text().splitlines() if line.strip()]
    config = json.loads((d / f"step_{steps[-1]}" / "run_config.json").read_text())
    versions = json.loads((d / "versions.json").read_text()) if (d / "versions.json").exists() else None
    return {"run_id": run_id, "label": label, "config": config, "versions": versions,
            "metrics": [r for r in rows if r["step"] <= steps[-1]]}


def pack_eval(eval_id: str, outputs: Path, checkpoints: Path) -> dict:
    d = outputs / eval_id
    config = json.loads((d / "run_config.json").read_text())
    _, bits = summarize(_read_jsonl(d / "per_problem.jsonl"), {})
    results = [json.loads(line) for line in (d / "results.jsonl").read_text().splitlines() if line.strip()]
    # hpc/eval.sbatch did not pass --seed, so rows say seed 0; the training seed is in the adapter's run_config
    if config.get("lora"):
        seed = json.loads((Path(config["lora"]) / "run_config.json").read_text())["seed"]
        results = [{**r, "seed": seed} for r in results]
    return {"run_id": eval_id, "results": results, "correct": bits, "versions": config["versions"]}


def pack_kuhn(run_id: str, notes: str, outputs: Path) -> dict:
    """Same layout as results/kuhn-base-qwen3-1.7b.json: results rows, a per-match summary and generation settings."""
    from collections import Counter

    d = outputs / run_id
    config = json.loads((d / "run_config.json").read_text())
    matches = _read_jsonl(d / "matches.jsonl")
    results = _read_jsonl(d / "results.jsonl")
    summary = {"matches": len(matches),
               "rewards": dict(Counter(str(m["reward"]) for m in matches)),
               "lost_by_invalid_move": sum(m["lost_by_invalid_move"] for m in matches),
               "wins_by_seat": {str(p): sum(m["reward"] > 0 for m in matches if m["model_pid"] == p) for p in (0, 1)},
               "model_turns_per_match_counts": dict(sorted(Counter(str(m["model_turns"]) for m in matches).items()))}
    generation = {k: config[k] for k in ("temperature", "top_p", "max_tokens", "max_model_len")}
    generation.update(prompt="qwen3-zs (UnstableBaselines)", error_allowance=0)
    return {"run_id": run_id, "results": results, "match_summary": summary, "generation": generation,
            "notes": notes, "versions": config["versions"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=["train", "eval", "kuhn"])
    parser.add_argument("id")
    parser.add_argument("label", nargs="?", default="")
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.kind == "train":
        obj, name = pack_train(args.id, args.label, args.checkpoints), f"train-{args.id}.json"
    elif args.kind == "kuhn":
        obj, name = pack_kuhn(args.id, args.label, args.outputs), f"{args.id}.json"
    else:
        obj, name = pack_eval(args.id, args.outputs, args.checkpoints), f"{args.id}.json"
    path = args.results / name
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
