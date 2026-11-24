"""Pack finished HPC runs into the files committed under results/ (same format as the existing ones).

    python hpc/pack_results.py train <run_id> "<label>"     -> results/train-<run_id>.json
    python hpc/pack_results.py eval <eval_id>                -> results/<eval_id>.json

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


def pack_eval(eval_id: str, outputs: Path) -> dict:
    d = outputs / eval_id
    config = json.loads((d / "run_config.json").read_text())
    _, bits = summarize(_read_jsonl(d / "per_problem.jsonl"), {})
    results = [json.loads(line) for line in (d / "results.jsonl").read_text().splitlines() if line.strip()]
    return {"run_id": eval_id, "results": results, "correct": bits, "versions": config["versions"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=["train", "eval"])
    parser.add_argument("id")
    parser.add_argument("label", nargs="?", default="")
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.kind == "train":
        obj, name = pack_train(args.id, args.label, args.checkpoints), f"train-{args.id}.json"
    else:
        obj, name = pack_eval(args.id, args.outputs), f"{args.id}.json"
    path = args.results / name
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
