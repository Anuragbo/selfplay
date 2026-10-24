"""Evaluate a model (optionally with a LoRA adapter) on the math suite.

Writes to --out-dir:
  per_problem.jsonl   one row per (benchmark, problem, sample): extracted answer, correctness, grading method
  completions.jsonl   the raw completions, same keys
  results.jsonl       rows in the project results contract, one per (benchmark, metric), plus the pooled suite
  run_config.json     model, backend, prompt, sampling settings and package versions

Runs resume: problems already in per_problem.jsonl are skipped, and rows are appended after every chunk, so an
interrupted Colab session loses at most one chunk. With --print-summary the script ends by printing the results
rows and a compact per-problem correctness string per benchmark between SUMMARY_BEGIN / SUMMARY_END markers.

Example (Colab, T4):
  python -m selfplay.evaluate --model Qwen/Qwen3-1.7B-Base --benchmarks math500 --run-id base-1.7b \\
      --out-dir outputs/base-1.7b --print-summary
"""

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from selfplay.benchmarks import SUITE, Problem, load_benchmark
from selfplay.generation import Generate, add_generation_args, make_generator, versions
from selfplay.grading import grade

PROMPTS = {
    "qwen3": ("<|im_start|>user\n{question}\nPlease reason step by step, and put your final answer within "
              "\\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n"),
    "plain": "Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n"
             "Answer:",
}

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _log(message: str) -> None:
    print(message, flush=True)


def evaluate(problems: list[Problem], generate: Generate, prompt: str, n_samples: int, out_dir: Path, *,
             chunk_size: int = 64, log: Callable[[str], None] = _log) -> list[dict]:
    """Generate, grade and append rows for problems not yet in out_dir/per_problem.jsonl."""
    per_problem = out_dir / "per_problem.jsonl"
    done = {(r["benchmark"], r["problem_id"]) for r in _read_jsonl(per_problem)}
    todo = [p for p in problems if (p.benchmark, p.problem_id) not in done]
    log(f"{len(problems) - len(todo)} problems already done, {len(todo)} to go")
    for start in range(0, len(todo), chunk_size):
        chunk, t0 = todo[start:start + chunk_size], time.time()
        outputs = generate([prompt.format(question=p.question) for p in chunk], n_samples)
        rows, completions = [], []
        for p, samples in zip(chunk, outputs, strict=True):
            for k, (text, n_tokens, finish) in enumerate(samples):
                key = {"benchmark": p.benchmark, "problem_id": p.problem_id, "sample": k}
                rows.append({**key, "gold": p.answer, **grade(text, p.answer).to_dict(), "completion_tokens": n_tokens,
                             "finish_reason": finish})
                completions.append({**key, "completion": text})
        _append_jsonl(out_dir / "completions.jsonl", completions)
        _append_jsonl(per_problem, rows)
        acc = sum(r["correct"] for r in rows) / max(len(rows), 1)
        log(f"  {start + len(chunk)}/{len(todo)}  chunk accuracy {acc:.3f}  {time.time() - t0:.0f}s")
    return _read_jsonl(per_problem)


def summarize(rows: list[dict], meta: dict) -> tuple[list[dict], dict[str, str]]:
    """Results-contract rows (per benchmark and pooled) and per-problem correctness strings ('1'/'0' per sample)."""
    by_problem: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_problem[(r["benchmark"], r["problem_id"])].append(r)
    by_benchmark: dict[str, list[list[dict]]] = defaultdict(list)
    for (benchmark, _), samples in sorted(by_problem.items()):
        by_benchmark[benchmark].append(sorted(samples, key=lambda r: r["sample"]))

    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    results, bits = [], {}

    def add(benchmark: str, problems: list[list[dict]]):
        n_samples = min(len(s) for s in problems)
        accuracy = sum(sum(r["correct"] for r in s) / len(s) for s in problems) / len(problems)
        no_answer = sum(r["method"] == "no_answer" for s in problems for r in s) / sum(len(s) for s in problems)
        truncated = sum(r["finish_reason"] == "length" for s in problems for r in s) / sum(len(s) for s in problems)
        for metric, value in (("accuracy", accuracy), ("no_answer_rate", no_answer), ("truncated_rate", truncated)):
            results.append({**meta, "benchmark": benchmark, "metric": metric, "value": value,
                            "n_samples": n_samples, "n_problems": len(problems), "timestamp": stamp})

    for benchmark, problems in by_benchmark.items():
        add(benchmark, problems)
        bits[benchmark] = ",".join("".join("1" if r["correct"] else "0" for r in s) for s in problems)
    if len(by_benchmark) > 1:
        add("pooled", [s for problems in by_benchmark.values() for s in problems])
    return results, bits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_generation_args(parser)
    parser.add_argument("--benchmarks", nargs="+", default=SUITE, choices=SUITE)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="first N problems per benchmark (smoke tests)")
    parser.add_argument("--samples", type=int, default=1, help="samples per problem")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="qwen3")
    parser.add_argument("--chunk-size", type=int, default=64,
                        help="problems per generation batch; larger is faster, smaller loses less on interruption")
    parser.add_argument("--print-summary", action="store_true")
    contract = parser.add_argument_group("results contract")
    contract.add_argument("--run-id", required=True)
    contract.add_argument("--step", type=int, default=0)
    contract.add_argument("--game", default="none")
    contract.add_argument("--estimator", default="none")
    contract.add_argument("--opponent", default="none")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    problems = []
    for key in args.benchmarks:
        problems += load_benchmark(key, args.data_dir)[:args.limit]
    generate = make_generator(args)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(prompt_template=PROMPTS[args.prompt], versions=versions())
    (args.out_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    rows = evaluate(problems, generate, PROMPTS[args.prompt], args.samples, args.out_dir, chunk_size=args.chunk_size)
    wanted = {(p.benchmark, p.problem_id) for p in problems}
    rows = [r for r in rows if (r["benchmark"], r["problem_id"]) in wanted]
    meta = {"run_id": args.run_id, "step": args.step, "model": args.model, "game": args.game,
            "estimator": args.estimator, "opponent": args.opponent, "seed": args.seed}
    results, bits = summarize(rows, meta)
    (args.out_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")

    for r in results:
        if r["metric"] == "accuracy":
            _log(f"{r['benchmark']:14s} accuracy {r['value']:.4f}  ({r['n_problems']} problems)")
    if args.print_summary:
        summary = json.dumps({"results": results, "correct": bits, "versions": config["versions"]})
        _log(f"SUMMARY_BEGIN\n{summary}\nSUMMARY_END")


if __name__ == "__main__":
    main()
