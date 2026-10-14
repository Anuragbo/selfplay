"""Profile the evaluation benchmarks: size, problem length and answer format."""

import argparse
import json
import re
import statistics
from collections import Counter
from collections.abc import Callable
from pathlib import Path

INTEGER = re.compile(r"^-?\d+$")
NUMBER = re.compile(r"^-?\d+(\.\d+)?$")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def last_boxed(text: str) -> str | None:
    """Content of the last brace-balanced \\boxed{...} in text, or None."""
    start = text.rfind("\\boxed")
    if start < 0:
        return None
    open_brace = text.find("{", start)
    if open_brace < 0:
        return None
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1:i]
    return None


def answer_kind(answer) -> str:
    """Classify an answer as 'integer', 'decimal' or 'expression/other'."""
    a = ("" if answer is None else str(answer)).strip().replace(",", "").replace("$", "")
    if INTEGER.match(a):
        return "integer"
    if NUMBER.match(a):
        return "integer" if float(a).is_integer() else "decimal"
    return "expression/other"


def _olympiad_answer(row: dict):
    ans = row["final_answer"]
    return ans[0] if isinstance(ans, list) else ans


# name -> (file stem, question field accessor, answer accessor)
BENCHMARKS: dict[str, tuple[str, Callable[[dict], str], Callable[[dict], object]]] = {
    "MATH-500": ("math500", lambda r: r["problem"], lambda r: r["answer"]),
    "OlympiadBench": ("olympiadbench", lambda r: r["question"], _olympiad_answer),
    "Minerva Math": ("minerva_math", lambda r: r["problem"], lambda r: last_boxed(r["solution"])),
    "AIME 2024": ("aime24", lambda r: r["problem"], lambda r: r["answer"]),
    "AMC 2023": ("amc23", lambda r: r["problem"], lambda r: r["answer"]),
    "GSM8K (not in evaluation suite)": ("gsm8k", lambda r: r["question"],
                                        lambda r: r["answer"].split("####")[-1].strip()),
}


def profile(raw_dir: Path) -> dict:
    stats = {}
    rows_by_name = {}
    for name, (stem, question, answer) in BENCHMARKS.items():
        rows = load_jsonl(raw_dir / f"{stem}.jsonl")
        rows_by_name[name] = rows
        answers = [answer(r) for r in rows]
        stats[name] = {
            "n": len(rows),
            "median_words": statistics.median(len(question(r).split()) for r in rows),
            "answer_kinds": dict(Counter(answer_kind(a) for a in answers)),
            "answer_missing": sum(a is None for a in answers),
        }

    math500 = rows_by_name["MATH-500"]
    stats["MATH-500"]["by_level"] = dict(sorted(Counter(str(r["level"]) for r in math500).items()))
    stats["MATH-500"]["by_subject"] = dict(Counter(r["subject"] for r in math500).most_common())

    olympiad = rows_by_name["OlympiadBench"]
    stats["OlympiadBench"]["answer_type"] = dict(Counter(r["answer_type"] for r in olympiad))
    stats["OlympiadBench"]["multiple_answer"] = sum(r["is_multiple_answer"] in (True, "True") for r in olympiad)
    stats["OlympiadBench"]["subfield"] = dict(Counter(r["subfield"] for r in olympiad))

    stats["Minerva Math"]["courses"] = len({r["type"] for r in rows_by_name["Minerva Math"]})
    stats["AMC 2023"]["answers_stored_as_float"] = sum(
        str(r["answer"]).endswith(".0") for r in rows_by_name["AMC 2023"])
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path, help="directory with the benchmark .jsonl files")
    parser.add_argument("output", type=Path, help="output JSON path")
    args = parser.parse_args()

    stats = profile(args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    for name, s in stats.items():
        print(f"{name:34s} n={s['n']:5d}  median words={s['median_words']:5.1f}  answers={s['answer_kinds']}")


if __name__ == "__main__":
    main()
