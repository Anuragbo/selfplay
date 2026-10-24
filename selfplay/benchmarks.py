"""Evaluation benchmarks: load data/raw/*.jsonl (from data/fetch_benchmarks.sh) into a common problem format."""

import json
from dataclasses import dataclass
from pathlib import Path

from selfplay.grading import last_boxed


@dataclass(frozen=True)
class Problem:
    benchmark: str
    problem_id: str
    question: str
    answer: str


def _olympiad_question(row: dict) -> str:
    context = row.get("context")
    return f"{context}\n{row['question']}" if context and context != "None" else row["question"]


# key -> (display name, file stem, id, question, answer)
BENCHMARKS = {
    "math500": ("MATH-500", "math500", lambda r: r["unique_id"], lambda r: r["problem"], lambda r: r["answer"]),
    "olympiadbench": ("OlympiadBench", "olympiadbench", lambda r: r["id"], _olympiad_question,
                      lambda r: r["final_answer"][0]),
    "minerva": ("Minerva Math", "minerva_math", lambda r: r["idx"], lambda r: r["problem"],
                lambda r: last_boxed(r["solution"])),
    "aime24": ("AIME 2024", "aime24", lambda r: r["id"], lambda r: r["problem"], lambda r: r["answer"]),
    "aime25": ("AIME 2025", "aime25", lambda r: r["id"], lambda r: r["problem"], lambda r: r["answer"]),
    "amc23": ("AMC 2023", "amc23", lambda r: r["id"], lambda r: r["problem"], lambda r: r["answer"]),
}
SUITE = list(BENCHMARKS)  # SPIRAL's math suite; GSM8K is deliberately absent


def load_benchmark(key: str, data_dir: Path) -> list[Problem]:
    if key not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {key!r}; choose from {SUITE}")
    name, stem, problem_id, question, answer = BENCHMARKS[key]
    lines = (data_dir / f"{stem}.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    problems = [Problem(name, str(problem_id(r)), question(r), str(answer(r))) for r in rows]
    if len({p.problem_id for p in problems}) != len(problems):
        raise ValueError(f"{key}: problem ids are not unique")
    return problems
