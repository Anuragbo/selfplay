import json

import pytest

from selfplay.benchmarks import SUITE, Problem, load_benchmark
from selfplay.evaluate import PROMPTS, evaluate, summarize

CONTRACT = {"run_id", "step", "model", "game", "estimator", "opponent", "seed", "benchmark", "metric", "value",
            "n_samples", "timestamp"}
META = {"run_id": "t", "step": 0, "model": "m", "game": "none", "estimator": "none", "opponent": "none", "seed": 0}

PROBLEMS = [Problem("A", str(i), f"What is {i} + 1?", str(i + 1)) for i in range(5)] + \
           [Problem("B", "x", "What is 1/2?", r"\frac{1}{2}")]


def fake_generate(prompts, n):
    """Correct on even numbers and on B; no answer on 3; wrong otherwise."""
    out = []
    for prompt in prompts:
        q = prompt.split("What is ")[1].split("?")[0]
        if q == "1/2":
            text = r"so \boxed{0.5}"
        elif int(q.split()[0]) == 3:
            text = "I don't know"
        else:
            i = int(q.split()[0])
            text = rf"\boxed{{{i + 1 if i % 2 == 0 else i}}}"
        out.append([(text, 10, "stop")] * n)
    return out


def test_prompt_contains_question_and_box_instruction():
    p = PROMPTS["qwen3"].format(question="Q?")
    assert "Q?" in p and r"\boxed{}" in p and p.endswith("<|im_start|>assistant\n")


def test_evaluate_and_summarize(tmp_path):
    rows = evaluate(PROBLEMS, fake_generate, PROMPTS["qwen3"], 2, tmp_path, chunk_size=4, log=lambda s: None)
    assert len(rows) == 12
    results, bits = summarize(rows, META)
    assert all(CONTRACT <= set(r) for r in results)
    acc = {r["benchmark"]: r["value"] for r in results if r["metric"] == "accuracy"}
    assert acc == {"A": pytest.approx(3 / 5), "B": 1.0, "pooled": pytest.approx(4 / 6)}
    no_answer = {r["benchmark"]: r["value"] for r in results if r["metric"] == "no_answer_rate"}
    assert no_answer["A"] == pytest.approx(1 / 5)
    assert bits["A"] == "11,00,11,00,11" and bits["B"] == "11"  # ids sorted as strings: 0,1,2,3,4


def test_evaluate_resumes_without_regenerating(tmp_path):
    calls = []

    def counting(prompts, n):
        calls.append(len(prompts))
        return fake_generate(prompts, n)

    evaluate(PROBLEMS[:3], counting, PROMPTS["qwen3"], 1, tmp_path, log=lambda s: None)
    rows = evaluate(PROBLEMS, counting, PROMPTS["qwen3"], 1, tmp_path, log=lambda s: None)
    assert calls == [3, 3] and len(rows) == 6
    completions = (tmp_path / "completions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(completions) == 6 and "completion" in json.loads(completions[0])


@pytest.fixture
def data_dir(request):
    path = request.config.rootpath / "data" / "raw"
    if not (path / "math500.jsonl").exists():
        pytest.skip("benchmarks not downloaded (run data/fetch_benchmarks.sh)")
    return path


EXPECTED_SIZES = {"math500": 500, "olympiadbench": 675, "minerva": 272, "aime24": 30, "aime25": 30, "amc23": 40}


@pytest.mark.parametrize("key", SUITE)
def test_benchmarks_load_with_answers(key, data_dir):
    problems = load_benchmark(key, data_dir)
    assert len(problems) == EXPECTED_SIZES[key]
    assert all(p.question and p.answer and p.answer != "None" for p in problems)
