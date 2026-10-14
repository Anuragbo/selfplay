import pytest

from benchmark_stats import answer_kind, last_boxed


@pytest.mark.parametrize("text, expected", [
    (r"so the answer is \boxed{42}.", "42"),
    (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
    (r"first \boxed{1} then \boxed{2}", "2"),
    ("no box here", None),
    (r"\boxed without braces", None),
    (r"\boxed{unbalanced", None),
])
def test_last_boxed(text, expected):
    assert last_boxed(text) == expected


@pytest.mark.parametrize("answer, expected", [
    ("42", "integer"),
    ("-7", "integer"),
    ("1,000", "integer"),
    (27.0, "integer"),
    ("3.5", "decimal"),
    (r"\frac{1}{2}", "expression/other"),
    (None, "expression/other"),
])
def test_answer_kind(answer, expected):
    assert answer_kind(answer) == expected
