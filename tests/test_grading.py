import pytest

from selfplay.grading import fallback_equal, grade, last_boxed, normalize


@pytest.mark.parametrize("text, expected", [
    (r"so the answer is \boxed{42}.", "42"),
    (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
    (r"first \boxed{1} then \boxed{2}", "2"),
    ("no box here", None),
    (r"\boxed{unbalanced", None),
])
def test_last_boxed(text, expected):
    assert last_boxed(text) == expected


@pytest.mark.parametrize("pred, gold", [
    ("27", "27.0"),             # AMC 2023 answers are stored as floats
    (r"\dfrac{1}{2}", "0.5"),
    ("10", r"10\%"),
    (r"\text{Evelyn}", "Evelyn"),
    ("18.90", r"\$18.90"),
    ("5", r"5^\circ"),
])
def test_fallback_equal(pred, gold):
    assert fallback_equal(pred, gold)


def test_fallback_rejects_different_values():
    assert not fallback_equal("3", "4")
    assert not fallback_equal("x+1", "x+2")
    assert normalize(r"\left( 3, 4 \right)") == "(3,4)"


def test_missing_box_is_wrong():
    g = grade("The answer is 42.", "42")
    assert (g.extracted, g.correct, g.method) == (None, False, "no_answer")
    assert grade(r"\boxed{}", "42").method == "no_answer"


math_verify = pytest.importorskip("math_verify")


@pytest.mark.parametrize("completion, gold", [
    (r"\boxed{0.5}", r"\frac{1}{2}"),                              # MATH-500
    (r"\boxed{\dfrac{1}{2}}", r"\frac12"),
    (r"\boxed{\left( 3, \frac{\pi}{2} \right)}", r"\left( 3, \frac{\pi}{2} \right)"),
    (r"\boxed{2x + x^2}", "x^2+2x"),
    (r"\boxed{\sqrt{8}}", r"2\sqrt{2}"),
    (r"\boxed{[2,5)}", r"x \in [2, 5)"),                           # OlympiadBench intervals
    (r"\boxed{2, 1}", "1,2"),                                      # unordered multi-answer
    (r"\boxed{\frac{41}{12}}", r"$\frac{41}{12}$"),                # OlympiadBench keeps $ delimiters
    (r"\boxed{0.0042}", r"4.2 \times 10^{-3}"),                    # Minerva scientific notation
    (r"\boxed{27}", "27.0"),                                       # AMC floats
    (r"\boxed{70}", 70),                                           # AIME integers
])
def test_equivalent_answers_are_correct(completion, gold):
    assert grade(completion, gold).correct


@pytest.mark.parametrize("completion, gold", [
    (r"\boxed{4}", "3"),
    (r"\boxed{\frac{1}{3}}", "0.3"),
    (r"\boxed{[2,5]}", "[2,5)"),
    (r"\boxed{x^2-2x}", "x^2+2x"),
])
def test_different_answers_are_wrong(completion, gold):
    g = grade(completion, gold)
    assert not g.correct and g.method == "math_verify"


def test_last_box_counts():
    assert grade(r"maybe \boxed{3}, no: \boxed{4}", "4").correct
    assert not grade(r"maybe \boxed{4}, no: \boxed{3}", "4").correct
