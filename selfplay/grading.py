"""Answer extraction and grading for the math benchmarks.

The model's answer is the content of the last brace-balanced \\boxed{...} in its completion; a completion without
one is graded wrong and reported as unextracted. Correctness is decided by Math-Verify (symbolic equivalence via
SymPy: 1/2 = 0.5 = \\dfrac{1}{2}, intervals, sets, equations, units and percentages). If Math-Verify is not
installed, cannot parse either side, or fails, a fallback compares normalized strings and numeric values, and the
result records which method decided.
"""

import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Literal

# Math-Verify times out via signals on POSIX but via subprocesses elsewhere, which fail under Windows test runners.
_TIMEOUT = 5 if os.name == "posix" else None


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


_STRIP = [r"\left", r"\right", r"\!", r"\,", r"\;", r"\$", "$", r"\%", "%", "^\\circ", "^{\\circ}", " "]
_TEXT = re.compile(r"\\(?:text|textbf|mathrm|mbox)\{([^{}]*)\}")
_FRAC = re.compile(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}")


def normalize(answer: str) -> str:
    """Light normalization for the fallback string comparison."""
    s = _TEXT.sub(r"\1", str(answer).strip())
    for token in _STRIP:
        s = s.replace(token, "")
    s = _FRAC.sub(r"(\1)/(\2)", s).replace("\\dfrac", "\\frac").rstrip(".")
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s.lower()


def _as_number(s: str) -> float | None:
    s = s.replace(",", "")
    frac = re.fullmatch(r"\(?(-?\d+(?:\.\d+)?)\)?/\(?(-?\d+(?:\.\d+)?)\)?", s)
    try:
        return float(frac.group(1)) / float(frac.group(2)) if frac else float(s)
    except (ValueError, ZeroDivisionError):
        return None


def fallback_equal(pred: str, gold: str) -> bool:
    p, g = normalize(pred), normalize(gold)
    if p == g:
        return True
    pn, gn = _as_number(p), _as_number(g)
    return pn is not None and gn is not None and math.isclose(pn, gn, rel_tol=1e-6, abs_tol=1e-9)


def _math_verify(pred: str, gold: str) -> bool | None:
    """True/False from Math-Verify, or None if it is unavailable or cannot parse an answer."""
    try:
        from math_verify import parse, verify
    except ImportError:
        return None
    try:
        # OlympiadBench answers carry their own $...$ delimiters; others are bare LaTeX.
        gold_parsed = parse(gold if "$" in gold else f"${gold}$", parsing_timeout=_TIMEOUT)
        pred_parsed = parse(f"\\boxed{{{pred}}}", parsing_timeout=_TIMEOUT)
        if not gold_parsed or not pred_parsed:
            return None
        return bool(verify(gold_parsed, pred_parsed, timeout_seconds=_TIMEOUT))
    except Exception:
        return None


@dataclass
class Grade:
    extracted: str | None
    correct: bool
    method: Literal["math_verify", "fallback", "no_answer"]

    def to_dict(self) -> dict:
        return asdict(self)


def grade(completion: str, gold: str | int | float) -> Grade:
    """Grade one completion against the reference answer."""
    pred = last_boxed(completion)
    if pred is None or not pred.strip():
        return Grade(None, False, "no_answer")
    gold = str(gold)
    verdict = _math_verify(pred, gold)
    if verdict is True:
        return Grade(pred, True, "math_verify")
    if fallback_equal(pred, gold):
        return Grade(pred, True, "fallback")
    return Grade(pred, False, "math_verify" if verdict is False else "fallback")
