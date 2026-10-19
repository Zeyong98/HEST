"""Decode-time answer extraction and equivalence (run inside the judge_safe worker process).

extract_bbox(text)   the last \\boxed{...} in the text with balanced braces (None if absent)
equiv(gold, pred)    math_verify (latex2sympy2 + sympy) first, then the unified protocol of judge_unified.py

Every number reported in the paper is re-graded offline by judge_unified.py; the decode-time flag stored in each
result row is informational only."""
import os
import sys

from math_verify import parse, verify

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hest.evaluation import judge_unified  # noqa: E402


def extract_bbox(response):
    if not response:
        return None
    start_idx = response.rfind("\\boxed{")
    if start_idx == -1:
        return None
    brace_count = 1
    pos = start_idx + 7
    while pos < len(response) and brace_count > 0:
        if response[pos] == "{":
            brace_count += 1
        elif response[pos] == "}":
            brace_count -= 1
        pos += 1
    if brace_count == 0:
        return response[start_idx + 7: pos - 1].strip()
    return None


def _wrap(s):
    s = str(s).strip()
    return s if s.startswith("$") else f"${s}$"


def equiv(gold, pred):
    if pred is None or gold is None:
        return False
    try:
        g, p = parse(_wrap(gold), parsing_timeout=None), parse(_wrap(pred), parsing_timeout=None)
        if g and p and verify(g, p, timeout_seconds=None):
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        return bool(judge_unified.equiv_unified(str(gold), str(pred))[0])
    except Exception:  # noqa: BLE001
        return False
