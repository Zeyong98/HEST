"""Unified grading protocol: one rule for every model, with or without \\boxed{}.

Stage 1 -- final-answer extraction, first hit wins, identical for every model:
  E1  the last non-empty \\boxed{...} in the response (balanced braces);
  E2  the expression after the last explicit marker ("the answer is", "final answer is", "Answer:"),
      up to the end of the sentence/line, math delimiters and trailing punctuation stripped;
  E3  the answer restated by the SAME model from its own solution (reformat prompt: "Answer: \\boxed{}");
      label-free, done once at decode time and stored as the row's `preds`;
  E4  the last number in the response (GSM8K-style fallback).
Stage 2 -- equivalence, first success wins, 20 s process guard (judge_safe), timeout = incorrect.
  Both sides are compared in two forms: raw (delimiters, spacing, \\text/\\mbox wrappers, "x=" prefixes removed) and
  normalised (Hendrycks-MATH / Qwen2.5-Math `strip_string`: units, \\left\\right, \\dfrac, ^\\circ, \\%, separators).
  L   lists / tuples / sets / intervals (top-level commas): same number of elements, element-wise equivalence,
      order-insensitive for sets and bare lists, order-sensitive for tuples/intervals, bracket types must match;
      a length mismatch is WRONG (no whole-string rescue).
  T1  exact match of raw or normalised strings (case-insensitive);
  T2  numeric: both sides parse as numbers (int / decimal / fraction; a base subscript "_8" is NOT a number);
      integers must match exactly, non-integers to a relative tolerance 1e-4; a factor 100 is allowed only when one
      side is written with a percent sign; both numeric and not equal is WRONG (no symbolic rescue);
  T3  symbolic (math_verify 0.9: latex2sympy2_extended + sympy) on the raw pair, then on the normalised pair;
  T4  symbolic fallback (Qwen2.5-Math `math_equal`) on scalars only.
Primitives: `strip_string` and `math_equal` of the Qwen2.5-Math evaluation toolkit (hest/third_party/qwen_math_eval,
itself derived from the Hendrycks MATH release), and math_verify.  The rule above (list / percent / integer /
base-subscript / raw-first handling) is this repository's own.
"""
import importlib, math, os, re, sys
TOOLKIT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "hest", "third_party", "qwen_math_eval")
if TOOLKIT not in sys.path:
    sys.path.insert(0, TOOLKIT)
_P = importlib.import_module("parser")
_G = importlib.import_module("grader")
strip_string = _P.strip_string

_MARK = re.compile(r"(?is)(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*|(?:^|\n)\s*\**answer\**\s*:\s*")
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


# ----------------------------------------------------------------------------------------------------- extraction
def extract_boxed(text):
    """E1: the last non-empty \\boxed{...} with balanced braces."""
    if not text:
        return None
    pos = len(text)
    while True:
        s = text.rfind("\\boxed{", 0, pos)
        if s == -1:
            return None
        depth, i = 1, s + 7
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            a = text[s + 7: i - 1].strip()
            if a:
                return a
        pos = s


def _clean_expr(a):
    a = a.strip()
    a = re.sub(r"^[\s:]*", "", a)
    a = a.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    a = a.replace("$", "").replace("**", "").strip()
    a = re.sub(r"[\.\s]+$", "", a)
    return a.strip()


def extract_marker(text):
    """E2: the expression after the last explicit final-answer marker, to the end of the sentence or line."""
    if not text:
        return None
    ms = list(_MARK.finditer(text))
    if not ms:
        return None
    rest = text[ms[-1].end():].split("\n")[0]
    m = re.search(r"(?<!\d)\.\s|\.$|(?<=\d)\.\s", rest)          # sentence end, but not a decimal point
    if m:
        rest = rest[: m.start()]
    a = _clean_expr(rest)
    return a or None


def extract_last_number(text):
    """E4"""
    if not text:
        return None
    ns = _NUM.findall(text.replace("$", ""))
    return ns[-1].replace(",", "") if ns else None


def extract_answer_unified(text, restated=None):
    """Returns (answer, stage) with stage in E1..E4 or None."""
    a = extract_boxed(text)
    if a:
        return a, "E1"
    a = extract_marker(text)
    if a:
        return a, "E2"
    if restated is not None and str(restated).strip():
        r = str(restated).strip()
        b = extract_boxed(r)
        return (b if b else _clean_expr(re.sub(r"(?i)^\s*answer\s*:\s*", "", r))), "E3"
    a = extract_last_number(text)
    if a:
        return a, "E4"
    return None, None


# ---------------------------------------------------------------------------------------------------- equivalence
_PREFIX = re.compile(r"^\s*[a-zA-Z](?:\([a-zA-Z]\))?\s*(?:=|\\in)\s*(?![=<>])")   # "x = 3", "f(x) = ...", "x \in [..]"
_UNIT_WORDS = {"inch", "inches", "in", "cm", "mm", "m", "km", "meter", "meters", "metre", "metres", "ft", "feet", "foot",
               "yard", "yards", "mile", "miles", "degree", "degrees", "deg", "dollar", "dollars", "cent", "cents", "unit",
               "units", "square", "sq", "cubic", "minute", "minutes", "min", "hour", "hours", "hr", "hrs", "second",
               "seconds", "sec", "day", "days", "week", "weeks", "month", "months", "year", "years", "mph", "kg", "g",
               "gram", "grams", "lb", "lbs", "pound", "pounds", "ounce", "ounces", "oz", "liter", "liters", "litre",
               "litres", "ml", "gallon", "gallons", "percent", "people", "students", "cups", "cup", "pieces", "piece",
               "points", "point", "times", "ways", "digits", "sides", "radians", "rad", "mph", "kph", "amps", "volts"}
_UNIT = re.compile(r"^(-?\d[\d.]*|-?\\frac\{-?\d+\}\{-?\d+\}|-?\d+/\d+)\s*([a-zA-Z][a-zA-Z ]*?)\s*(?:\^\s*\{?\d\}?)?\s*$")


def _raw(s):
    """light normalisation that keeps the mathematics: delimiters, wrappers, spacing, trailing punctuation."""
    s = str(s).strip()
    s = re.sub(r",\\!\s*(?=\d{3})", "", s)                                 # LaTeX thin-space thousands: 11,\! 111 -> 11111
    s = s.replace("\\$", "").replace("$", "").replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    s = re.sub(r"\\(?:text|mbox|textbf|mathrm|textit)\s*\{([^{}]*)\}", r" \1 ", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\^\s*\{?\\circ\}?|°", "", s)
    if re.fullmatch(r"\s*-?\d{1,3}(,\d{3})+(\.\d+)?\s*", s):                # a thousands-formatted number: 32,348 -> 32348
        s = s.replace(",", "")
    s = re.sub(r"\\pm\s*(\\frac\{[^{}]*\}\{[^{}]*\}|[\w.]+)", r"\1,-\1", s)  # \pm a -> a,-a
    s = re.sub(r"\s+(?:or|and)\s+", ",", s)                                 # "x or y" -> "x,y"
    s = _PREFIX.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[\.\s]+$", "", s)
    return s.strip()


def _elem(s):
    """per-element cleanup: drop 'x =' prefixes and trailing unit words (864 inches^2 -> 864)."""
    s = _PREFIX.sub("", s.strip()).strip()
    m = _UNIT.match(s)
    if m:
        words = m.group(2).split()
        known = all(w.lower() in _UNIT_WORDS for w in words)
        phrase = s[len(m.group(1)):][:1].isspace() and all(w.isalpha() and len(w) >= 2 for w in words)
        if known or phrase:                                             # "864 inches^2", "40 pairs of legs" -> the number
            return m.group(1)
    return s


def _is_number(s):
    s = str(s).strip().replace(",", "").replace("\\%", "").replace("%", "").replace("\\$", "")
    if not s or "_" in s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        pass
    return bool(re.fullmatch(r"-?\s*\\frac\{-?\d+\}\{-?\d+\}|-?\d+\s*/\s*-?\d+", s))


def _to_number(s):
    s = str(s).strip().replace(",", "").replace("\\%", "").replace("%", "").replace("\\$", "").replace(" ", "")
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        v = int(m.group(2)) / int(m.group(3))
        return -v if m.group(1) else v
    m = re.fullmatch(r"(-?\d+)/(-?\d+)", s)
    if m:
        return int(m.group(1)) / int(m.group(2))
    return float(s)


def _num_equal(g, p):
    """T2 rule; returns True / False, or None when not both numeric."""
    if not (_is_number(g) and _is_number(p)):
        return None
    gv, pv = _to_number(g), _to_number(p)
    pct = ("%" in str(g)) != ("%" in str(p))                       # exactly one side written as a percentage
    cands = [gv] + ([gv / 100, gv * 100] if pct else [])
    for c in cands:
        if float(c).is_integer() and float(pv).is_integer():
            if int(round(c)) == int(round(pv)):
                return True
        elif math.isclose(c, pv, rel_tol=1e-4, abs_tol=1e-9):
            return True
    return False


def _mv_equal(g, p):
    from math_verify import parse, verify
    def wrap(s):
        s = str(s).strip()
        return s if s.startswith("$") else f"${s}$"
    gg, pp = parse(wrap(g), parsing_timeout=None), parse(wrap(p), parsing_timeout=None)
    return bool(gg and pp and verify(gg, pp, timeout_seconds=None))


def _split_top(s):
    """split on commas outside {}, (), []"""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur); cur = ""
        else:
            cur += ch
    out.append(cur)
    return [x.strip() for x in out if x.strip() != ""]


def _outer_closes_at_end(s):
    """True when the bracket opening at s[0] is closed by the last character (so s is ONE bracketed group)."""
    depth = 0
    for i, ch in enumerate(s):
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
            if depth == 0:
                return i == len(s) - 1
    return False


def _container(s):
    """(kind, inner) with kind in set / tuple / interval / list / None"""
    s = s.strip()
    if s.startswith("\\{") and s.endswith("\\}") and _outer_closes_at_end(s[1:-1]):
        return "set", s[2:-2]
    if s.startswith("{") and s.endswith("}") and _outer_closes_at_end(s):
        return "set", s[1:-1]
    if s[:1] in "([" and s[-1:] in ")]" and _outer_closes_at_end(s):
        inner = s[1:-1]
        parts = _split_top(inner)
        if len(parts) == 2 and (s[0] == "[" or s[-1] == "]" or "infty" in inner):
            return "interval" + s[0] + s[-1], inner
        return "tuple", inner
    if "," in s and len(_split_top(s)) > 1:
        return "list", s
    return None, s


def _scalar_equiv(g, p):
    """T1 / T2 / T3 for one element (raw strings)."""
    rg, rp = _elem(_raw(g)), _elem(_raw(p))
    if rg.lower() == rp.lower() and rg != "":
        return True, "T1"
    ng, np_ = strip_string(rg), strip_string(rp)
    if ng.lower() == np_.lower() and ng != "":
        return True, "T1"
    ne = _num_equal(rg, rp)
    if ne is None:
        ne = _num_equal(ng, np_)
    if ne is True:
        return True, "T2"
    if ne is False:
        return False, "T2-mismatch"
    for a, b in ((rg, rp), (ng, np_)):
        try:
            if _mv_equal(a, b):
                return True, "T3"
        except Exception:
            pass
    return False, "no"


def equiv_unified(gold, pred):
    """Returns (correct, tier)."""
    if pred is None or gold is None:
        return False, "none"
    rg, rp = _raw(gold), _raw(pred)
    if not rp:
        return False, "empty"
    if rg.replace(" ", "").lower() == rp.replace(" ", "").lower():
        return True, "T1"
    if "\\cup" in rg or "\\cup" in rp:                                  # unions: unordered set of pieces
        pg, pp = [x.strip() for x in rg.split("\\cup")], [x.strip() for x in rp.split("\\cup")]
        if len(pg) != len(pp):
            return False, "L-length"
        left = list(pp)
        for a in pg:
            j = next((k for k, b in enumerate(left) if equiv_unified(a, b)[0]), None)
            if j is None:
                return False, "L-elem"
            left.pop(j)
        return True, "L-union"
    kg, ig = _container(rg)
    kp, ip = _container(rp)
    if kg or kp:                                                        # L: lists / tuples / sets / intervals
        if (kg or "list").startswith("interval") or (kp or "list").startswith("interval"):
            if kg != kp:
                return False, "L-bracket"
        eg, ep = _split_top(ig), _split_top(ip)
        ordered = (kg or kp) == "tuple" or str(kg or kp).startswith("interval")
        if ordered:                                                     # tuples / intervals: same length, in order
            if len(eg) != len(ep):
                return False, "L-length"
            if len(eg) > 1:
                oks = [equiv_unified(a, b)[0] for a, b in zip(eg, ep)]
                return (True, "L-ordered") if all(oks) else (False, "L-elem")
        elif len(eg) > 1 or len(ep) > 1:                                # sets / bare lists: mutual matching, duplicates ignored
            if not all(any(equiv_unified(a, b)[0] for b in ep) for a in eg):
                return False, "L-elem"
            if not all(any(equiv_unified(a, b)[0] for a in eg) for b in ep):
                return False, "L-elem"
            return True, "L-set"
        # one element inside brackets: compare the element (e.g. "(3)" vs "3") but keep interval brackets strict
    ok, tier = _scalar_equiv(gold, pred)
    if ok or tier == "T2-mismatch":
        return ok, tier
    try:                                                                # T4: Qwen toolkit fallback, scalars only
        if _G.math_equal(strip_string(rp), strip_string(rg), timeout=False):
            return True, "T4"
    except Exception:
        pass
    return False, "no"


def score(text, gold, restated=None):
    """Full protocol on one response: (correct, extracted answer, extraction stage, equivalence tier)."""
    a, st = extract_answer_unified(text, restated)
    ok, tier = equiv_unified(gold, a)
    return ok, a, st, tier


if __name__ == "__main__":
    tests = [("The final answer is 4.", "4", None, True), ("Therefore, 196 has 9 positive whole-number divisors.", "9", "9", True),
             ("so the answer is $\\frac{14}{3}$.", "\\frac{14}{3}", None, True), ("$\\boxed{q - (1 + p)}$", "p - q", None, False),
             ("Thus $\\boxed{\\left(3, \\frac{\\pi}{2}\\right)}$", "\\left( 3, \\frac{\\pi}{2} \\right)", None, True),
             ("Profit = -$10,000\n\nJosh actually lost $10,000 on the deal.", "70000", "-$10,000", False),
             ("The final answer is: $\\boxed{0.5}$", "\\frac{1}{2}", None, True), ("The answer is 50\\%.", "0.5", None, True),
             ("Answer: 18", "18", None, True), ("x = 3.5 is the value. The answer is 3.5. Done", "7/2", None, True),
             ("\\boxed{32349}", "\\$32,\\!348", None, False), ("\\boxed{1}", "100", None, False), ("\\boxed{2516}", "2516_8", None, True),
             ("\\boxed{-4, -5}", "-4", None, False), ("\\boxed{10, 11, 12, 13, 14}", "10,11,12,13,14,-2,-1,0,1,2", None, False),
             ("\\boxed{(-\\infty, 0)}", "(-\\infty, 0]", None, False), ("\\boxed{1, -2}", "-2,1", None, True),
             ("\\boxed{\\text{east}}", "\\text{east}", None, True), ("\\boxed{864}", "864 \\mbox{ inches}^2", None, True),
             ("\\boxed{f(x) = x \\text{ or } f(x) = -x}", "$f(x)=x,f(x)=-x$", None, True), ("\\boxed{\\pm 1, \\pm 7}", "1,-1,7,-7", None, True),
             ("\\boxed{\\frac{1}{x}}", "$f(x)=\\frac{1}{x}$", None, True), ("\\boxed{16.00}", "16", None, True),
             ("\\boxed{\\textbf{(B)}}", "\\text{(B)}", None, True), ("\\boxed{1, p^3, pq, p^8, p^2 q^2}", "1", None, False),
             ("\\boxed{36}", "36^\\circ", None, True), ("\\boxed{[-2, 7]}", "x \\in [-2,7]", None, True), ("\\boxed{3.14159}", "\\pi", None, False),
             ("\\boxed{11111111100}", "11,\\! 111,\\! 111,\\! 100", None, True), ("\\boxed{2n^2}", "$2 n$", None, False),
             ("\\boxed{(2, 12) \\cup (12, 102)}", "(2,12) \\cup (12,102)", None, True), ("\\boxed{(12,102) \\cup (2,12)}", "(2,12) \\cup (12,102)", None, True),
             ("\\boxed{5 \\text{ inches}}", "5", None, True), ("\\boxed{5x}", "5", None, False), ("\\boxed{1,2,3}", "1, 2, 3", None, True),
             ("\\boxed{(1, 0), (100, 8), (0.01, -8)}", "$(1,0),(\\frac{1}{100},-8),(100,8)$", None, True),
             ("\\boxed{-2}", "-2,-2", None, True), ("\\boxed{(2, 6), (-6, 6)}", "$(-6,6)$, $(2,6)$", None, True),
             ("The answer is 40 pairs of legs.", "40", None, True), ("\\boxed{85.71}", "$\\frac{600}{7}$", None, True),
             ("\\boxed{2 n}", "2n^2", None, False), ("\\boxed{(1,2)}", "(2,1)", None, False), ("\\boxed{\\{1,2\\}}", "\\{2,1\\}", None, True)]
    bad = 0
    for t, g, r, want in tests:
        got = score(t, g, r)
        flag = "" if got[0] == want else "   <-- MISMATCH"
        bad += got[0] != want
        print(f"{got}  gold {g!r}{flag}")
    print(f"{len(tests) - bad}/{len(tests)} as intended")
