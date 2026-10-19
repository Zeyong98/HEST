"""Grader calls in a separate process with a hard wall-clock limit.

Why: math_verify / sympy can spin forever on a pathological answer (e.g. a power tower 10^{10^{10^{10}}} makes the
big-integer arithmetic run for ages), the library's own timeouts are disabled on Windows, and the spin happens in C
code that holds the GIL, so a thread-based guard never wakes up.  A worker process can be killed.

    from hest.evaluation.judge_safe import equiv, extract_bbox, toolkit_equal, uni_score
    equiv(gold, pred)          judge_gen.equiv          (math_verify, then the unified protocol)
    extract_bbox(text)         judge_gen.extract_bbox   (last balanced \\boxed{...})
    toolkit_equal(a, b)        Qwen2.5-Math toolkit grader.math_equal(a, b, timeout=False)
    uni_score(text, gold, restated)   judge_unified.score (the paper's grading)
A call that exceeds TIMEOUT seconds returns the failure value (False / None), the worker is killed and a fresh one
is started for the next call.  The worker is persistent, so the import cost is paid once.
"""
import multiprocessing as mp
import os
import sys

TIMEOUT = float(os.environ.get("JUDGE_TIMEOUT", "20"))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLKIT = os.path.join(ROOT, "hest", "third_party", "qwen_math_eval")


def _worker(conn):
    fns = {}
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg is None:
            return
        name, args = msg
        try:
            if name not in fns:
                if ROOT not in sys.path:
                    sys.path.insert(0, ROOT)
                if name in ("equiv", "extract_bbox"):
                    from hest.evaluation import judge_gen
                    fns["equiv"] = judge_gen.equiv
                    fns["extract_bbox"] = judge_gen.extract_bbox
                elif name == "toolkit_equal":
                    if TOOLKIT not in sys.path:
                        sys.path.insert(0, TOOLKIT)
                    import importlib
                    G = importlib.import_module("grader")
                    fns["toolkit_equal"] = lambda a, b: bool(G.math_equal(a, b, timeout=False))
                elif name in ("uni_equal", "uni_score"):
                    from hest.evaluation import judge_unified
                    fns["uni_equal"] = judge_unified.equiv_unified      # (gold, pred) -> (bool, tier)
                    fns["uni_score"] = judge_unified.score              # (text, gold, restated) -> (ok, ans, stage, tier)
            conn.send(("ok", fns[name](*args)))
        except Exception as e:  # noqa: BLE001
            conn.send(("err", repr(e)))


class _Judge:
    def __init__(self):
        self.proc = self.conn = None
        self.timeouts = 0

    def _start(self):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker, args=(child,), daemon=True)
        self.proc.start()
        child.close()

    def _stop(self):
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:  # noqa: BLE001
                pass
            self.proc = self.conn = None

    def call(self, name, args, fail, timeout=None):
        timeout = TIMEOUT if timeout is None else timeout
        if self.proc is None or not self.proc.is_alive():
            self._stop(); self._start()
        try:
            self.conn.send((name, args))
            if not self.conn.poll(timeout):
                self.timeouts += 1
                print(f"[judge_safe] {name}{tuple(str(a)[:60] for a in args)} exceeded {timeout:.0f}s -> {fail!r}; "
                      f"worker restarted", flush=True)
                self._stop()
                return fail
            status, val = self.conn.recv()
            return val if status == "ok" else fail
        except (EOFError, BrokenPipeError, OSError):
            self._stop()
            return fail


_J = _Judge()


def equiv(gold, pred):
    if pred is None or gold is None:
        return False
    return bool(_J.call("equiv", (gold, pred), False))


def extract_bbox(text):
    return _J.call("extract_bbox", (text,), None)


def toolkit_equal(a, b):
    return bool(_J.call("toolkit_equal", (a, b), False))


def uni_equal(gold, pred):
    """our protocol, equivalence only: (correct, tier); timeout -> (False, 'timeout')"""
    return _J.call("uni_equal", (gold, pred), (False, "timeout"))


def uni_score(text, gold, restated=None):
    """our protocol, extraction + equivalence: (correct, answer, stage, tier); timeout -> (False, None, None, 'timeout')"""
    return _J.call("uni_score", (text, gold, restated), (False, None, None, "timeout"))


def shutdown():
    _J._stop()


if __name__ == "__main__":
    import time
    t0 = time.time(); print("equiv('9','9') =", equiv("9", "9"), f"{time.time()-t0:.1f}s (includes worker start)")
    t0 = time.time(); print("equiv('9','10^{10}') =", equiv("9", "10^{10}"), f"{time.time()-t0:.1f}s")
    t0 = time.time(); print("extract_bbox =", extract_bbox("so the answer is \boxed{10^{10}}."), f"{time.time()-t0:.1f}s")
    t0 = time.time(); print("pathological:", equiv("9", "10^{10^{10^{10}}}"), f"{time.time()-t0:.1f}s")
    t0 = time.time(); print("after restart equiv('9','9') =", equiv("9", "9"), f"{time.time()-t0:.1f}s")
    t0 = time.time(); print("toolkit_equal('1/2','0.5') =", toolkit_equal("1/2", "0.5"), f"{time.time()-t0:.1f}s")
    shutdown(); print("JUDGE_SAFE_OK")
