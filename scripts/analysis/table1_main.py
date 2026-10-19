"""Table 1: one table per dataset for the entropy-gated runs (2 readouts x 4 step lengths), plus the paired
Busemann-vs-radius comparison.

Runs: <results_dir>/chart_r1.0_K1_reroll_t{t}_{ideal|centre}_{chart}_{tag}.jsonl   baseline: <results_dir>/none_r0_K1_last_{tag}.jsonl
Usage (the paper's grading): python scripts/analysis/table1_main.py --tags c3 g8 --judge unified --safeguard [--results_dir outputs]
       python scripts/analysis/table1_main.py --tags oly --arm-tag oly2 --split data/ids/oly_split.json --judge unified --safeguard
"""
import argparse, glob, gzip, importlib, json, os, sys
from math import comb
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "hest", "third_party", "qwen_math_eval"))
RES = os.path.join(ROOT, "outputs")
P = importlib.import_module("parser")
G = importlib.import_module("grader")
ex = lambda t: P.strip_string(P.extract_answer(t, "math") or "") or None


from hest.evaluation.judge_safe import toolkit_equal as eq, uni_score  # noqa: E402  (graders in a killable worker: sympy can spin forever)
NAMES = {"c3": "MATH-500", "g8": "GSM8K", "oly": "OlympiadBench", "l8": "MATH-500 (Llama-3.1-8B)", "l8g": "GSM8K (Llama-3.1-8B)", "q7": "MATH-500 (Qwen2.5-Math-7B)", "q7g": "GSM8K (Qwen2.5-Math-7B)"}
TS = ["0.1", "0.25", "0.5", "1.0"]


def load(p):
    """result rows of a .jsonl file, or of its .jsonl.gz twin (the released result files are gzipped)"""
    if not os.path.exists(p) and os.path.exists(p + ".gz"):
        p = p + ".gz"
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return {json.loads(l)["idx"]: json.loads(l) for l in f if l.strip()}


def two_sided(up, dn):
    n = up + dn
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(max(up, dn), n + 1)) / 2 ** n)


JUDGE = "unified"    # "unified": the paper's protocol; "toolkit": Qwen2.5-Math toolkit regrade; "decode": the row's own decode-time flag


def correct(d, ids):
    if JUDGE == "decode":
        return np.array([bool(d[i]["ok"][0]) for i in ids], float)
    if JUDGE == "unified":                                   # our protocol (judge_unified): boxed / marker / restated / last number
        return np.array([bool(uni_score(d[i]["texts"][0], str(d[i]["gold"]),
                                        d[i]["preds"][0] if d[i].get("preds") else None)[0]) for i in ids], float)
    out = []
    for i in ids:
        t = d[i]["texts"][0]
        p = ex(t)
        out.append(bool(p) and eq(p, P.strip_string(str(d[i]["gold"])))) 
    return np.array(out, float)


SUFFIX = ""          # optional suffix of the reroll variant in the run name (reroll{SUFFIX}_t...)
SAFEGUARD = False    # runaway safeguard: a steered decode that hits the token cap is replaced by the greedy answer
CAP = 3072


def arm_path(tag, t, tgt):
    pat = os.path.join(RES, f"chart_r1.0_K1_reroll{SUFFIX}_t{t}_{tgt}_*_{tag}.jsonl")
    f = sorted(x[:-3] if x.endswith(".gz") else x for x in glob.glob(pat) + glob.glob(pat + ".gz"))
    return f[0] if f else None


def one(tag, restrict=None, arm_tag=None):
    arm_tag = arm_tag or tag                       # runs may carry their own tag (e.g. oly2) while the baseline is oly
    gp = os.path.join(RES, f"none_r0_K1_last_{tag}.jsonl")
    if not (os.path.exists(gp) or os.path.exists(gp + ".gz")):
        print(f"{NAMES.get(tag, tag)}: no baseline yet"); return None
    g = load(gp)
    arms = {}
    for tgt, lbl in (("ideal", "Busemann"), ("centre", "radius  ")):
        for t in TS:
            f = arm_path(arm_tag, t, tgt)
            if not f:
                continue
            d = load(f)
            need = len(restrict) if restrict is not None else len(g)   # runs on the split's test half are complete at its size
            if len(d) < need:                         # run still in progress: exclude, do not shrink the common set
                print(f"  skipping {lbl} t={t}: {len(d)}/{need} problems, still running")
                continue
            arms[(lbl, t)] = d
    if not arms:
        print(f"{NAMES.get(tag, tag)}: baseline {len(g)} problems, no runs yet"); return None
    ids = sorted(set(g).intersection(*[set(v) for v in arms.values()]))
    if restrict is not None:
        ids = [i for i in ids if i in restrict]
    gc = correct(g, ids)
    print(f"\n{NAMES.get(tag, tag)}: {len(ids)} problems common to {len(arms)} runs + greedy; "
          f"greedy {gc.mean():.4f}   (baseline file has {len(g)})")
    print(f"{'chart':10s} {'t':>5s} {'acc':>7s} {'vs greedy':>10s} {'fixed':>6s} {'broken':>7s} {'net':>5s} {'p':>6s} "
          f"{'changed':>8s} {'fires':>6s} {'len':>6s}" + (f" {'fallback':>8s}" if SAFEGUARD else ""))
    C = {}
    for (lbl, t), d in sorted(arms.items(), key=lambda kv: (kv[0][0], float(kv[0][1]))):
        a = correct(d, ids)
        fb = 0
        if SAFEGUARD:                                 # capped steered decode -> the greedy answer stands
            capped = np.array([d[i]["n_tokens"][0] >= CAP for i in ids])
            a = np.where(capped, gc, a); fb = int(capped.sum())
        C[(lbl, t)] = a
        up = int(((a == 1) & (gc == 0)).sum()); dn = int(((a == 0) & (gc == 1)).sum())
        ch = np.mean([d[i]["texts"][0] != g[i]["texts"][0] for i in ids])
        fr = np.mean([np.mean(d[i].get("fire_count", [0])) for i in ids])
        lr = np.mean([d[i]["n_tokens"][0] for i in ids]) / np.mean([g[i]["n_tokens"][0] for i in ids])
        print(f"{lbl:10s} {t:>5s} {a.mean():7.4f} {(a.mean()-gc.mean())*100:+9.2f}% {up:6d} {dn:7d} {up-dn:+5d} "
              f"{two_sided(up, dn):6.3f} {ch:8.0%} {fr:6.1f} {lr:6.2f}x" + (f" {fb:8d}" if SAFEGUARD else ""))
    # paired Busemann vs radius at matched t
    print(f"\n  paired Busemann vs radius (same t, same problems)")
    tot_u = tot_d = 0
    for t in TS:
        if ("Busemann", t) in C and ("radius  ", t) in C:
            b, r = C[("Busemann", t)], C[("radius  ", t)]
            u = int(((b == 1) & (r == 0)).sum()); dd = int(((b == 0) & (r == 1)).sum())
            tot_u += u; tot_d += dd
            print(f"    t={t:<5s} delta {b.mean()-r.mean():+.4f}   B-only {u:3d}  r-only {dd:3d}  p {two_sided(u, dd):.3f}")
    print(f"    pooled   B-only {tot_u:3d}  r-only {tot_d:3d}  p {two_sided(tot_u, tot_d):.4f}")
    best = max(C.items(), key=lambda kv: kv[1].mean())
    return dict(name=NAMES.get(tag, tag), n=len(ids), greedy=gc.mean(),
                best=(best[0][0].strip(), best[0][1], best[1].mean()),
                bus={t: C[("Busemann", t)].mean() for t in TS if ("Busemann", t) in C},
                rad={t: C[("radius  ", t)].mean() for t in TS if ("radius  ", t) in C})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="*", default=["c3", "g8", "oly"])
    ap.add_argument("--split", default=None, help="data/ids/oly_split.json -> report only its test half")
    ap.add_argument("--suffix", default="", help="suffix of the reroll variant in the run name (reroll<suffix>_t...)")
    ap.add_argument("--arm-tag", default=None, help="tag of the steered run files when it differs from the baseline tag")
    ap.add_argument("--ids", default=None, help="plain id-list json: restrict every dataset to these problems (pilots)")
    ap.add_argument("--judge", default="unified", choices=["unified", "toolkit", "decode"], help="unified = the paper's protocol (judge_unified.py); toolkit = Qwen2.5-Math toolkit regrade; decode = the row's own decode-time flag")
    ap.add_argument("--safeguard", action="store_true", help="runaway safeguard: steered decodes that hit the cap fall back to the greedy answer")
    ap.add_argument("--results_dir", default=None, help="directory of the result rows (default outputs/)")
    a = ap.parse_args()
    if a.results_dir:
        RES = a.results_dir if os.path.isabs(a.results_dir) else os.path.join(ROOT, a.results_dir)
    SUFFIX = a.suffix
    JUDGE = a.judge
    SAFEGUARD = a.safeguard
    R = set(json.load(open(a.split, encoding="utf-8"))["test"]) if a.split else None
    RI = set(json.load(open(a.ids, encoding="utf-8"))) if a.ids else None
    S = [s for s in (one(t, RI if RI is not None else (R if t == "oly" else None), a.arm_tag) for t in a.tags) if s]
    if len(S) > 1:
        print("\n\n=== three-dataset summary (Busemann row = accuracy at each dose, greedy in brackets) ===")
        print(f"{'dataset':16s} {'n':>5s} {'greedy':>8s} " + " ".join(f"{'t='+t:>8s}" for t in TS) + f" {'best':>22s}")
        for s in S:
            print(f"{s['name']:16s} {s['n']:5d} {s['greedy']:8.4f} " +
                  " ".join(f"{s['bus'].get(t, float('nan')):8.4f}" for t in TS) +
                  f"   Busemann t={s['best'][1]} {s['best'][2]:.4f}" if s['best'][0] == 'Busemann' else
                  f"{s['name']:16s} {s['n']:5d} {s['greedy']:8.4f} " +
                  " ".join(f"{s['bus'].get(t, float('nan')):8.4f}" for t in TS) +
                  f"   {s['best'][0]} t={s['best'][1]} {s['best'][2]:.4f}")
        print(f"\n{'dataset':16s} {'n':>5s} {'greedy':>8s} " + " ".join(f"{'t='+t:>8s}" for t in TS) + "   (radius)")
        for s in S:
            print(f"{s['name']:16s} {s['n']:5d} {s['greedy']:8.4f} " +
                  " ".join(f"{s['rad'].get(t, float('nan')):8.4f}" for t in TS))
    print("DS_TABLE_DONE")
