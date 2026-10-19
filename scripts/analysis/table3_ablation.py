"""Ablation table (Table 3 and the appendix ablations): Qwen2.5-Math-1.5B-Instruct, MATH-500; baseline = Table 1's
Busemann t=0.5, theta 1.0, layer 19 (<results_dir>/chart_r1.0_K1_reroll_t0.5_ideal_busemann_L19_c3.jsonl), greedy .7420.
Per run: accuracy with the runaway safeguard (unified judge; capped steered decodes report the greedy answer) and its
change vs greedy, fixed/broken, cap hits, accuracy WITHOUT the safeguard, steered-token share (fire_count / n_tokens;
per-problem median and pooled), fires per problem (median), |dh|/|h| (median over problems of the per-solution mean
logged by the chart field; the fixed-direction run logs its own per-fire values), solution length (median tokens).
Judgements are cached in outputs/judge_cache_unified.json (file@mtime, shared with appendix_regimes.py).
Writes outputs/tables/TABLE_ABL.md.  Main-guarded (judge_safe spawns a worker).  Usage: python scripts/analysis/table3_ablation.py [--results_dir outputs]"""
import glob
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "outputs")
CAP = 3072
CACHE_P = os.path.join(ROOT, "outputs", "judge_cache_unified.json")
GREEDY = "none_r0_K1_last_c3.jsonl"
BASE = "chart_r1.0_K1_reroll_t0.5_ideal_busemann_L19_c3.jsonl"
EVERY = "chart_r1.0_K1_allgen_t0.5_ideal_busemann_L19_abl.jsonl"
ARMS = [  # section, label, file (glob allowed)
    ("main", "HEST: Busemann t=0.5, θ=1.0, L19 (Table 1)", BASE),
    ("main", "#1 every token (θ = 0, no gate)", EVERY),
    ("main", "#2 fixed direction (CAA unit vector at gate hits, |Δh| = rel·|h|)", "vecgate_r0_K1_reroll_rel*_q15_math_abl.jsonl"),
    ("main", "#3 step boundaries (fire on tokens containing a double newline)", "chart_r1.0_K1_reroll_bnd_t0.5_ideal_busemann_L19_abl.jsonl"),
    ("main", "#2b reverse direction (same gate, geodesic step AWAY from the ideal point, t = -0.5)", "chart_r1.0_K1_reroll_t-0.5_ideal_busemann_L19_abl.jsonl"),
    ("thr", "θ = 0 (= #1)", EVERY),
    ("thr", "θ = ln 2 = 0.6931", "chart_r1.0_K1_rerolle0.6931_t0.5_ideal_busemann_L19_abl.jsonl"),
    ("thr", "θ = 1.0 (= HEST)", BASE),
    ("thr", "θ = ln 4 = 1.3863", "chart_r1.0_K1_rerolle1.3863_t0.5_ideal_busemann_L19_abl.jsonl"),
    ("thr", "θ = ln 8 = 2.0794", "chart_r1.0_K1_rerolle2.0794_t0.5_ideal_busemann_L19_abl.jsonl"),
    ("layer", "layer 9 (probe entchart_busemann_all_L9, same recipe)", "chart_r1.0_K1_reroll_t0.5_ideal_busemann_all_L9_abl.jsonl"),
    ("layer", "layer 19 (= HEST)", BASE),
    ("layer", "layer 25 (probe entchart_busemann_all_L25, same recipe)", "chart_r1.0_K1_reroll_t0.5_ideal_busemann_all_L25_abl.jsonl"),
]
HDR = ("| run | acc, safeguard (Δ vs greedy) | fixed/broken | cap hits | acc, no safeguard | steered tokens: median / pooled | "
       "fires per problem (median) | ‖Δh‖/‖h‖ median | length median (greedy 470) |")
SEP = "|---|---|---|---|---|---|---|---|---|"


def load(p):
    """result rows of a .jsonl file, or of its .jsonl.gz twin (the released result files are gzipped)"""
    if not os.path.exists(p) and os.path.exists(p + ".gz"):
        p = p + ".gz"
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return {json.loads(l)["idx"]: json.loads(l) for l in f if l.strip()}


def rel_of(rows, fallback):
    v = [r["vec"]["rel_med"] for r in rows if r.get("vec")]
    if v:
        return f"{np.median(v):.4f} (fixed by construction; realised bf16 {np.median([r['vec']['rel_real_med'] for r in rows if r.get('vec')]):.4f})"
    f = [r["field"]["rho"] for r in rows if r.get("field")]
    if f:
        return f"{np.median(f):.4f}"
    return fallback


def main():
    global R
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--results_dir", default=None, help="directory of the result rows (default outputs/)")
    a = ap.parse_args()
    if a.results_dir:
        R = a.results_dir if os.path.isabs(a.results_dir) else os.path.join(ROOT, a.results_dir)
    out_dir = os.path.join(ROOT, "outputs", "tables"); os.makedirs(out_dir, exist_ok=True)
    from hest.evaluation.judge_safe import uni_score, shutdown
    cache = json.load(open(CACHE_P, encoding="utf-8")) if os.path.exists(CACHE_P) else {}

    def ok(fname, r):
        fname = fname if os.path.exists(fname) else fname + ".gz"
        key = f"{os.path.basename(fname)}@{int(os.path.getmtime(fname))}"
        c = cache.setdefault(key, {})
        k = str(r["idx"])
        if k not in c:
            c[k] = bool(uni_score(r["texts"][0], str(r["gold"]), r["preds"][0] if r.get("preds") else None)[0])
        return c[k]

    gp = os.path.join(R, GREEDY)
    g = load(gp); ids = sorted(g)
    gc = np.array([ok(gp, g[i]) for i in ids], float)
    glen = np.median([g[i]["n_tokens"][0] for i in ids])
    fb_rel = "n/a"
    dc = os.path.join(R, "abl_disp_check.txt")
    if not os.path.exists(dc):
        dc = os.path.join(ROOT, "outputs", "abl_disp_check.txt")
    if os.path.exists(dc):
        for line in open(dc, encoding="utf-8"):
            if line.startswith("L19") and "hesitation" in line:
                fb_rel = line.split("median")[1].split("(")[0].strip() + " (test hesitation states, abl_disp_check.py)"
    rows_md = {"main": [], "thr": [], "layer": []}
    for sec, label, pat in ARMS:
        fs = sorted(x[:-3] if x.endswith(".gz") else x for x in glob.glob(os.path.join(R, pat)) + glob.glob(os.path.join(R, pat + ".gz")))
        if not fs:
            rows_md[sec].append(f"| {label} | (not run) | | | | | | | |"); continue
        f = fs[-1]; d = load(f)
        if any(i not in d for i in ids):
            rows_md[sec].append(f"| {label} | ({len(d)}/{len(ids)} running) | | | | | | | |"); continue
        raw = np.array([ok(f, d[i]) for i in ids], float)
        capped = np.array([d[i]["n_tokens"][0] >= CAP for i in ids])
        sg = np.where(capped, gc, raw)
        up = int(((sg == 1) & (gc == 0)).sum()); dn = int(((sg == 0) & (gc == 1)).sum())
        fc = np.array([d[i]["fire_count"][0] if d[i].get("fire_count") else 0 for i in ids], float)
        nt = np.array([d[i]["n_tokens"][0] for i in ids], float)
        share = fc / np.maximum(nt, 1)
        rel = rel_of([d[i] for i in ids], fb_rel)
        rows_md[sec].append(f"| {label} | {sg.mean():.4f} ({(sg.mean() - gc.mean()) * 100:+.2f}) | {up}/{dn} | {int(capped.sum())} | "
                            f"{raw.mean():.4f} ({(raw.mean() - gc.mean()) * 100:+.2f}) | {np.median(share):.1%} / {fc.sum() / nt.sum():.1%} | "
                            f"{np.median(fc):.0f} | {rel} | {np.median(nt):.0f} |")
    # GSM8K counterparts of the direction ablations: greedy g8, Busemann t=0.5 (Table 1), fixed CAA direction, reverse direction
    gsm_rows = []
    gp8 = os.path.join(R, "none_r0_K1_last_g8.jsonl")
    if os.path.exists(gp8) or os.path.exists(gp8 + ".gz"):
        g8 = load(gp8); ids8 = sorted(g8); gc8 = np.array([ok(gp8, g8[i]) for i in ids8], float)
        for label, fn in (("HEST: Busemann t=0.5, θ=1.0, L19 (Table 1, GSM8K)", "chart_r1.0_K1_reroll_t0.5_ideal_busemann_gsm_L19_g8.jsonl"),
                          ("#2 fixed direction on GSM8K (caa_q15_gsm unit vector at gate hits, 0.060|h|)", "vecgate_r0_K1_reroll_rel0.06_q15_gsm_abl.jsonl"),
                          ("#2b reverse direction on GSM8K (same gate, geodesic step AWAY from the ideal point, t = -0.5)", "chart_r1.0_K1_reroll_t-0.5_ideal_busemann_gsm_L19_abl.jsonl")):
            f8 = os.path.join(R, fn)
            if not (os.path.exists(f8) or os.path.exists(f8 + ".gz")):
                gsm_rows.append(f"| {label} | (not run) | | | | | | | |"); continue
            d8 = load(f8)
            if any(i not in d8 for i in ids8):
                gsm_rows.append(f"| {label} | ({len(d8)}/{len(ids8)} running) | | | | | | | |"); continue
            raw8 = np.array([ok(f8, d8[i]) for i in ids8], float)
            cap8 = np.array([d8[i]["n_tokens"][0] >= CAP for i in ids8]); sg8 = np.where(cap8, gc8, raw8)
            up8 = int(((sg8 == 1) & (gc8 == 0)).sum()); dn8 = int(((sg8 == 0) & (gc8 == 1)).sum())
            fc8 = np.array([d8[i]["fire_count"][0] if d8[i].get("fire_count") else 0 for i in ids8], float)
            nt8 = np.array([d8[i]["n_tokens"][0] for i in ids8], float)
            gsm_rows.append(f"| {label} | {sg8.mean():.4f} ({(sg8.mean() - gc8.mean()) * 100:+.2f}) | {up8}/{dn8} | {int(cap8.sum())} | "
                            f"{raw8.mean():.4f} ({(raw8.mean() - gc8.mean()) * 100:+.2f}) | {np.median(fc8 / np.maximum(nt8, 1)):.1%} / {fc8.sum() / nt8.sum():.1%} | "
                            f"{np.median(fc8):.0f} | {rel_of([d8[i] for i in ids8], 'n/a')} | {np.median(nt8):.0f} |")
        gsm_hdr = [f"## Robustness: fixed direction on GSM8K (greedy {gc8.mean():.4f}; same protocol; the GSM8K geodesic step has median |dh|/|h| .058 at t=0.5 on the training states)",
                   HDR.replace("(greedy 470)", f"(greedy {np.median([g8[i]['n_tokens'][0] for i in ids8]):.0f})"), SEP, *gsm_rows, ""]
    else:
        gsm_hdr = []
    abl0 = os.path.join(R, "ABL0_boundary_stats.txt")
    abl0_txt = open(abl0, encoding="utf-8").read().strip() if os.path.exists(abl0) else "(abl0_boundary_stats.py not run)"
    out = ["# Ablations -- Qwen2.5-Math-1.5B-Instruct, MATH-500",
           f"Baseline: Table 1's Busemann t=0.5, θ=1.0, layer 19; greedy {gc.mean():.4f} (unified judge). Every run changes one component. "
           "Grading: unified judge + runaway safeguard (a steered decode that hits the 3072-token cap reports the greedy answer); the "
           "no-safeguard accuracy is also given. Steered-token share = fires / generated tokens per problem. ‖Δh‖/‖h‖ = median over problems "
           "of the per-solution mean relative step logged by the chart field (the Table 1 file predates that log: its value is the median at "
           "the test-set hesitation states, abl_disp_check.py).", "",
           "## Table 3 (main text): gate, direction, trigger", HDR, SEP, *rows_md["main"], "",
           "## Appendix M1: gate threshold θ (steered-token share rises as θ falls)", HDR, SEP, *rows_md["thr"], "",
           "## Appendix M2: steering layer (same probe recipe: 100 MATH-train problems, train_ent_chart defaults, seed 42)",
           HDR, SEP, *rows_md["layer"], "",
           *gsm_hdr,
           "## Appendix M0: entropy-gate positions vs reasoning-step boundaries (CPU, greedy solutions)", "```", abl0_txt, "```"]
    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(out_dir, "TABLE_ABL.md"), "w", encoding="utf-8") as fo:
        fo.write(txt + "\n")
    os.makedirs(os.path.dirname(CACHE_P), exist_ok=True)
    json.dump(cache, open(CACHE_P, "w", encoding="utf-8"))
    shutdown()
    print("ABL_TABLE_DONE")


if __name__ == "__main__":
    main()
