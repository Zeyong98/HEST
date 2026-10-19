"""Geometry ablation (Table 3(c) and Table 16, Appendix N): the Euclidean probes (kappa = 0) against the hyperbolic
probes of HEST, Qwen2.5-Math-1.5B-Instruct, MATH-500 and GSM8K.
Per run: accuracy with the runaway safeguard (unified judge; capped steered decodes report the greedy answer) and its
change vs greedy, fixed/broken, cap hits, accuracy WITHOUT the safeguard, steered-token share (per-problem median and
pooled), fires per problem (median), |dh|/|h| (median over problems of the per-solution mean logged by the probe field),
solution length (median tokens), changed solutions, fix / break rates among the changed solutions, and the paired counts
against the hyperbolic runs.  The held-out quality of the probes (pair accuracy, Spearman rho, curvature) is read from
the probe files in data/checkpoints/ when they are present.
Runs: configs/ablation/euclid_*_math500.json with --tag _c3 and configs/ablation/euclid_*_gsm8k.json with --data gsm8k
--tag _g8, next to the greedy and HEST rows of Table 1 (same tags).
Judgements are cached in outputs/judge_cache_unified.json (file@mtime, shared with the other table scripts).
Writes outputs/tables/TABLE_EUCLID.md.  Main-guarded (judge_safe spawns a worker).
Usage: python scripts/analysis/appendix_euclid.py [--results_dir outputs]"""
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
DS = {"MATH-500": ("c3", "none_r0_K1_last_c3.jsonl", "chart_r1.0_K1_reroll_t0.5_ideal_busemann_L19_c3.jsonl",
                   "chart_r1.0_K1_reroll_t0.25_centre_radius_L19_c3.jsonl", "chart_r1.0_K1_reroll_t*_ideal_euclid_busemann_L19_c3.jsonl",
                   "chart_r1.0_K1_reroll_t*_centre_euclid_radius_L19_c3.jsonl"),
      "GSM8K": ("g8", "none_r0_K1_last_g8.jsonl", "chart_r1.0_K1_reroll_t0.5_ideal_busemann_gsm_L19_g8.jsonl",
                "chart_r1.0_K1_reroll_t0.25_centre_radius_gsm_L19_g8.jsonl", "chart_r1.0_K1_reroll_t*_ideal_euclid_busemann_gsm_L19_g8.jsonl",
                "chart_r1.0_K1_reroll_t*_centre_euclid_radius_gsm_L19_g8.jsonl")}
PROBES = {"MATH-500": ["entchart_busemann_L19.pt", "entchart_euclid_busemann_L19.pt", "entchart_radius_L19.pt", "entchart_euclid_radius_L19.pt"],
          "GSM8K": ["entchart_busemann_gsm_L19.pt", "entchart_euclid_busemann_gsm_L19.pt", "entchart_radius_gsm_L19.pt",
                    "entchart_euclid_radius_gsm_L19.pt"]}
HDR = ("| run | acc, safeguard (Δ vs greedy) | fixed/broken | cap hits | acc, no safeguard | steered tokens: median / pooled | "
       "fires per problem (median) | ‖Δh‖/‖h‖ median | length median | changed | fix rate among changed (fixed / changed & greedy wrong) | "
       "break rate among changed (broken / changed & greedy right) |")
SEP = "|---|---|---|---|---|---|---|---|---|---|---|---|"


def load(p):
    """result rows of a .jsonl file, or of its .jsonl.gz twin"""
    if not os.path.exists(p) and os.path.exists(p + ".gz"):
        p = p + ".gz"
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return {json.loads(l)["idx"]: json.loads(l) for l in f if l.strip()}


def find(pat):
    fs = sorted(x[:-3] if x.endswith(".gz") else x for x in glob.glob(os.path.join(R, pat)) + glob.glob(os.path.join(R, pat + ".gz")))
    return fs[0] if fs else None


def exists(p):
    return p is not None and (os.path.exists(p) or os.path.exists(p + ".gz"))


def probe_quality(names):
    """held-out pair accuracy, Spearman rho and curvature magnitude c saved in the probe files (scripts/train_probe.py)"""
    import torch
    lines = []
    for n in names:
        p = os.path.join(ROOT, "data", "checkpoints", n)
        if not os.path.exists(p):
            lines.append(f"* {n}: (not in data/checkpoints/; train it with scripts/train_probe.py)"); continue
        pk = torch.load(p, map_location="cpu", weights_only=False)
        v = pk.get("val") or {}
        c = 0.0 if pk.get("geometry") == "euclid" else abs(float(pk.get("kappa", float("nan"))))
        lines.append(f"* {n}: geometry {pk.get('geometry', 'hyp')}, c = {c:.2f}, pair acc {v.get('pair_acc', float('nan')):.3f}, "
                     f"Spearman rho {v.get('spearman', float('nan')):.3f}")
    return lines


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

    out = ["# Geometry ablation: Euclidean probes (kappa = 0) vs the hyperbolic probes of HEST -- Qwen2.5-Math-1.5B-Instruct",
           "Same protocol as Table 1 / Table 3: layer 19, theta = 1.0, reroll gate, rho 1.0, cap 3072, unified judge + runaway safeguard "
           "(capped steered decodes report the greedy answer); the Euclidean step lengths match the median |dh|/|h| of HEST on training "
           "hesitation states (scripts/match_step_size.py).  Euclidean Busemann = linear readout -<xi, z>; Euclidean radius = 2|z|.", ""]
    for ds, (tag, gfn, hb, hr, el, er) in DS.items():
        gp = os.path.join(R, gfn)
        g = load(gp); ids = sorted(g)
        gc = np.array([ok(gp, g[i]) for i in ids], float)
        glen = np.median([g[i]["n_tokens"][0] for i in ids])
        arms = [("HEST Busemann t=0.5 (hyperbolic)", os.path.join(R, hb)), ("HEST radius t=0.25 (hyperbolic)", os.path.join(R, hr)),
                ("Euclidean Busemann (matched t)", find(el)), ("Euclidean radius (matched t)", find(er))]
        out += [f"## {ds} (greedy {gc.mean():.4f}, n={len(ids)}, greedy length median {glen:.0f})", HDR, SEP]
        acc = {}
        for label, f in arms:
            if not exists(f):
                out.append(f"| {label} | (not run) | | | | | | | | | | |"); continue
            d = load(f)
            if any(i not in d for i in ids):
                out.append(f"| {label} | ({len(d)}/{len(ids)} running) | | | | | | | | | | |"); continue
            raw = np.array([ok(f, d[i]) for i in ids], float)
            capped = np.array([d[i]["n_tokens"][0] >= CAP for i in ids])
            sg = np.where(capped, gc, raw); acc[label] = sg
            up = int(((sg == 1) & (gc == 0)).sum()); dn = int(((sg == 0) & (gc == 1)).sum())
            fc = np.array([d[i]["fire_count"][0] for i in ids], float); nt = np.array([d[i]["n_tokens"][0] for i in ids], float)
            changed = np.array([d[i]["texts"][0] != g[i]["texts"][0] for i in ids])
            cw, cr = changed & (gc == 0), changed & (gc == 1)
            fixr = f"{up}/{int(cw.sum())} = {up / max(1, cw.sum()):.3f}"; brkr = f"{dn}/{int(cr.sum())} = {dn / max(1, cr.sum()):.3f}"
            rho = [d[i]["field"]["rho"] for i in ids if d[i].get("field")]
            t = os.path.basename(f).split("_t")[1].split("_")[0]
            out.append(f"| {label} (t={t}) | {sg.mean():.4f} ({(sg.mean() - gc.mean()) * 100:+.2f}) | {up}/{dn} | {int(capped.sum())} | "
                       f"{raw.mean():.4f} ({(raw.mean() - gc.mean()) * 100:+.2f}) | {np.median(fc / np.maximum(nt, 1)):.1%} / {fc.sum() / nt.sum():.1%} | "
                       f"{np.median(fc):.0f} | {np.median(rho) if rho else float('nan'):.4f} | {np.median(nt):.0f} | {int(changed.sum())} | {fixr} | {brkr} |")
        out.append("")
        pairs = [("Euclidean Busemann (matched t)", "HEST Busemann t=0.5 (hyperbolic)"), ("Euclidean radius (matched t)", "HEST radius t=0.25 (hyperbolic)"),
                 ("Euclidean radius (matched t)", "HEST Busemann t=0.5 (hyperbolic)")]
        out += [f"Paired counts, {ds} (only the hyperbolic run right : only the Euclidean run right):", ""]
        for e, h in pairs:
            if e in acc and h in acc:
                x, y = acc[h], acc[e]
                out.append(f"* {h} vs {e}: {int(((x == 1) & (y == 0)).sum())} : {int(((x == 0) & (y == 1)).sum())}  (net {int(x.sum() - y.sum()):+d} problems for the hyperbolic run)")
        out += ["", f"Held-out probe quality, {ds} probes:", "", *probe_quality(PROBES[ds]), ""]
    for extra in ("euclid_calibration.txt", "euclid_diversity.txt"):
        p = os.path.join(ROOT, "outputs", extra)
        if os.path.exists(p):
            out += [f"## {extra}", "```", open(p, encoding="utf-8").read().strip(), "```", ""]
    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(out_dir, "TABLE_EUCLID.md"), "w", encoding="utf-8") as fo:
        fo.write(txt + "\n")
    os.makedirs(os.path.dirname(CACHE_P), exist_ok=True)
    json.dump(cache, open(CACHE_P, "w", encoding="utf-8"))
    shutdown()
    print("EUCLID_TABLE_DONE")


if __name__ == "__main__":
    main()
