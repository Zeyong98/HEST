"""Table 10 (Appendix J; output files TABLE_S1_*): accuracy change vs greedy by hesitation-count regime (gate fires per solution: 0-10 / 11-24 / 25+),
under the paper's pipeline grading: unified judge (judge_unified) + runaway safeguard (a steered decode that hits the
token cap reports the greedy answer).

--bins arm     bins use the steered run's own fire count (subject to reverse causality: a broken trajectory tends
               to hesitate more and drift into 25+)
--bins greedy  bins use the fire count of the GREEDY output at the same gate (teacher-forced probe,
               <results_dir>/{tag}_probe_all.npz from scripts/calibrate_threshold.py --include-capped): a quantity that exists
               before any intervention.
Cells: change in accuracy (pts) [share of problems; g = greedy acc in the bin; f = P(fixed | greedy wrong);
b = P(broken | greedy right)].  "all t" pools the four step lengths.
Usage (the paper's Table 10): python scripts/analysis/appendix_regimes.py --bins greedy --fine [--results_dir outputs]
       (Table 11, groups by hesitation density: --bins density)"""
import argparse, gzip, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "outputs")
CAP = 3072
BINS = [(0, 10, "0-10"), (11, 24, "11-24"), (25, 10 ** 6, "25+")]
# dataset -> (greedy file, run pattern, split file, doses, gate theta, greedy-probe npz)
DS = {"MATH-500": ("none_r0_K1_last_c3.jsonl", "chart_r1.0_K1_reroll_t{t}_{tg}_{ch}_L19_c3.jsonl", None, ["0.1", "0.25", "0.5", "1.0"], 1.0, "c3_probe_all.npz"),
      "GSM8K": ("none_r0_K1_last_g8.jsonl", "chart_r1.0_K1_reroll_t{t}_{tg}_{ch}_gsm_L19_g8.jsonl", None, ["0.1", "0.25", "0.5", "1.0"], 1.0, "g8_probe_all.npz"),
      "OlympiadBench-575": ("none_r0_K1_last_oly.jsonl", "chart_r1.0_K1_reroll_t{t}_{tg}_{ch}_oly_L19_oly2.jsonl", "oly_split.json", ["0.1", "0.25", "0.5", "1.0"], 1.0, "oly_probe_all.npz"),
      "Llama MATH-500": ("none_r0_K1_last_l8.jsonl", "chart_r1.0_K1_rerolle2.0_t{t}_{tg}_{ch}_llama_L13_l8.jsonl", None, None, 2.0, "l8_probe_all.npz"),
      "Llama GSM8K": ("none_r0_K1_last_l8g.jsonl", "chart_r1.0_K1_rerolle1.4_t{t}_{tg}_{ch}_llama_gsm_L13_l8g.jsonl", None, None, 1.4, "l8g_probe_all.npz"),
      "Qwen-7B MATH-500": ("none_r0_K1_last_q7.jsonl", "chart_r1.0_K1_reroll_t{t}_{tg}_{ch}_q7_L19_q7.jsonl", None, None, 1.0, "q7_probe_all.npz"),
      "Qwen-7B GSM8K": ("none_r0_K1_last_q7g.jsonl", "chart_r1.0_K1_rerolle0.8_t{t}_{tg}_{ch}_q7_gsm_L19_q7g.jsonl", None, None, 0.8, "q7g_probe_all.npz")}
CH = {"ideal": "busemann", "centre": "radius"}


def load(p):
    """result rows of a .jsonl file, or of its .jsonl.gz twin (the released result files are gzipped)"""
    if not os.path.exists(p) and os.path.exists(p + ".gz"):
        p = p + ".gz"
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return {json.loads(l)["idx"]: json.loads(l) for l in f if l.strip()}


def greedy_counts(npz, thr):
    z = np.load(os.path.join(R, npz))
    return {int(i): int((z[f"e{int(i)}"] > thr).sum()) for i in z["ids"]}


def greedy_density(npz, thr):
    z = np.load(os.path.join(R, npz))
    return {int(i): float((z[f"e{int(i)}"] > thr).mean()) for i in z["ids"]}


CACHE_P = os.path.join(ROOT, "outputs", "judge_cache_unified.json")


def main():
    global R
    from hest.evaluation.judge_safe import uni_score, shutdown
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=None, help="directory of the result rows and *_probe_all.npz files (default outputs/)")
    ap.add_argument("--bins", default="arm", choices=["arm", "greedy", "density"]); ap.add_argument("--no-safeguard", action="store_true")
    ap.add_argument("--fine", action="store_true", help="split the 25+ bin into 25-49 and 50+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.results_dir:
        R = a.results_dir if os.path.isabs(a.results_dir) else os.path.join(ROOT, a.results_dir)
    out_dir = os.path.join(ROOT, "outputs", "tables"); os.makedirs(out_dir, exist_ok=True)
    out = a.out or os.path.join(out_dir, f"TABLE_S1_{a.bins}{'_fine' if a.fine else ''}.txt")
    safeguard = not a.no_safeguard
    if a.bins == "density":
        bins = [(0.0, 0.02, "<2%"), (0.02, 0.10, "2-10%"), (0.10, 9.0, ">10%")]
    else:
        bins = [(0, 10, "0-10"), (11, 24, "11-24"), (25, 49, "25-49"), (50, 10 ** 6, "50+")] if a.fine else BINS
    cache = json.load(open(CACHE_P, encoding="utf-8")) if os.path.exists(CACHE_P) else {}

    def ok_cached(fname, r):
        fname = fname if os.path.exists(fname) else fname + ".gz"
        key = f"{os.path.basename(fname)}@{int(os.path.getmtime(fname))}"
        c = cache.setdefault(key, {})
        k = str(r["idx"])
        if k not in c:
            c[k] = bool(uni_score(r["texts"][0], str(r["gold"]), r["preds"][0] if r.get("preds") else None)[0])
        return c[k]
    lines = [f"Table S1 -- unified judge, safeguard={'on' if safeguard else 'off'}, bins by {'the steered run' if a.bins == 'arm' else 'the GREEDY output'}'s fire {'density (fires / tokens)' if a.bins == 'density' else 'count'}; "
             f"cells: change (pts) [share; g / f / b]"]
    hdr = f"{'dataset':18s} {'setting':16s} {'overall acc (delta; fixed/broken)':>34s} |" + "".join(f" {lab:>34s}" for _, _, lab in bins)
    print(hdr); lines.append(hdr)
    md_rows = []
    for name, (gp, pat, split, doses, thr, npz) in DS.items():
        gpath = os.path.join(R, gp)
        g = load(gpath); ids = sorted(g)
        if split:
            keep = set(json.load(open(os.path.join(ROOT, "data", "ids", split), encoding="utf-8"))["test"]); ids = [i for i in ids if i in keep]
        gc = {i: ok_cached(gpath, g[i]) for i in ids}
        gfc = None
        if a.bins in ("greedy", "density"):
            p = os.path.join(R, npz)
            if not os.path.exists(p):
                print(f"{name}: no greedy probe {npz} yet, skipped"); continue
            gfc = greedy_counts(npz, thr) if a.bins == "greedy" else greedy_density(npz, thr)
            ids = [i for i in ids if i in gfc]
        acache = {}

        def arm(tg, t):
            k = (tg, t)
            if k not in acache:
                p = os.path.join(R, pat.format(t=t, tg=tg, ch=CH[tg]))
                if not (os.path.exists(p) or os.path.exists(p + ".gz")):
                    acache[k] = None
                else:
                    d = load(p)
                    if any(i not in d for i in ids):                 # run still in progress: skip, do not shrink the id set
                        acache[k] = None
                        return None
                    acc = {i: (gc[i] if (safeguard and d[i]["n_tokens"][0] >= CAP) else ok_cached(p, d[i])) for i in ids}
                    acache[k] = (acc, {i: (gfc[i] if gfc is not None else d[i]["fire_count"][0]) for i in ids})
            return acache[k]

        settings = [("Busemann", "ideal", "0.5"), ("Busemann", "ideal", "0.25"), ("radius", "centre", "0.25")]
        if doses:
            settings += [("Busemann", "ideal", "all"), ("radius", "centre", "all")]
        for lbl, tg, t in settings:
            ts = doses if t == "all" else [t]
            GC, AC, FI = [], [], []
            for tt in ts:
                res = arm(tg, tt)
                if res is None:
                    continue
                acc, fc = res
                GC += [gc[i] for i in ids]; AC += [acc[i] for i in ids]; FI += [fc[i] for i in ids]
            if not GC:
                continue
            GC, AC, FI = np.array(GC, float), np.array(AC, float), np.array(FI, float)
            nrep = len(ts) if t == "all" else 1
            fx = int(((AC == 1) & (GC == 0)).sum()) // nrep; br = int(((AC == 0) & (GC == 1)).sum()) // nrep
            row = f"{name:18s} {lbl + ' t=' + t:16s} {AC.mean():.4f} ({(AC.mean() - GC.mean()) * 100:+.2f}; {fx}/{br}) |"
            md = [name, f"{lbl} t={t}", f"{AC.mean():.4f} ({(AC.mean() - GC.mean()) * 100:+.2f}; {fx}/{br})"]
            for lo, hi, lab in bins:
                m = ((FI >= lo) & (FI < hi)) if a.bins == "density" else ((FI >= lo) & (FI <= hi)); n = int(m.sum())
                if n:
                    d = (AC[m].mean() - GC[m].mean()) * 100
                    right, wrong = m & (GC == 1), m & (GC == 0)
                    gg = GC[m].mean()
                    f = ((AC == 1) & wrong).sum() / max(1, wrong.sum())
                    b = ((AC == 0) & right).sum() / max(1, right.sum())
                    cell = f"{d:+.2f} [{m.mean():.0%}; g{gg:.2f} f{f:.3f} b{b:.3f}]"
                    md.append(f"{d:+.2f} [{n}; {gg:.2f}/{f:.3f}/{b:.3f}]")
                else:
                    cell = "-"
                    md.append("-")
                row += f" {cell:>34s}"
            print(row, flush=True); lines.append(row); md_rows.append(md)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(out.replace(".txt", ".md"), "w", encoding="utf-8") as f:       # markdown twin for the paper tables
        f.write("| dataset | run | acc (Δ; fixed/broken) | " + " | ".join(lab for _, _, lab in bins) + " |\n")
        f.write("|---|---|---|" + "---|" * len(bins) + "\n")
        for r in md_rows:
            f.write("| " + " | ".join(r) + " |\n")
    os.makedirs(os.path.dirname(CACHE_P), exist_ok=True)
    json.dump(cache, open(CACHE_P, "w", encoding="utf-8"))
    shutdown()
    print("REGIME_DONE")


if __name__ == "__main__":
    main()
