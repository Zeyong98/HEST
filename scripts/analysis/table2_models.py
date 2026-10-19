"""Table 2: three models x two datasets, Busemann t=0.5 and radius t=0.25 at each model's gate threshold.
Pipeline grading for every row: unified judge (judge_unified) + runaway safeguard (a steered decode that hits the
token cap reports the greedy answer).  Prints the table and writes outputs/tables/TABLE2.md.  Missing runs: '(not run)'.
Main-guarded: judge_safe spawns worker processes."""
import glob, gzip, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "outputs")
CAP = 3072


def load(p):
    """result rows of a .jsonl file, or of its .jsonl.gz twin (the released result files are gzipped)"""
    if not os.path.exists(p) and os.path.exists(p + ".gz"):
        p = p + ".gz"
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return {json.loads(l)["idx"]: json.loads(l) for l in f if l.strip()}


ROWS = [  # (model label, layer, dataset label, greedy tag, run tag, Busemann chart tag, radius chart tag, gate threshold)
    ("Qwen2.5-Math-1.5B-Instruct", 19, "MATH-500", "c3", "c3", "busemann_L19", "radius_L19", "1.0"),
    ("Qwen2.5-Math-1.5B-Instruct", 19, "GSM8K", "g8", "g8", "busemann_gsm_L19", "radius_gsm_L19", "1.0"),
    ("Llama-3.1-8B-Instruct", 13, "MATH-500", "l8", "l8", "busemann_llama_L13", "radius_llama_L13", "2.0"),
    ("Llama-3.1-8B-Instruct", 13, "GSM8K", "l8g", "l8g", "busemann_llama_gsm_L13", "radius_llama_gsm_L13", "1.4"),
    ("Qwen2.5-Math-7B-Instruct", 19, "MATH-500", "q7", "q7", "busemann_q7_L19", "radius_q7_L19", "1.0"),
    ("Qwen2.5-Math-7B-Instruct", 19, "GSM8K", "q7g", "q7g", "busemann_q7_gsm_L19", "radius_q7_gsm_L19", "0.8"),
]


def main():
    global R
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--results_dir", default=None, help="directory of the result rows (default outputs/)")
    a = ap.parse_args()
    if a.results_dir:
        R = a.results_dir if os.path.isabs(a.results_dir) else os.path.join(ROOT, a.results_dir)
    out_dir = os.path.join(ROOT, "outputs", "tables"); os.makedirs(out_dir, exist_ok=True)
    from hest.evaluation.judge_safe import uni_score, shutdown
    ok = lambda r: bool(uni_score(r["texts"][0], str(r["gold"]), r["preds"][0] if r.get("preds") else None)[0])
    out = ["| model | layer | gate | dataset | greedy | Busemann t=0.5 | radius t=0.25 |", "|---|---|---|---|---|---|---|"]
    for model, L, ds, gtag, atag, bch, rch, thr in ROWS:
        gp = f"{R}/none_r0_K1_last_{gtag}.jsonl"
        if not (os.path.exists(gp) or os.path.exists(gp + ".gz")) or thr is None:
            out.append(f"| {model} | {L} | ? | {ds} | (running) | | |"); continue
        g = load(gp); ids = sorted(g); gc = np.array([ok(g[i]) for i in ids], float)
        sfx = "" if float(thr) == 1.0 else f"e{thr}"
        cells = []
        for t, ch in (("0.5", bch), ("0.25", rch)):
            fs = sorted(x[:-3] if x.endswith(".gz") else x for x in glob.glob(f"{R}/chart_r1.0_K1_reroll{sfx}_t{t}_*_{ch}_{atag}.jsonl*"))
            d = load(fs[0]) if fs else {}
            if len(d) < len(ids):
                cells.append("(not run)" if not fs else f"({len(d)}/{len(ids)} running)"); continue
            ac = np.array([gc[k] if d[i]["n_tokens"][0] >= CAP else ok(d[i]) for k, i in enumerate(ids)], float)
            fb = int(sum(d[i]["n_tokens"][0] >= CAP for i in ids))
            up = int(((ac == 1) & (gc == 0)).sum()); dn = int(((ac == 0) & (gc == 1)).sum())
            cells.append(f"{ac.mean():.4f} ({(ac.mean() - gc.mean()) * 100:+.2f}; {up}/{dn}; fb {fb})")
        out.append(f"| {model} | {L} | H>{thr} | {ds} | {gc.mean():.4f} | {cells[0]} | {cells[1]} |")
    hdr = ["# Table 2 -- Busemann t=0.5 and radius t=0.25 at each model's gate threshold",
           "Grading: the paper's protocol for every row -- unified judge (boxed / explicit marker / model restatement / last number; "
           "tiered equivalence) + runaway safeguard (a steered decode that hits the 3072-token cap reports the greedy answer).",
           "Cells: accuracy (change vs greedy in pts; fixed/broken; fb = problems that fell back to greedy).",
           "Gate thresholds: Qwen-1.5B 1.0 (reference); Llama and Qwen-7B by the label-free profile-matching rule on their own greedy "
           "outputs: Llama 2.0 / 1.4, Qwen-7B 1.0 / 0.8.  Layers: Qwen 19, Llama 13.  "
           "Step lengths are the Qwen-1.5B defaults (Busemann 0.5, radius 0.25).", ""]
    txt = "\n".join(hdr + out)
    print(txt)
    open(os.path.join(out_dir, "TABLE2.md"), "w", encoding="utf-8").write(txt + "\n")
    shutdown()


if __name__ == "__main__":
    main()
