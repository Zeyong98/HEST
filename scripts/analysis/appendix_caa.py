"""CAA baseline (appendix): the six-cell table (unified judge, no safeguard; the safeguarded value is a diagnostic of
how much of the loss is runaways) and the exposure comparison with the entropy-gated method (tokens steered per
solution and per-token push).  Usage: python scripts/analysis/appendix_caa.py [--results_dir outputs]
Writes outputs/tables/TABLE_CAA.md and outputs/tables/caa_exposure.txt."""
import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from hest.config import OUT_DIR  # noqa: E402

CELLS = [("Qwen-1.5B", "MATH-500", "c3", "q15_math"), ("Qwen-1.5B", "GSM8K", "g8", "q15_gsm"),
         ("Llama-8B", "MATH-500", "l8", "l8_math"), ("Llama-8B", "GSM8K", "l8g", "l8_gsm"),
         ("Qwen-7B", "MATH-500", "q7", "q7_math"), ("Qwen-7B", "GSM8K", "q7g", "q7_gsm")]
EXPOSURE = [  # name, CAA run, entropy-gated Busemann t=0.5 run, |v|/median|h| (build_caa.py output), |dh|/|h| at t=0.5 (appendix_step_size.py / paper)
    ("Qwen-1.5B MATH", "vec_r0_K1_all_m1_q15_math_c3.jsonl", "chart_r1.0_K1_reroll_t0.5_ideal_busemann_L19_c3.jsonl", .329, .060),
    ("Qwen-1.5B GSM8K", "vec_r0_K1_all_m1_q15_gsm_g8.jsonl", "chart_r1.0_K1_reroll_t0.5_ideal_busemann_gsm_L19_g8.jsonl", .235, .060),
    ("Llama-8B MATH", "vec_r0_K1_all_m1_l8_math_l8.jsonl", "chart_r1.0_K1_rerolle2.0_t0.5_ideal_busemann_llama_L13_l8.jsonl", .295, .058),
    ("Llama-8B GSM8K", "vec_r0_K1_all_m1_l8_gsm_l8g.jsonl", "chart_r1.0_K1_rerolle1.4_t0.5_ideal_busemann_llama_gsm_L13_l8g.jsonl", .234, .058),
    ("Qwen-7B MATH", "vec_r0_K1_all_m1_q7_math_q7.jsonl", "chart_r1.0_K1_reroll_t0.5_ideal_busemann_q7_L19_q7.jsonl", .182, .046),
    ("Qwen-7B GSM8K", "vec_r0_K1_all_m1_q7_gsm_q7g.jsonl", "chart_r1.0_K1_rerolle0.8_t0.5_ideal_busemann_q7_gsm_L19_q7g.jsonl", .217, .052),
]
CAP = 3072


def load(p):
    """result rows of a .jsonl file, or of its .jsonl.gz twin"""
    if not os.path.exists(p) and os.path.exists(p + ".gz"):
        p = p + ".gz"
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return {json.loads(l)["idx"]: json.loads(l) for l in f if l.strip()}


def caa_table(R, out_dir):
    from hest.evaluation.judge_safe import uni_score, shutdown
    ok = lambda r: bool(uni_score(r["texts"][0], str(r["gold"]), r["preds"][0] if r.get("preds") else None)[0])
    out = ["| model | dataset | greedy | CAA all-token, multiplier 1 (no safeguard) | fixed/broken | cap-hits | mean len ratio | changed | [with safeguard, diagnostic] |",
           "|---|---|---|---|---|---|---|---|---|"]
    for model, ds, gtag, vtag in CELLS:
        g = load(os.path.join(R, f"none_r0_K1_last_{gtag}.jsonl")); ids = sorted(g)
        fs = sorted(x[:-3] if x.endswith(".gz") else x for x in glob.glob(os.path.join(R, f"vec_r0_K1_all_m1_{vtag}_{gtag}.jsonl*")))
        if not fs:
            out.append(f"| {model} | {ds} | | (not run) | | | | | |"); continue
        d = load(fs[0])
        if any(i not in d for i in ids):
            out.append(f"| {model} | {ds} | | ({len(d)}/{len(ids)} running) | | | | | |"); continue
        gc = np.array([ok(g[i]) for i in ids], float); a = np.array([ok(d[i]) for i in ids], float)
        cap = np.array([d[i]["n_tokens"][0] >= 3072 for i in ids]); a_sg = np.where(cap, gc, a)
        up = int(((a == 1) & (gc == 0)).sum()); dn = int(((a == 0) & (gc == 1)).sum())
        lr = np.mean([d[i]["n_tokens"][0] for i in ids]) / np.mean([g[i]["n_tokens"][0] for i in ids])
        ch = np.mean([d[i]["texts"][0] != g[i]["texts"][0] for i in ids])
        out.append(f"| {model} | {ds} | {gc.mean():.4f} | {a.mean():.4f} ({(a.mean() - gc.mean()) * 100:+.2f}) | {up}/{dn} | {int(cap.sum())} | {lr:.2f}x | {ch:.0%} | {a_sg.mean():.4f} ({(a_sg.mean() - gc.mean()) * 100:+.2f}) |")
    txt = "\n".join(out); print(txt)
    open(os.path.join(out_dir, "TABLE_CAA.md"), "w", encoding="utf-8").write("# CAA baseline (all-token, raw mean-difference vector x multiplier 1, unified judge, no safeguard)\n\n" + txt + "\n")
    shutdown()


def exposure(R, out_dir):
    out = [f"{'cell':16} | CAA tokens/solution mean (uncapped mean, median) | ours fires/solution mean (uncapped mean, median)"
           f" | tokens x (uncapped means) | tokens x (medians) | push x/token | push x total (uncapped)"]
    for name, cf, of, vrel, drel in EXPOSURE:
        if not (os.path.exists(os.path.join(R, cf)) or os.path.exists(os.path.join(R, cf) + ".gz")) or \
           not (os.path.exists(os.path.join(R, of)) or os.path.exists(os.path.join(R, of) + ".gz")):
            out.append(f"{name:16} | (not run)"); continue
        c, o = load(os.path.join(R, cf)), load(os.path.join(R, of))
        ct = np.array([r["n_tokens"][0] for r in c.values()])
        cu = ct[ct < CAP]
        fc = np.array([(r.get("fire_count") or 0) for r in o.values()])
        ot = np.array([r["n_tokens"][0] for r in o.values()])
        fu = fc[ot < CAP]
        out.append(f"{name:16} | {ct.mean():5.0f} ({cu.mean():5.0f}, {np.median(ct):4.0f}) n={len(c):4d} | "
                   f"{fc.mean():5.1f} ({fu.mean():5.1f}, {np.median(fc):3.0f}) n={len(o):4d} capped {int((ot >= CAP).sum()):3d}, max {int(fc.max()):5d} | "
                   f"{cu.mean() / fu.mean():5.0f} | {np.median(ct) / max(np.median(fc), 1):5.0f} | {vrel / drel:4.1f} | {vrel * cu.mean() / (drel * fu.mean()):5.0f}")
    print("\n".join(out))
    with open(os.path.join(out_dir, "caa_exposure.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=OUT_DIR)
    a = ap.parse_args()
    out_dir = os.path.join(OUT_DIR, "tables"); os.makedirs(out_dir, exist_ok=True)
    caa_table(a.results_dir, out_dir)
    exposure(a.results_dir, out_dir)


if __name__ == "__main__":
    main()
