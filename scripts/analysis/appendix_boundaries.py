"""Appendix statistic (CPU only, tokenizer only): how much do entropy-gate positions coincide with reasoning-step boundaries?

Data: the 500 greedy MATH-500 solutions of Qwen2.5-Math-1.5B-Instruct (<results_dir>/none_r0_K1_last_c3.jsonl) and their
teacher-forced per-token entropies (<results_dir>/c3_probe_all.npz: e{idx}[k] = entropy of the distribution that predicts
generated token k; k = 0 comes from the prompt's last position).  Tokenisation exactly as in scripts/calibrate_threshold.py:
tok(text, add_special_tokens=False), so e{idx} and the token list align (checked per problem; mismatches are skipped).

Definitions: boundary token b = a generated token whose decoding contains "\n\n"; token b+1 is the
first token of a new step.  Hesitation k = e[k] > THR with k >= 1 (the gate never fires on the prompt position).
  (a) share of step starts that are hesitations : #{b : e[b+1] > THR} / #{b}
  (b) share of hesitations that are step starts : #{k : e[k] > THR and token k-1 contains "\n\n"} / #{k : e[k] > THR}
Offsets: for each hesitation k, the number of tokens between the last boundary token and the token read at k-1
(0 = the hesitation is the first token of a step).
Usage: python scripts/analysis/appendix_boundaries.py [--results_dir outputs]  ->  outputs/tables/ABL0_boundary_stats.txt"""
import json
import os

import numpy as np
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, "outputs")
THR = float(os.environ.get("ABL_THR", "1.0"))
MODEL = os.environ.get("CX_MODEL", "Qwen/Qwen2.5-Math-1.5B-Instruct")
CAP = 3072


def stats(tok, rows, P, ids, label):
    n_b = n_b_hes = n_hes = n_hes_b = n_hes0 = n_before = 0
    per_b, per_h, per_len, offs, skipped = [], [], [], [], []
    dec = {}
    for i in ids:
        e = P[f"e{i}"]
        toks = tok(rows[i]["texts"][0], add_special_tokens=False).input_ids
        if len(toks) != len(e):
            skipped.append((i, len(toks), len(e)))
            continue
        n = len(toks)
        isb = np.array([("\n\n" in dec.setdefault(t, tok.decode([t]))) for t in toks], bool)
        bpos = np.where(isb)[0]                                    # boundary tokens b; the new step starts at b+1
        hes = np.where(e > THR)[0]
        n_hes0 += int((hes == 0).sum())
        hes = hes[hes >= 1]
        starts = bpos[bpos + 1 < n] + 1
        n_b += len(starts)
        n_b_hes += int((e[starts] > THR).sum())
        n_hes += len(hes)
        n_hes_b += int(isb[hes - 1].sum())
        per_b.append(len(bpos)); per_h.append(len(hes)); per_len.append(n)
        for k in hes:
            prev = bpos[bpos <= k - 1]
            if len(prev):
                offs.append(int(k - 1 - prev.max()))
            else:
                n_before += 1
    offs = np.array(offs)
    out = [f"[{label}] problems {len(per_b)} (skipped for tokenisation mismatch: {len(skipped)} {skipped[:5]}), THR {THR}",
           f"  tokens per solution median {np.median(per_len):.0f} mean {np.mean(per_len):.0f}; boundary tokens per solution median "
           f"{np.median(per_b):.0f} mean {np.mean(per_b):.1f}; hesitations (k>=1) per solution median {np.median(per_h):.0f} mean {np.mean(per_h):.1f}",
           f"  (a) step starts that are hesitations: {n_b_hes}/{n_b} = {n_b_hes / max(n_b, 1):.3f}",
           f"  (b) hesitations that are step starts: {n_hes_b}/{n_hes} = {n_hes_b / max(n_hes, 1):.3f}",
           f"  hesitations at k=0 (prompt position, never gated): {n_hes0}; hesitations before the first boundary: {n_before}",
           f"  offset of a hesitation from the last boundary (tokens): median {np.median(offs):.0f} p75 {np.percentile(offs, 75):.0f} "
           f"p90 {np.percentile(offs, 90):.0f}; within 0/1/2/5/10 tokens: " + " / ".join(f"{np.mean(offs <= k):.2f}" for k in (0, 1, 2, 5, 10))]
    return out


def main():
    global R
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--results_dir", default=None, help="directory of the greedy rows and c3_probe_all.npz (default outputs/)")
    a = ap.parse_args()
    if a.results_dir:
        R = a.results_dir if os.path.isabs(a.results_dir) else os.path.join(ROOT, a.results_dir)
    out_dir = os.path.join(ROOT, "outputs", "tables"); os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = {json.loads(l)["idx"]: json.loads(l) for l in open(os.path.join(R, "none_r0_K1_last_c3.jsonl"), encoding="utf-8") if l.strip()}
    P = np.load(os.path.join(R, "c3_probe_all.npz"))
    ids = [int(i) for i in P["ids"]]
    lines = stats(tok, rows, P, ids, "all 500 greedy solutions")
    unc = [i for i in ids if rows[i]["n_tokens"][0] < CAP]
    lines += stats(tok, rows, P, unc, f"excluding the {len(ids) - len(unc)} capped solutions")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(out_dir, "ABL0_boundary_stats.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("ABL0_DONE")


if __name__ == "__main__":
    main()
