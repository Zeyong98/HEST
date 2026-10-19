"""Ablation prerequisite (GPU, a few minutes): the relative size |dh|/|h| of the HEST step (Busemann, t = 0.5) at the
states where the gate fires on the TEST set, for the layer-19 main chart and the layer-9 / layer-25 charts.

States: teacher-forced layer-L states of the greedy MATH-500 solutions of Qwen2.5-Math-1.5B-Instruct
(none_r0_K1_last_c3.jsonl; capped solutions excluded, their junk loops would dominate) at every hesitation position
k >= 1 with e[k] > 1.0 (c3_probe_all.npz), i.e. the state that HEST edits when its gate fires on the greedy trajectory.
A second population, 2000 random generated positions, gives the step size the every-token ablation applies.
Writes outputs/abl_disp_check.txt and outputs/abl_rel_l19.txt (the layer-19 hesitation median, 3 decimals: the step
size of the fixed-direction ablation).
Usage: CX_MODEL=Qwen/Qwen2.5-Math-1.5B-Instruct CX_LAYER=19 python scripts/analysis/appendix_step_size.py [--results_dir outputs]"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from hest.config import CHECKPOINTS, DATA  # noqa: E402
from hest.decoding import LMB  # noqa: E402
from hest.field import ChartField  # noqa: E402

R = os.path.join(ROOT, "outputs")
CH = CHECKPOINTS
CHARTS = {19: "entchart_busemann_L19.pt", 9: "entchart_busemann_all_L9.pt", 25: "entchart_busemann_all_L25.pt"}
CAP, THR, T, NRAND = 3072, 1.0, 0.5, 2000


def main():
    global R
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--results_dir", default=None, help="directory of the greedy rows and c3_probe_all.npz (default outputs/)")
    a = ap.parse_args()
    if a.results_dir:
        R = a.results_dir if os.path.isabs(a.results_dir) else os.path.join(ROOT, a.results_dir)
    OUT = os.path.join(ROOT, "outputs"); os.makedirs(OUT, exist_ok=True)
    probs = json.load(open(DATA["math500"], encoding="utf-8"))
    rows = {json.loads(l)["idx"]: json.loads(l) for l in open(os.path.join(R, "none_r0_K1_last_c3.jsonl"), encoding="utf-8") if l.strip()}
    P = np.load(os.path.join(R, "c3_probe_all.npz"))
    ids = [int(i) for i in P["ids"] if rows[int(i)]["n_tokens"][0] < CAP]
    lm = LMB(); lm.active = False
    rng = np.random.default_rng(0)
    hes_states = {L: [] for L in CHARTS}
    rnd_states = {L: [] for L in CHARTS}
    n_hes, n_skip = 0, 0
    with torch.no_grad():
        for i in ids:
            e = P[f"e{i}"]
            pid = lm.ids(lm.chat(probs[i]["problem"])); Pn = pid.shape[1]
            gen = lm.tok(rows[i]["texts"][0], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            if gen.shape[1] != len(e):
                n_skip += 1; continue
            full = torch.cat([pid, gen], 1)
            hes = np.where(e > THR)[0]; hes = hes[hes >= 1]
            want = {int(Pn - 1 + k): "h" for k in hes}
            for k in rng.choice(len(e), size=min(4, len(e)), replace=False):     # ~4 random positions per solution
                want.setdefault(int(Pn - 1 + k), "r")
            past, pos = None, 0
            while pos < full.shape[1]:
                chunk = full[:, pos: pos + 512]
                out = lm.model(input_ids=chunk, past_key_values=past, use_cache=True, output_hidden_states=True)
                past = out.past_key_values
                for p, kind in want.items():
                    if pos <= p < pos + chunk.shape[1]:
                        for L in CHARTS:
                            (hes_states if kind == "h" else rnd_states)[L].append(out.hidden_states[L + 1][0, p - pos, :].float().cpu())
                del out
                pos += chunk.shape[1]
            del past
            n_hes += len(hes)
            torch.cuda.empty_cache()
    lines = [f"greedy MATH-500 solutions used {len(ids)} (capped excluded), tokenisation mismatches skipped {n_skip}; "
             f"hesitation states (k>=1, H>{THR}) {n_hes}; random positions {len(rnd_states[19])}; Busemann step t={T}, rho_max 1.0"]
    for L, cf in CHARTS.items():
        f = ChartField(os.path.join(CH, cf), target="ideal", t=T, rho_max=1.0)
        for lab, pool in (("hesitation", hes_states[L]), ("random", rnd_states[L][:NRAND])):
            X = torch.stack(pool).cuda()
            f.log.clear()
            for s in range(0, X.shape[0], 256):
                f(X[s: s + 256])
            rel = np.array([x[3] for x in f.log]); dhyp = np.array([x[2] for x in f.log])
            rb = np.array([x[0] for x in f.log]); ra = np.array([x[1] for x in f.log])
            lines.append(f"L{L:<2d} {cf:32s} {lab:10s} n {len(rel):5d}: |dh|/|h| median {np.median(rel):.4f} (p25 {np.percentile(rel, 25):.4f}, "
                         f"p75 {np.percentile(rel, 75):.4f}, mean {rel.mean():.4f}); clipped at rho_max: {int((rel >= 0.999).sum())}; "
                         f"hyperbolic distance moved median {np.median(dhyp):.3f}; reading before/after median {np.median(rb):.3f} -> {np.median(ra):.3f}")
            if L == 19 and lab == "hesitation":
                with open(os.path.join(OUT, "abl_rel_l19.txt"), "w") as fo:
                    fo.write(f"{np.median(rel):.3f}\n")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "abl_disp_check.txt"), "w", encoding="utf-8") as fo:
        fo.write(txt + "\n")
    print("ABL_DISP_DONE")


if __name__ == "__main__":
    main()
