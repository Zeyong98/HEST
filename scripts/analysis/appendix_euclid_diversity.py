"""Diversity of the edits dh of the Euclidean and hyperbolic probes (Appendix N; the hyperbolic rows are Table 13 of
Appendix L) on the 400 MATH training hesitation states (layer 19, H > 1, 58 problems): median pairwise cosine of dh
(10th / 90th percentiles), the variance of the leading principal component and of the first five, the number of
components needed for half of the variance, and the median |dh|/|h|.
Runs: the hyperbolic Busemann probe at t = 0.5 and radius probe at t = 0.25, and the two Euclidean probes at the step
lengths of configs/ablation/euclid_{busemann,radius}_math500.json.  Probes that are not in data/checkpoints/ are skipped
(entchart_radius_L19.pt and the Euclidean probes are trained with scripts/train_probe.py).
Input: outputs/states/ent_states_Qwen2.5-Math-1.5B-Instruct_train100.pt (scripts/collect_states.py).
Writes outputs/euclid_diversity.txt.
Usage (GPU): python scripts/analysis/appendix_euclid_diversity.py"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from hest.config import CHECKPOINTS, OUT_DIR, out_path  # noqa: E402
from hest.field import ChartField  # noqa: E402


def states():
    d = torch.load(os.path.join(OUT_DIR, "states", "ent_states_Qwen2.5-Math-1.5B-Instruct_train100.pt"), map_location="cpu", weights_only=False)
    X, H = d["acts"][19].float(), d["ent"]
    return X[torch.where(H > 1.0)[0][:400]].cuda()


def diversity(D):
    D = D.cpu().double()
    Dn = D / D.norm(dim=1, keepdim=True).clamp_min(1e-9)
    C = Dn @ Dn.T
    iu = torch.triu_indices(C.shape[0], C.shape[0], offset=1)
    cos = C[iu[0], iu[1]].numpy()
    Dc = D - D.mean(0, keepdim=True)
    ev = torch.linalg.svdvals(Dc) ** 2
    frac = (ev / ev.sum()).numpy()
    k50 = int(np.searchsorted(np.cumsum(frac), 0.5) + 1)
    return float(np.median(cos)), float(np.percentile(cos, 10)), float(np.percentile(cos, 90)), float(frac[0]), float(frac[:5].sum()), k50


def main():
    X = states()
    hn = X.norm(dim=1)
    arms = [("hyperbolic Busemann t=0.5", "entchart_busemann_L19.pt", "ideal", 0.5),
            ("hyperbolic radius t=0.25", "entchart_radius_L19.pt", "centre", 0.25)]
    for key in ("busemann", "radius"):
        s = json.load(open(os.path.join(ROOT, "configs", "ablation", f"euclid_{key}_math500.json"), encoding="utf-8"))[0]
        arms.append((f"Euclidean {key} t={s['t']}", os.path.basename(s["chart"]), s["target"], float(s["t"])))
    lines = ["400 MATH training hesitation states (layer 19, H > 1); columns: median pairwise cos (p10/p90), leading PC var, "
             "top-5 PCs var, PCs for 50% var, median |dh|/|h|"]
    for label, chart, target, t in arms:
        p = os.path.join(CHECKPOINTS, chart)
        if not os.path.exists(p):
            lines.append(f"{label:36s} | ({chart} not in data/checkpoints/)"); continue
        f = ChartField(p, target=target, t=t, rho_max=1.0)
        D = torch.cat([f(X[s: s + 200]) for s in range(0, X.shape[0], 200)])
        med, p10, p90, pc1, pc5, k50 = diversity(D)
        rel = float((D.norm(dim=1) / hn).median())
        lines.append(f"{label:36s} | cos {med:.3f} ({p10:.3f}/{p90:.3f}) | PC1 {pc1:.1%} | top5 {pc5:.1%} | PCs50 {k50:3d} | |dh|/|h| {rel:.4f}")
    txt = "\n".join(lines)
    print(txt)
    with open(out_path("euclid_diversity.txt"), "w", encoding="utf-8") as fo:
        fo.write(txt + "\n")
    print("EUCLID_DIVERSITY_DONE")


if __name__ == "__main__":
    main()
