"""Step lengths of the Euclidean probes (geometry ablation, Table 3(c) and Appendix N), matched on TRAINING states.

The two geometries differ in their metric and scale, so the same t does not change the hidden state by the same
amount.  The size of the edit is matched instead of the step length.  Each Euclidean probe is matched to the HEST
probe with the same readout at its step length (Busemann t = 0.5, radius t = 0.25): t is found by bisection so that
its median |dh|/|h| over a fixed set of training hesitation states (H > 1, layer 19) equals that of the hyperbolic
probe on the same states.  No test label enters the choice.
    MATH-500 runs: the first 400 states with H > 1 of the MATH training-state file, in problem order (58 problems).
                   The Busemann target is 0.060, the hyperbolic median (0.0597) to three decimals.
    GSM8K runs:    the states with H > 1 of the GSM8K training-state file (300 states, 85 problems).
Inputs: outputs/states/ent_states_Qwen2.5-Math-1.5B-Instruct_{train100,gsm100}.pt (scripts/collect_states.py); the
hyperbolic probes data/checkpoints/entchart_{busemann,radius}{,_gsm}_L19.pt (the Busemann ones are released, the radius
ones are trained with scripts/train_probe.py --score radius); the four Euclidean probes
data/checkpoints/entchart_euclid_{busemann,radius}{,_gsm}_L19.pt (scripts/train_probe.py --geometry euclid).
Writes outputs/euclid_calibration.txt and outputs/configs/euclid_{busemann,radius}_{math500,gsm8k}.json.  The released
configs/ablation/euclid_*.json hold the values used in the paper (0.613, 0.252, 0.6724, 0.2411).
Usage (GPU): python scripts/match_step_size.py"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hest.config import CHECKPOINTS, OUT_DIR, out_path  # noqa: E402
from hest.field import ChartField  # noqa: E402

SETS = {"math500": ("ent_states_Qwen2.5-Math-1.5B-Instruct_train100.pt", 400),
        "gsm8k": ("ent_states_Qwen2.5-Math-1.5B-Instruct_gsm100.pt", 400)}
PROBES = {"math500": {"busemann": ("entchart_euclid_busemann_L19.pt", "ideal"), "radius": ("entchart_euclid_radius_L19.pt", "centre")},
          "gsm8k": {"busemann": ("entchart_euclid_busemann_gsm_L19.pt", "ideal"), "radius": ("entchart_euclid_radius_gsm_L19.pt", "centre")}}
HYP_REF = {"math500": {"busemann": ("entchart_busemann_L19.pt", "ideal", 0.5), "radius": ("entchart_radius_L19.pt", "centre", 0.25)},
           "gsm8k": {"busemann": ("entchart_busemann_gsm_L19.pt", "ideal", 0.5), "radius": ("entchart_radius_gsm_L19.pt", "centre", 0.25)}}
LAYER, THR = 19, 1.0


def calib_states(name):
    fn, n = SETS[name]
    d = torch.load(os.path.join(OUT_DIR, "states", fn), map_location="cpu", weights_only=False)
    X, H, prob = d["acts"][LAYER].float(), d["ent"], d["prob"]
    idx = torch.where(H > THR)[0][:n]
    return X[idx].cuda(), len(set(prob[idx].tolist()))


def median_rel(chart, target, t, X):
    f = ChartField(os.path.join(CHECKPOINTS, chart), target=target, t=t, rho_max=1.0)
    f.log.clear()
    for s in range(0, X.shape[0], 256):
        f(X[s: s + 256])
    rel = np.array([x[3] for x in f.log]); rb = np.array([x[0] for x in f.log]); ra = np.array([x[1] for x in f.log])
    return float(np.median(rel)), float(np.median(rb)), float(np.median(ra)), int((rel >= 0.999).sum())


def bisect_t(chart, target, X, goal, lo=1e-3, hi=8.0, iters=40):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        m = median_rel(chart, target, mid, X)[0]
        if m < goal:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-5:
            break
    return 0.5 * (lo + hi)


def main():
    lines = []
    for name in ("math500", "gsm8k"):
        X, nprob = calib_states(name)
        lines.append(f"[{name}] calibration states {X.shape[0]} from {nprob} problems (H > {THR})")
        goals = {}
        for key, (chart, target, t_hest) in HYP_REF[name].items():
            m, rb, ra, clip = median_rel(chart, target, t_hest, X)
            goals[key] = m
            lines.append(f"[{name}] hyperbolic {key} {chart} at t={t_hest}: median |dh|/|h| {m:.4f}, reading {rb:.3f} -> {ra:.3f}, clipped {clip}")
        if name == "math500":
            goals["busemann"] = 0.060                                     # 0.0597 to three decimals
        for key, (chart, target) in PROBES[name].items():
            goal = goals[key]
            t = bisect_t(chart, target, X, goal)
            m, rb, ra, clip = median_rel(chart, target, t, X)
            lines.append(f"[{name}] Euclidean {key} {chart}: t = {t:.4f} -> median |dh|/|h| {m:.4f} (goal {goal:.4f}); "
                         f"reading {rb:.3f} -> {ra:.3f}; clipped at rho_max {clip}")
            cfg = [{"kind": "chart", "chart": f"data/checkpoints/{chart}", "target": target, "t": round(t, 4), "rho": 1.0, "K": 1,
                    "mode": "reroll", "ent_thr": 1.0, "retrigger": True, "refractory": 1}]
            with open(out_path("configs", f"euclid_{key}_{name}.json"), "w", encoding="utf-8") as fo:
                json.dump(cfg, fo, indent=1)
    txt = "\n".join(lines)
    print(txt)
    with open(out_path("euclid_calibration.txt"), "w", encoding="utf-8") as fo:
        fo.write(txt + "\n")
    print("EUCLID_CALIB_DONE")


if __name__ == "__main__":
    main()
