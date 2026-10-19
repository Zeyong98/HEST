"""Gate-threshold calibration (label-free; no answer labels are used).

Step 1 (GPU): teacher-forced next-token entropies of the already decoded greedy solutions of a model on a dataset,
one forward pass per problem, dumped as an npz (e{idx} = entropy at each generated position, ids, ok) and summarised
as the share of solutions with 0-10 / 11-24 / 25+ hesitation points at several thresholds:
    CX_MODEL=meta-llama/Llama-3.1-8B-Instruct CX_LAYER=13 python scripts/calibrate_threshold.py --tag l8 --data math500 \\
        --ids data/ids/math_all500.json --include-capped --dump outputs/l8_probe_all.npz
    (the greedy rows outputs/none_r0_K1_last_<tag>.jsonl must exist; --results_dir points elsewhere if needed)
Step 2 (CPU): the L1 profile-matching rule on such a dump, after excluding degenerate solutions (more than `--junk`
of the tokens above 2 nats): the threshold whose 0-10 / 11-24 / 25+ profile is closest (L1) to the reference profile
of Qwen2.5-Math-1.5B-Instruct (72/18/10 on MATH-500, 76/19/4 on GSM8K):
    python scripts/calibrate_threshold.py --npz outputs/l8_probe_all.npz --ref math
The same dumps give the hesitation-count groups of Table 10 (scripts/analysis/appendix_regimes.py).
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hest.config import DATA, OUT_DIR, resolve  # noqa: E402


def dump_entropies(a):
    import torch
    from hest.decoding import LMB
    dump = a.dump or os.path.join(OUT_DIR, f"{a.tag}_probe_all.npz")
    os.makedirs(os.path.dirname(dump), exist_ok=True)
    probs = json.load(open(DATA[a.data], encoding="utf-8"))
    for p in probs:
        p.setdefault("problem", p.get("question"))
    ids = sorted(json.load(open(a.ids if os.path.exists(a.ids) else resolve(a.ids))))
    rows = {json.loads(l)["idx"]: json.loads(l) for l in open(os.path.join(a.results_dir, f"none_r0_K1_last_{a.tag}.jsonl"), encoding="utf-8") if l.strip()}
    lm = LMB(); lm.active = False
    per = {}; lens = []
    with torch.no_grad():
        for i in ids:
            if i not in rows: continue
            if not a.include_capped and rows[i]["n_tokens"][0] >= a.cap: continue      # degenerate tails distort the profile
            pid = lm.ids(lm.chat(probs[i]["problem"])); P = pid.shape[1]
            gen = lm.tok(rows[i]["texts"][0], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            full = torch.cat([pid, gen], 1)
            # chunked teacher forcing with a KV cache: never materialises full-sequence attention or full-vocab logits
            ents, past, pos = [], None, 0
            while pos < full.shape[1]:
                chunk = full[:, pos: pos + 512]
                out = lm.model(input_ids=chunk, past_key_values=past, use_cache=True)
                past = out.past_key_values
                lo = max(P - 1, pos); hi = min(P - 1 + gen.shape[1], pos + chunk.shape[1])
                if hi > lo:
                    lsm = torch.log_softmax(out.logits[0, lo - pos: hi - pos].float(), -1)
                    ents.append((-(lsm.exp() * lsm).sum(-1)).cpu().numpy())
                del out
                pos += chunk.shape[1]
            del past
            e = np.concatenate(ents)
            per[i] = e; lens.append(len(e))
            torch.cuda.empty_cache()
    E = np.concatenate(list(per.values()))
    print(f"{a.tag} ({a.data}): {len(per)} test problems, solution tokens median {np.median(lens):.0f} mean {np.mean(lens):.0f}; "
          f"H quantiles 90/95/98/99: {np.percentile(E, [90, 95, 98, 99]).round(3).tolist()}")
    okv = np.array([bool(rows[i]["ok"][0]) for i in per], float)
    print(f"greedy accuracy on these {len(per)}: {okv.mean():.3f}" + ("" if a.include_capped else "  (capped solutions excluded)"))
    np.savez_compressed(dump, ids=np.array(list(per.keys())), ok=okv, **{f"e{i}": e for i, e in per.items()})
    print(f"per-solution entropies dumped to {dump}")
    print(f"{'thr':>5s} {'%tok':>6s} {'mean/sol':>9s} {'median':>7s} | {'0-10 share':>10s} {'acc':>5s} | {'11-24 share':>11s} {'acc':>5s} | {'25+ share':>9s} {'acc':>5s}")
    for t in [float(x) for x in a.thr.split()]:
        c = np.array([(e > t).sum() for e in per.values()])
        b0, b1, b2 = c <= 10, (c >= 11) & (c <= 24), c >= 25
        f = lambda m: f"{okv[m].mean():5.2f}" if m.any() else "  -  "
        print(f"{t:5.2f} {100 * (E > t).mean():6.2f} {c.mean():9.1f} {np.median(c):7.0f} | {b0.mean():10.0%} {f(b0)} | {b1.mean():11.0%} {f(b1)} | {b2.mean():9.0%} {f(b2)}")
    print("PROBE_DONE")


def rebin(a):
    REF = {"math": (72, 18, 10), "gsm": (76, 19, 4)}[a.ref]
    z = np.load(a.npz if os.path.exists(a.npz) else resolve(a.npz))
    ids = z["ids"]; ok = z["ok"]
    E = {int(i): z[f"e{int(i)}"] for i in ids}
    junk = np.array([(E[int(i)] > 2.0).mean() > a.junk for i in ids])
    print(f"{a.npz}: {len(ids)} solutions; degenerate by the >{a.junk:.0%}-tokens-above-2-nats rule: {junk.sum()} "
          f"(idx {sorted(int(i) for i, j in zip(ids, junk) if j)}), their accuracy {ok[junk].mean() if junk.any() else float('nan'):.2f}; "
          f"accuracy on the rest {ok[~junk].mean():.3f}")
    keep = [int(i) for i, j in zip(ids, junk) if not j]
    okk = np.array([ok[k] for k, i in enumerate(ids) if not junk[k]], float)
    Ecat = np.concatenate([E[i] for i in keep])
    print(f"H quantiles 90/95/98/99 on the clean set: {np.percentile(Ecat, [90, 95, 98, 99]).round(3).tolist()}")
    print(f"{'theta':>6s} {'%tok':>6s} {'mean/sol':>9s} {'median':>7s} | {'0-10':>5s} {'acc':>5s} | {'11-24':>5s} {'acc':>5s} | {'25+':>5s} {'acc':>5s} | {'L1 to ' + '/'.join(map(str, REF)):>14s}")
    best = None
    for t in [float(x) for x in a.rebin_thr.split()]:
        c = np.array([(E[i] > t).sum() for i in keep])
        b0, b1, b2 = c <= 10, (c >= 11) & (c <= 24), c >= 25
        s = (100 * b0.mean(), 100 * b1.mean(), 100 * b2.mean())
        l1 = sum(abs(x - r) for x, r in zip(s, REF))
        f = lambda m: f"{okk[m].mean():5.2f}" if m.any() else "  -  "
        print(f"{t:6.2f} {100 * (Ecat > t).mean():6.2f} {c.mean():9.1f} {np.median(c):7.0f} | {s[0]:4.0f}% {f(b0)} | {s[1]:4.0f}% {f(b1)} | {s[2]:4.0f}% {f(b2)} | {l1:14.0f}")
        if best is None or l1 < best[1]:
            best = (t, l1)
    print(f"CHOSEN theta = {best[0]} (L1 {best[1]:.0f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None, help="step 2: apply the L1 rule to this dump (CPU only)")
    ap.add_argument("--ref", choices=["math", "gsm"], default="math", help="step 2: reference profile")
    ap.add_argument("--junk", type=float, default=0.20, help="step 2: exclude solutions with more than this share of tokens above 2 nats")
    ap.add_argument("--rebin-thr", default="0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.8 2.0", help="step 2: candidate thresholds")
    ap.add_argument("--tag", default=None, help="step 1: tag of the greedy rows (none_r0_K1_last_<tag>.jsonl)")
    ap.add_argument("--data", default="math500", choices=list(DATA))
    ap.add_argument("--ids", default="data/ids/math_all500.json")
    ap.add_argument("--thr", default="1.0 1.4 1.5 1.6 1.7 1.8 1.9 2.0", help="step 1: thresholds of the printed profile")
    ap.add_argument("--include-capped", action="store_true", help="keep solutions that hit the cap (default: excluded)")
    ap.add_argument("--cap", type=int, default=3072)
    ap.add_argument("--dump", default=None, help="step 1: npz of per-solution entropies (default outputs/<tag>_probe_all.npz)")
    ap.add_argument("--results_dir", default=OUT_DIR, help="step 1: directory of the greedy rows")
    a = ap.parse_args()
    if a.npz:
        rebin(a)
    else:
        if not a.tag:
            ap.error("--tag is required for step 1 (or give --npz for step 2)")
        dump_entropies(a)


if __name__ == "__main__":
    main()
