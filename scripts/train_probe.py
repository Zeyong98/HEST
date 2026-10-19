"""Hyperbolic probe trained on TOKEN-LEVEL ENTROPY.

The label is the model's own next-token entropy at each generated position.  A state that precedes a hesitant token must read
HIGHER than a state that precedes a confident one.

    L_rank = mean relu( s(z_i) - s(z_j) + 0.1 )     over pairs with  H_i < H_j - DELTA
    L_trip = mean relu( d(z_a, z_p) - d(z_a, z_n) + 0.1 )   p = same solution, |dstep| <= 5;  n = another problem
    s = hyperbolic radius r(z)  |  Busemann b_xi(z) with a learnable ideal point xi
No regularisers, learnable curvature; encoder MLP d->1024->128 -> exp0.

Usage: python scripts/train_probe.py --data outputs/states/ent_states_<model>_<tag>.pt --layer 19 --score busemann|radius
       [--geometry euclid]   (the Euclidean probe of the geometry ablation: same encoder, loss and data, kappa = 0)
Output: data/checkpoints/entchart_{score}<tag>_L{layer}.pt  (Euclidean probes: data/checkpoints/entchart_euclid_{score}<tag>_L{layer}.pt)
"""
import argparse
import re
import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hest.probe import ChartV3, DEV, radius, dist  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=None)
ap.add_argument("--layer", type=int, default=19)
ap.add_argument("--score", default="radius", choices=["radius", "busemann"])
ap.add_argument("--geometry", default="hyp", choices=["hyp", "euclid"], help="probe space: hyperbolic (default) or Euclidean (kappa = 0)")
ap.add_argument("--epochs", type=int, default=1500)
ap.add_argument("--delta", type=float, default=0.3, help="min entropy gap (nats) for a ranking pair")
ap.add_argument("--npairs", type=int, default=400_000)
ap.add_argument("--ntrip", type=int, default=200_000)
ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--val_frac", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

path = args.data or os.path.join(ROOT, "outputs", "states", "ent_states_Qwen2.5-Math-1.5B-Instruct_train100.pt")
path = path if os.path.isabs(path) else os.path.join(ROOT, path)
d = torch.load(path, map_location="cpu", weights_only=False)
acts = d["acts"][args.layer].float()
H, prob, step = d["ent"], d["prob"], d["step"]
print(f"{path}: layer {args.layer}, {acts.shape[0]} states from {len(set(prob.tolist()))} problems, "
      f"H median {float(H.median()):.3f}, frac>1 {float((H > 1).float().mean()):.3f}", flush=True)

rng = np.random.default_rng(args.seed)
probs = sorted(set(prob.tolist())); rng.shuffle(probs)
val_p = set(probs[: int(len(probs) * args.val_frac)])
tr_m = torch.tensor([int(p) not in val_p for p in prob.tolist()]); va_m = ~tr_m
mu = acts[tr_m].mean(0, keepdim=True); sd = acts[tr_m].std(0, keepdim=True).clamp_min(1e-4)


def pools(mask, seed):
    """sampled ranking pairs (low-entropy i, high-entropy j) and same-problem triplets, both problem-aware."""
    g = np.random.default_rng(seed)
    idx = np.where(mask.numpy())[0]
    h = H.numpy()[idx]; p = prob.numpy()[idx]; s = step.numpy()[idx]
    n = len(idx)
    lo, hi = [], []
    while len(lo) < args.npairs:                                     # rejection sampling, half within-problem
        a = g.integers(0, n, args.npairs); b = g.integers(0, n, args.npairs)
        same = g.random(args.npairs) < 0.5
        b = np.where(same & (p[a] == p[b]), b, b)                    # (within-problem pairs arise naturally too)
        ok = h[a] < h[b] - args.delta
        lo += list(a[ok]); hi += list(b[ok])
        if not ok.any():
            break
    lo, hi = np.array(lo[: args.npairs]), np.array(hi[: args.npairs])
    order = np.argsort(np.lexsort((s, p)))                           # positions grouped by problem
    by_p = {}
    for k, pp in enumerate(p):
        by_p.setdefault(int(pp), []).append(k)
    keys = [k for k, v in by_p.items() if len(v) >= 3]
    a_l, p_l, n_l = [], [], []
    for _ in range(args.ntrip):
        pk = keys[g.integers(len(keys))]; v = by_p[pk]
        ai = v[g.integers(len(v))]
        near = [q for q in v if q != ai and abs(s[q] - s[ai]) <= 5]
        if not near:
            continue
        pj = near[g.integers(len(near))]
        p2 = keys[g.integers(len(keys))]
        while p2 == pk:
            p2 = keys[g.integers(len(keys))]
        nk = by_p[p2][g.integers(len(by_p[p2]))]
        a_l.append(ai); p_l.append(pj); n_l.append(nk)
    T = lambda x: torch.as_tensor(np.asarray(x), device=DEV)
    return {"rank": (T(lo), T(hi)), "trip": (T(a_l), T(p_l), T(n_l)), "n": n}


tr = pools(tr_m, args.seed); va = pools(va_m, args.seed + 1)
Xtr = ((acts[tr_m] - mu) / sd).to(DEV); Xva = ((acts[va_m] - mu) / sd).to(DEV)
print(f"train {tr['n']} states, {tr['rank'][0].numel()} pairs, {tr['trip'][0].numel()} triplets; val {va['n']} states", flush=True)

torch.manual_seed(args.seed); gen = torch.Generator().manual_seed(args.seed)
model = ChartV3(args.geometry, args.score, hid=1024, d_in=acts.shape[1], d_out=args.dim).to(DEV)
par = [{"params": model.mlp.parameters(), "lr": 1e-3}, {"params": [model.raw_kappa], "lr": 1e-2}]
if args.score == "busemann":
    par.append({"params": [model.raw_xi], "lr": 1e-2})
opt = torch.optim.AdamW(par, weight_decay=0.01)
read = (lambda z, k: radius(z, k)) if args.score == "radius" else (lambda z, k: model.busemann(z))


def auroc(sc, y):
    o = np.argsort(sc); rk = np.argsort(o); n1, n0 = y.sum(), (1 - y).sum()
    return float((rk[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


best, ev = None, max(100, args.epochs // 15)
Hva = H[va_m].numpy()
for ep in range(args.epochs):
    model.train()
    i = torch.randint(0, tr["rank"][0].numel(), (4096,), generator=gen).to(DEV)
    j = torch.randint(0, tr["trip"][0].numel(), (4096,), generator=gen).to(DEV)
    v = model.encode(Xtr); rms = v.norm(dim=-1).pow(2).mean().sqrt()
    with torch.no_grad():
        model.scale.mul_(0.99).add_(0.01 * rms.detach())
    z = model.embed(v, rms); kap = model.kappa(); s = read(z, kap)
    lo, hi = tr["rank"][0][i], tr["rank"][1][i]
    loss = F.relu(s[lo] - s[hi] + 0.1).mean()
    a, p, n = (t[j] for t in tr["trip"])
    loss = loss + F.relu(dist(z[a], z[p], kap) - dist(z[a], z[n], kap) + 0.1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if (ep + 1) % ev == 0 or ep == args.epochs - 1:
        model.eval()
        with torch.no_grad():
            zv = model.embed(model.encode(Xva)); kv = model.kappa(); sv = read(zv, kv)
            pair = (sv[va["rank"][0]] < sv[va["rank"][1]]).float().mean().item()
            au = auroc(sv.cpu().numpy(), (Hva > 1.0).astype(float))
            sp = float(np.corrcoef(np.argsort(np.argsort(sv.cpu().numpy())), np.argsort(np.argsort(Hva)))[0, 1])
            yn = float((zv * (-kv).sqrt()).norm(dim=-1).median())
        if best is None or pair > best["val"]["pair_acc"]:
            best = {"epoch": ep + 1, "val": {"pair_acc": round(pair, 4), "auroc_H1": round(au, 4), "spearman": round(sp, 4)},
                    "state_dict": copy.deepcopy(model.state_dict()), "kappa": float(kv)}
        print(f"ep{ep+1} loss {float(loss):.3f} val_pair {pair:.4f} AUROC(H>1) {au:.4f} spearman {sp:.3f} "
              f"kappa {float(kv):+.3f} |y| med {yn:.3f}", flush=True)

out_dir = os.path.join(ROOT, "data", "checkpoints"); os.makedirs(out_dir, exist_ok=True)
b = os.path.basename(path)
tag = "_gsm" if "gsm" in b else ("_oly" if "oly" in b else "")
mtag = "_llama" if "Llama" in b else ("_q7" if "Math-7B" in b else "")   # model tag; Qwen2.5-Math-1.5B charts carry no model tag
atag = "_all" if re.search(r"all(_L\d+)?\.pt$", b) else ""                # per-layer state files give charts with an _all suffix
tag = mtag + tag + atag
gtag = "euclid_" if args.geometry == "euclid" else ""                  # Euclidean probes get their own file names
out = os.path.join(out_dir, f"entchart_{gtag}{args.score}{tag}_L{args.layer}" + (f"_d{args.dim}" if args.dim != 128 else "") + ".pt")
torch.save({"state_dict": best["state_dict"], "mu": mu, "sd": sd, "geometry": args.geometry, "score": args.score,
            "hid": 1024, "d_in": int(acts.shape[1]), "d_out": args.dim, "kappa": best["kappa"], "best": best["epoch"],
            "val": best["val"], "val_problems": sorted(val_p), "labels": path, "layer": args.layer}, out)
print(f"saved {out}  best@{best['epoch']} {best['val']}  ENT_CHART_DONE", flush=True)
