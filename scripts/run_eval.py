"""Decode one or more runs of a config in one process (one model load) and append one result row per problem.

Usage: python scripts/run_eval.py --config configs/main/qwen1.5b_math500_busemann_t0.5.json --data math500 --tag _c3 --max-new 3072
A config is a JSON list of runs.  Keys:
  kind      none (greedy) | chart (geodesic step of the hyperbolic probe) | vec (fixed vector, all tokens) |
            vecgate (fixed unit vector at gate hits)
  mode      last (greedy) | reroll (entropy gate with cache rollback) | allgen (every generated position) |
            all (last prompt position and every generated position)
  chart     probe file (relative to the repository root); target ideal = Busemann readout, centre = radius readout
  t         geodesic step length (negative = away from the ideal point); rho = clip of |dh|/|h| (1.0 = no clipping)
  ent_thr   gate threshold theta (nats) on the next-token entropy; retrigger / refractory as in the paper (True / 1)
  trigger   entropy (default) | boundary (fire on tokens containing a double newline)
  vec       vector file; mult (multiplier of the raw mean-difference vector) or alpha_rel (|dh| / median |h|);
            rel = |dh|/|h| of the vecgate step
Result rows go to outputs/<run name><tag>.jsonl (idx, gold, preds, ok, n_tokens, fire_count, fire_steps, texts, ...);
the run name is derived from the config content by arm_name(); a run is resumable (problems already in the file are
skipped).  Environment: see hest/decoding.py.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hest.config import DATA, OUT_DIR, resolve  # noqa: E402
from hest.decoding import LMB  # noqa: E402


def arm_name(s):
    rho_def = 1.0 if s.get("kind") == "chart" else 0
    n = f"{s['kind']}_r{s.get('rho', rho_def)}_K{s.get('K',5)}_{s.get('mode','last')}"
    if s.get("mode") == "reroll":                      # suffix only when a parameter deviates from the defaults
        if float(s.get("ent_thr", 0.7)) != 1.0:
            n += f"e{s.get('ent_thr', 0.7)}"
        if not s.get("retrigger"):
            n += "_once"
        if s.get("retrigger") and int(s.get("refractory", 1)) != 1:
            n += f"r{s['refractory']}"
        if s.get("trigger") == "boundary":
            n += "_bnd"
    if s["kind"] == "chart":
        n += f"_t{s.get('t', 1.0)}_" + s.get("target", "ideal") + "_" + os.path.basename(s["chart"]).replace("entchart_", "").replace(".pt", "")
    if s["kind"] == "vec":
        n += (f"_m{s['mult']}_" if s.get("mult") is not None else f"_a{s['alpha_rel']}_") + os.path.basename(s["vec"]).replace("caa_", "").replace(".pt", "")
    if s["kind"] == "vecgate":
        n += f"_rel{s['rel']}_" + os.path.basename(s["vec"]).replace("caa_", "").replace(".pt", "")
    return n


def run_arm(lm, s, ids, probs, tag, out_dir, max_new=1024):
    name = arm_name(s)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}{tag}.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["idx"] for l in open(out_path, encoding="utf-8") if l.strip()}
    sp = out_path.replace(".jsonl", ".spec.json")                 # every result file carries its full configuration
    if not os.path.exists(sp):
        json.dump(s, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    todo = [i for i in ids if i not in done]
    print(f"=== RUN {name}{tag}: {len(todo)} to do ({len(done)} cached)", flush=True)
    t00 = time.time()
    for i in todo:
        p = probs[i]
        q, gold = p["problem"], p["gold"]
        t0 = time.time()
        a = lm.state(q)
        lm.field = None
        lm.vec_rel, lm.vec_log = None, []
        deltas = None
        K = s.get("K", 5)
        if s["kind"] == "chart":                            # state-dependent probe direction (hest/field.py), zeros = placeholder
            from hest.field import ChartField
            lm.field = ChartField(resolve(s["chart"]), s.get("target", "ideal"), t=float(s.get("t", 1.0)),
                                  rho_max=float(s.get("rho", 1.0)), iters=int(s.get("iters", 4)))
            if lm.field.layer is None:
                print(f"WARNING chart {lm.field.name}: training layer unknown, steering layer {lm.layer} unverified", flush=True)
            elif int(lm.field.layer) != int(lm.layer):
                raise ValueError(f"chart {lm.field.name} was fitted on layer {lm.field.layer} but LMB steers layer {lm.layer}")
            deltas = torch.zeros(K, lm.model.config.hidden_size, device="cuda")
        elif s["kind"] == "vec":                                # all-token steering with one fixed vector
            vd = torch.load(resolve(s["vec"]), map_location="cpu", weights_only=False)
            if s.get("mult") is not None:                       # multiplier x the RAW mean-difference vector
                alpha = float(s["mult"]) * float(vd["meandiff_norm"])
            else:                                               # relative displacement: alpha_rel = |dh| / median|h| of the training states
                alpha = float(s["alpha_rel"]) * float(vd["hnorm_med"])
            deltas = (vd["v"].float() * alpha).cuda()[None, :].repeat(K, 1)
        elif s["kind"] == "vecgate":                            # fixed unit direction at gate hits only (reroll); the hook scales it to rel * |h|
            vd = torch.load(resolve(s["vec"]), map_location="cpu", weights_only=False)
            lm.vec_rel = float(s["rel"])
            deltas = vd["v"].float().cuda()[None, :].repeat(K, 1)
        elif s["kind"] != "none":
            raise ValueError(f"unknown kind {s['kind']}")
        if lm.field is not None:
            lm.field.log.clear()
        texts, ntok = lm.gen_batch(q, deltas, K=K, mode=s.get("mode", "last"), max_new=max_new,
                                   ent_thr=s.get("ent_thr", 0.7), retrigger=bool(s.get("retrigger", False)),
                                   refractory=int(s.get("refractory", 1)), trigger=s.get("trigger", "entropy"))
        fl = list(getattr(lm.field, "log", []) or []) if lm.field is not None else []
        vl = list(lm.vec_log)
        preds, oks = lm.judge_batch(q, texts, gold)
        torch.cuda.empty_cache()
        row = {"idx": i, "level": p.get("level"), "gold": gold, "preds": preds, "ok": oks, "n_tokens": ntok,
               "fire_step": lm.fire_step, "fire_count": lm.fire_count, "fire_steps": lm.fire_steps,
               "logp": [round(x, 4) for x in lm.last_logp], "ent": [round(x, 4) for x in lm.last_ent],
               "dose": None if deltas is None else [round(float(d.norm() / a.norm()), 4) for d in deltas],
               "a_norm": round(float(a.norm()), 2), "sec": round(time.time() - t0, 1), "texts": texts,
               "field": None if not fl else {"n": len(fl),
                                             "read_before": round(sum(x[0] for x in fl) / len(fl), 4),
                                             "read_after": round(sum(x[1] for x in fl) / len(fl), 4),
                                             "d_hyp": round(sum(x[2] for x in fl) / len(fl), 4),
                                             "rho": round(sum(x[3] for x in fl) / len(fl), 4)},
               "vec": None if not vl else {"n": len(vl),                       # vecgate: per-fire checks
                                           "rel_med": round(statistics.median(x[0] for x in vl), 5),
                                           "cos_min": round(min(x[1] for x in vl), 5),
                                           "rel_real_med": round(statistics.median(x[2] for x in vl), 5),
                                           "cos_real_med": round(statistics.median(x[3] for x in vl), 5)}}
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"arm": name, "idx": i, "ok": oks, "ntok": ntok, "sec": row["sec"]}), flush=True)
    print(f"RUN_DONE {name}{tag} {time.time()-t00:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config file (relative to the repository root) or inline JSON")
    ap.add_argument("--ids", default="all", help="all | id-list json file")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--tag", default="")
    ap.add_argument("--data", default="math500", choices=list(DATA))
    ap.add_argument("--out_dir", default=OUT_DIR, help="directory of the result rows (default outputs/)")
    args = ap.parse_args()
    sp = args.config if os.path.exists(args.config) else resolve(args.config)
    spec = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else json.loads(args.config)
    probs = json.load(open(DATA[args.data], encoding="utf-8"))
    for p in probs:                                                    # the GSM8K files use "question"
        p.setdefault("problem", p.get("question"))
    if args.ids == "all":
        ids = list(range(len(probs)))
    else:
        ip = args.ids if os.path.exists(args.ids) else resolve(args.ids)
        ids = sorted(json.load(open(ip, encoding="utf-8")))
    ids = ids[args.start: (args.start + args.limit) if args.limit else None]
    lm = LMB()
    for s in spec:
        run_arm(lm, s, ids, probs, args.tag, args.out_dir, args.max_new)
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
