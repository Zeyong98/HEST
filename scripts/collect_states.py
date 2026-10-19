"""Token-level entropy dataset for the hyperbolic probe.

For each of N MATH *training* problems (20 per level), greedy-decode once and record, at EVERY generated token:
  * the next-token entropy at that position (a property of the token, not of any layer: it comes from the logits)
  * the hidden state at several layers (the chart's input; layer-specific)
No rollouts, no correctness labels -- the label is the model's own entropy.

Output: outputs/states/ent_states_<model>_<tag>.pt
  acts   {L: [S, d] fp16}   ent [S]   prob [S]   step [S]   tok [S]   lens, ids
Usage: CX_MODEL=Qwen/Qwen2.5-Math-1.5B-Instruct CX_LAYER=19 python scripts/collect_states.py --n 100 --max-new 1024 --layers "19" --tag train100
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hest.config import DATA, OUT_DIR  # noqa: E402
from hest.decoding import LMB  # noqa: E402

TRAIN = DATA["mathtrain"]
GSM = DATA["gsmtrain"]
OLY = DATA["olympiad"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="problems in total (n/5 per level)")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--layers", default="8 10 12 16 20")
    ap.add_argument("--tag", default="train100")
    ap.add_argument("--data", default="math", choices=["math", "gsm8k", "olympiad"])
    ap.add_argument("--per-layer", action="store_true", help="one output file per layer (needed for --layers all)")
    args = ap.parse_args()
    layers = None if args.layers.strip() == "all" else [int(x) for x in args.layers.split()]

    if args.data == "olympiad":
        probs = json.load(open(OLY, encoding="utf-8"))
        picked = json.load(open(os.path.join(ROOT, "data", "ids", "oly_split.json"), encoding="utf-8"))["train"]
        print(f"{len(picked)} OlympiadBench problems, the fixed training half of data/ids/oly_split.json "
              f"(the other {675 - len(picked)} are never seen by the chart)", flush=True)
    elif args.data == "gsm8k":
        probs = json.load(open(GSM, encoding="utf-8"))
        for p in probs:
            p.setdefault("problem", p.get("question"))
        picked = sorted(np.random.RandomState(7).choice(len(probs), args.n, replace=False).tolist())
        print(f"{len(picked)} GSM8K train problems, randomly sampled (seed 7) from {len(probs)}", flush=True)
    else:
        probs = json.load(open(TRAIN, encoding="utf-8"))
        per = args.n // 5
        picked, seen = [], {f"Level {k}": 0 for k in range(1, 6)}
        for i, p in enumerate(probs):
            lv = str(p.get("level"))
            if seen.get(lv, per) < per:
                picked.append(i)
                seen[lv] += 1
            if len(picked) >= args.n:
                break
        print(f"{len(picked)} problems, per level {seen}", flush=True)

    lm = LMB(layer=(layers[0] if layers else 0))
    lm.active = False
    if layers is None:                                           # every decoder block
        layers = list(range(len(lm.model.model.layers)))
        print(f"saving all {len(layers)} layers", flush=True)
    acts = {L: [] for L in layers}
    ent, prob_id, step_id, toks_all, lens = [], [], [], [], []
    t0 = time.time()
    for n_done, i in enumerate(picked):
        pid = lm.ids(lm.chat(probs[i]["problem"]))
        with torch.no_grad():                                    # greedy decode, no cache reuse across problems
            out = lm.model(input_ids=pid, use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1]
            gen, ents = [], []
            for _ in range(args.max_new):
                lsm = torch.log_softmax(logits.float(), -1)
                ents.append(float(-(lsm.exp() * lsm).sum()))
                nxt = int(logits.argmax(-1))
                if nxt == lm.eot:
                    break
                gen.append(nxt)
                out = lm.model(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=past, use_cache=True)
                past, logits = out.past_key_values, out.logits[:, -1]
            del past, out
            full = torch.cat([pid, torch.tensor([gen], device="cuda")], 1) if gen else pid
            hs = lm.model(input_ids=full, output_hidden_states=True).hidden_states
            P = pid.shape[1]
            keep = len(gen)                                      # state at position P-1+t emits token t (t = 0..keep-1)
            for L in layers:
                acts[L].append(hs[L + 1][0, P - 1: P - 1 + keep, :].half().cpu())
            ent += ents[:keep]
            prob_id += [i] * keep
            step_id += list(range(keep))
            toks_all += gen
            lens.append(keep)
            del hs
        torch.cuda.empty_cache()
        if (n_done + 1) % 10 == 0:
            print(f"[{n_done + 1}/{len(picked)}] states {len(ent)} mean len {np.mean(lens):.0f} "
                  f"frac H>1 {np.mean(np.array(ent) > 1):.3f} {time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.join(OUT_DIR, "states"), exist_ok=True)
    out_path = os.path.join(OUT_DIR, "states", f"ent_states_{os.environ.get('CX_MODEL', 'model').split('/')[-1]}_{args.tag}.pt")
    meta = {"ent": torch.tensor(ent, dtype=torch.float32), "prob": torch.tensor(prob_id), "step": torch.tensor(step_id),
            "tok": torch.tensor(toks_all), "ids": picked, "lens": lens, "layers": layers}
    if args.per_layer:                                           # one file per layer: ..._<tag>_L<L>.pt (acts of that layer only)
        for L, v in acts.items():
            torch.save(dict(meta, acts={L: torch.cat(v)}), out_path.replace(".pt", f"_L{L}.pt"))
        torch.save(dict(meta, acts={}), out_path)                # index file: metadata only (ids, entropies, no activations)
        print(f"saved {len(acts)} per-layer files {out_path.replace('.pt', '_L<L>.pt')} + metadata {out_path}", flush=True)
    else:
        torch.save(dict(meta, acts={L: torch.cat(v) for L, v in acts.items()}), out_path)
    e = np.array(ent)
    print(f"saved {out_path}: {len(e)} states, layers {layers}", flush=True)
    print(f"entropy quantiles 10/25/50/75/90/99: {np.percentile(e, [10, 25, 50, 75, 90, 99]).round(3).tolist()}; "
          f"frac>0.5 {np.mean(e > .5):.3f} frac>1 {np.mean(e > 1):.3f}", flush=True)
    print("ENT_STATES_DONE", flush=True)


if __name__ == "__main__":
    main()
