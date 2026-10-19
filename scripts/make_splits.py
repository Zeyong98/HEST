"""The fixed OlympiadBench split and the dev sets (both label-free).

--oly   100 / 575 split of OlympiadBench for probe training vs evaluation.  The probe needs no answer labels, only
        the model's own next-token entropy, so the "training" half is a set of PROMPTS.  The split is made once,
        deterministically, and saved, so the 575 evaluation problems are fixed before any probe is trained on the
        other 100.  Strata: subject x answer format (Numerical vs other), allocated proportionally; within each cell
        the problems are ordered by the length of the GREEDY solution (a label-free difficulty proxy) and sampled
        systematically, so the two halves match in length.
            python scripts/make_splits.py --oly --greedy outputs/none_r0_K1_last_oly.jsonl   -> data/ids/oly_split.json
--dev   dev sets for choosing the step length t: the probe is trained on 100 training problems; t is chosen on a
        DIFFERENT 100 training problems (disjoint).  MATH: 20 per level from math_train2000, excluding the probe's
        100 (ids stored in the state file), deterministic.  GSM8K: 100 random from gsm8k_train (seed 11), excluding
        the probe's 100 (seed 7 sample, ids in the state file).
            python scripts/make_splits.py --dev --states_dir outputs/states   -> data/ids/dev_math100.json, dev_gsm100.json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hest.config import DATA, OUT_DIR, resolve  # noqa: E402

N_TRAIN = 100
IDS = os.path.join(ROOT, "data", "ids")


def oly_split(greedy_path):
    probs = json.load(open(DATA["olympiad"], encoding="utf-8"))
    glen = {}
    for line in open(greedy_path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            glen[r["idx"]] = r["n_tokens"][0]

    cells = {}
    for i, p in enumerate(probs):
        cells.setdefault((p["level"], p["answer_type"] == "Numerical"), []).append(i)

    # largest-remainder allocation so the cell sizes sum to exactly N_TRAIN
    raw = {k: len(v) * N_TRAIN / len(probs) for k, v in cells.items()}
    take = {k: int(v) for k, v in raw.items()}
    for k in sorted(cells, key=lambda k: -(raw[k] - take[k]))[: N_TRAIN - sum(take.values())]:
        take[k] += 1

    train = []
    for k, idxs in sorted(cells.items(), key=lambda kv: str(kv[0])):
        n = take[k]
        if n == 0:
            continue
        order = sorted(idxs, key=lambda i: (glen.get(i, 0), i))            # by greedy length, deterministic
        pos = np.linspace(0, len(order) - 1, n).round().astype(int)        # systematic, spans the whole range
        train += [order[j] for j in dict.fromkeys(pos.tolist())]
    train = sorted(set(train))
    test = sorted(set(range(len(probs))) - set(train))

    out = os.path.join(IDS, "oly_split.json")
    json.dump({"train": train, "test": test}, open(out, "w"), indent=0)
    print(f"saved {out}: train {len(train)}, test {len(test)}")
    for name, ids in (("train", train), ("test", test)):
        sub = {}
        for i in ids:
            sub[probs[i]["level"]] = sub.get(probs[i]["level"], 0) + 1
        L = np.array([glen.get(i, 0) for i in ids], float)
        num = np.mean([probs[i]["answer_type"] == "Numerical" for i in ids])
        print(f"  {name:5s} n={len(ids):3d}  " + " ".join(f"{k[:4]} {v}" for k, v in sorted(sub.items())) +
              f" | numerical {num:.2f} | greedy length median {np.median(L):.0f} mean {L.mean():.0f}")
    print("OLY_SPLIT_DONE")


def dev_sets(states_dir):
    import torch

    def used_ids(tag):
        pk = torch.load(os.path.join(states_dir, f"ent_states_Qwen2.5-Math-1.5B-Instruct_{tag}.pt"),
                        map_location="cpu", weights_only=False)
        return sorted(int(i) for i in pk["ids"])

    # ---- MATH ----
    math = json.load(open(DATA["mathtrain"], encoding="utf-8"))
    used = set(used_ids("train100"))
    dev, per = [], {f"Level {k}": 0 for k in range(1, 6)}
    for i, p in enumerate(math):
        lv = str(p.get("level"))
        if i in used or per.get(lv, 20) >= 20:
            continue
        dev.append(i); per[lv] += 1
        if len(dev) == 100:
            break
    assert not (set(dev) & used) and len(dev) == 100
    json.dump(sorted(dev), open(os.path.join(IDS, "dev_math100.json"), "w"))
    print(f"MATH dev: {len(dev)} problems, per level {per}, disjoint from the probe's {len(used)}")

    # ---- GSM8K ----
    gsm = json.load(open(DATA["gsmtrain"], encoding="utf-8"))
    used = set(used_ids("gsm100"))
    pool = [i for i in range(len(gsm)) if i not in used]
    dev = sorted(np.random.RandomState(11).choice(pool, 100, replace=False).tolist())
    assert not (set(dev) & used) and len(dev) == 100
    json.dump(dev, open(os.path.join(IDS, "dev_gsm100.json"), "w"))
    print(f"GSM8K dev: {len(dev)} problems (seed 11), disjoint from the probe's {len(used)}")
    print("DEV_SETS_DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oly", action="store_true", help="make the OlympiadBench split")
    ap.add_argument("--greedy", default=os.path.join(OUT_DIR, "none_r0_K1_last_oly.jsonl"), help="greedy rows of OlympiadBench (all 675 problems)")
    ap.add_argument("--dev", action="store_true", help="make the dev sets")
    ap.add_argument("--states_dir", default=os.path.join(OUT_DIR, "states"), help="directory of the probe training-state files")
    a = ap.parse_args()
    if not (a.oly or a.dev):
        ap.error("give --oly and/or --dev")
    if a.oly:
        oly_split(a.greedy if os.path.exists(a.greedy) else resolve(a.greedy))
    if a.dev:
        dev_sets(a.states_dir if os.path.isabs(a.states_dir) else resolve(a.states_dir))


if __name__ == "__main__":
    main()
