"""CAA baseline directions: on each model's 100 training-problem states (the same states that trained the probe), split tokens by the model's gate threshold -- positive = entropy > theta (hesitant), negative = entropy <=
theta -- and take the mean difference.  We steer TOWARD confidence, so the stored unit vector is
    v = mean(h | H <= theta) - mean(h | H > theta),  normalised,
and a run adds alpha * v to the steering-layer state at EVERY generated token (fixed direction, all tokens).
alpha is expressed as a relative displacement r = |dh| / median|h| over the training tokens, so the same r means the
same relative push on every model (as t does for the chart step).
Writes data/checkpoints/caa_{model}_{data}.pt = {"v": unit vector [d], "hnorm_med": float, "layer": L, "theta": theta,
"n_pos": int, "n_neg": int, "cos_busemann_step": cosine to the mean chart displacement at gate states (diagnostic)}.
Usage: python scripts/build_caa.py [--states_dir outputs/states]  (needs the six training-state files)"""
import os, sys, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hest.config import CHECKPOINTS, OUT_DIR  # noqa: E402
R = sys.argv[sys.argv.index("--states_dir") + 1] if "--states_dir" in sys.argv else os.path.join(OUT_DIR, "states")
R = R if os.path.isabs(R) else os.path.join(ROOT, R)
CK = CHECKPOINTS
os.makedirs(CK, exist_ok=True)
CFG = [  # tag, states file, layer, theta, busemann chart (diagnostic)
    ("q15_math", "ent_states_Qwen2.5-Math-1.5B-Instruct_train100.pt", 19, 1.0, "entchart_busemann_L19.pt"),
    ("q15_gsm", "ent_states_Qwen2.5-Math-1.5B-Instruct_gsm100.pt", 19, 1.0, "entchart_busemann_gsm_L19.pt"),
    ("l8_math", "ent_states_Llama-3.1-8B-Instruct_train100_L13.pt", 13, 2.0, "entchart_busemann_llama_L13.pt"),
    ("l8_gsm", "ent_states_Llama-3.1-8B-Instruct_gsm100_L13.pt", 13, 1.4, "entchart_busemann_llama_gsm_L13.pt"),
    ("q7_math", "ent_states_Qwen2.5-Math-7B-Instruct_train100_L19.pt", 19, 1.0, "entchart_busemann_q7_L19.pt"),
    ("q7_gsm", "ent_states_Qwen2.5-Math-7B-Instruct_gsm100_L19.pt", 19, 0.8, "entchart_busemann_q7_gsm_L19.pt"),
]
for tag, sf, L, thr, chart in CFG:
    d = torch.load(os.path.join(R, sf), map_location="cpu", weights_only=False)
    X = d["acts"][L].float(); H = d["ent"].float()
    pos, neg = H > thr, H <= thr
    v = X[neg].mean(0) - X[pos].mean(0)
    vn = v.norm(); vu = v / vn
    hn = X.norm(dim=1); hmed = float(hn.median())
    cos = float("nan")
    try:                                                   # diagnostic: does CAA point where the chart step points?
        from hest.field import ChartField
        f = ChartField(os.path.join(CK, chart), target="ideal", t=0.5, rho_max=1.0)
        idx = torch.where(pos)[0][:200]
        D = f(X[idx].cuda()).cpu()
        cos = float(torch.nn.functional.cosine_similarity(D.mean(0), vu, dim=0))
        cos_each = torch.nn.functional.cosine_similarity(D, vu[None, :].expand_as(D), dim=1)
        cos_med = float(cos_each.median())
    except Exception as e:  # noqa: BLE001
        cos_med = float("nan"); print("  chart diagnostic skipped:", repr(e)[:80])
    torch.save({"v": vu, "hnorm_med": hmed, "layer": L, "theta": thr, "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
                "meandiff_norm": float(vn), "cos_busemann_step": cos, "cos_busemann_step_median": cos_med},
               os.path.join(CK, f"caa_{tag}.pt"))
    print(f"{tag:9s} L{L} theta {thr}: pos {int(pos.sum()):6d} neg {int(neg.sum()):6d} | |mean diff| {float(vn):7.2f} vs median|h| {hmed:7.1f} "
          f"({float(vn) / hmed:.3f}) | cos(CAA, mean chart step) {cos:+.3f}, per-state median {cos_med:+.3f}")
print("CAA_DIRS_DONE")
