# Reproducing the paper

The full pipeline, the commands behind every table, and the mapping from the paper's terms to the code. The quick start and the installation are in [README.md](README.md).

## Environment variables and output files

`CX_MODEL` (model id) and `CX_LAYER` (0-based decoder block whose output is steered: 19 for the Qwen models, 13 for
Llama; must equal the probe's training layer) select the model and the layer. `CX_MEMFRAC` is the fraction of GPU
memory a process may use (default 0.72; the 1.5B model fits two processes at `CX_MEMFRAC=0.45`, the 7B and 8B models
run one process at a time), `CX_ATTN` the attention implementation (default `sdpa`) and `JUDGE_TIMEOUT` the seconds
per grading call (default 20).

The two quick-start runs of the README write `outputs/chart_r1.0_K1_reroll_t0.5_ideal_busemann_L19_c3.jsonl` and
`outputs/none_r0_K1_last_c3.jsonl` (one row per problem: `idx`, `gold`, `texts`, `preds`, `n_tokens`, `fire_count`,
`fire_steps`, ...); the table script grades both and prints accuracy, fixed / broken counts and fallbacks. Runs are
resumable: running the same command with `--ids all` afterwards decodes only the remaining problems.

## Data

`data/math500_test.json` (500), `data/gsm8k_test.json` (1319), `data/olympiadbench_test.json` (675; the
100 / 575 split in `data/ids/oly_split.json` and `oly_test575.json`), `data/math_train2000.json` and
`data/gsm8k_train.json` (probe training problems: 100 per dataset, chosen inside `scripts/collect_states.py`; dev
sets in `data/ids/dev_*.json`).  All are public datasets stored as JSON lists with `problem` / `question`, `gold` and
`level` fields.

## Full pipeline

Each step is one command (Qwen2.5-Math-1.5B on MATH-500 shown; `--tag` is the label of the result files, see the
tag list in the next section):

```
# 1. training states: greedy solutions of 100 training problems, hidden states of the steered layer, next-token entropies
CX_MODEL=Qwen/Qwen2.5-Math-1.5B-Instruct CX_LAYER=19 python scripts/collect_states.py --data math --n 100 --max-new 1024 --layers "19" --tag train100
# 2. probe (Busemann readout; --score radius for the radius readout; --geometry euclid for the Euclidean probe of the geometry ablation)
python scripts/train_probe.py --data outputs/states/ent_states_Qwen2.5-Math-1.5B-Instruct_train100.pt --layer 19 --score busemann
# 3. greedy baseline on the evaluation set
CX_MODEL=Qwen/Qwen2.5-Math-1.5B-Instruct CX_LAYER=19 python scripts/run_eval.py --config configs/greedy/greedy.json --ids all --data math500 --tag _c3 --max-new 3072
# 4. gate threshold: teacher-forced entropies of the greedy outputs, then the L1 profile rule (theta = 1.0 for the 1.5B model)
CX_MODEL=Qwen/Qwen2.5-Math-1.5B-Instruct CX_LAYER=19 python scripts/calibrate_threshold.py --tag c3 --data math500 --ids data/ids/math_all500.json --include-capped --dump outputs/c3_probe_all.npz
python scripts/calibrate_threshold.py --npz outputs/c3_probe_all.npz --ref math
# 5. steered run
CX_MODEL=Qwen/Qwen2.5-Math-1.5B-Instruct CX_LAYER=19 python scripts/run_eval.py --config configs/main/qwen1.5b_math500_busemann_t0.5.json --ids all --data math500 --tag _c3 --max-new 3072
# 6. grading and table
python scripts/analysis/table1_main.py --tags c3 --judge unified --safeguard
```

Step 1 for the other settings: `--data gsm8k --tag gsm100` (GSM8K), `--data olympiad --max-new 3072 --tag oly100`
(OlympiadBench, the training half of the fixed split), `--max-new 2048 --layers "13"` with the Llama model
(`CX_LAYER=13`), `--max-new 2048` with the 7B model, and `--layers all --per-layer --tag train100all` for the
layer ablation (per-layer files `..._train100all_L<L>.pt`).  The probe file name follows from the state file
(`_gsm`, `_oly`, `_llama`, `_q7`, `_all` tags) and the layer: `data/checkpoints/entchart_<score><tag>_L<layer>.pt`.
Training uses seed 42, 1500 epochs and a 20% problem-level validation split (the released probes use these defaults).
Step 4 for the other models: `--tag l8 --ids data/ids/math_all500.json` etc.; the chosen thresholds are Llama 2.0
(MATH-500) / 1.4 (GSM8K) and Qwen-7B 1.0 / 0.8 (already written into the configs).  The CAA vectors
(`scripts/build_caa.py`) need the six training-state files of the three models.  `scripts/make_splits.py --oly`
rebuilds the OlympiadBench split from the greedy lengths; `--dev` the dev sets.

## Reproducing the paper

Result-file tags (kept in the file names because the table scripts key on them): `c3` = Qwen2.5-Math-1.5B on
MATH-500, `g8` = 1.5B on GSM8K, `oly` / `oly2` = 1.5B on OlympiadBench (greedy on all 675 problems / steered runs on
the 575 evaluation problems), `l8` / `l8g` = Llama-3.1-8B on MATH-500 / GSM8K, `q7` / `q7g` = Qwen2.5-Math-7B on
MATH-500 / GSM8K, `abl` = ablations (1.5B, MATH-500 or GSM8K).  The result file name is derived from the config
content (`run_eval.arm_name`) plus the tag, so a config can be run with any tag.

| paper | configs | commands |
|---|---|---|
| Table 1 (1.5B; 3 datasets x 2 readouts x 4 step lengths) | `configs/main/qwen1.5b_{math500,gsm8k,olympiad}_{busemann,radius}_t{0.1,0.25,0.5,1.0}.json` | greedy + arms with `--data math500 --tag _c3`, `--data gsm8k --tag _g8`, `--data olympiad --ids data/ids/oly_test575.json --tag _oly2` (greedy: `--ids all --tag _oly`); then `python scripts/analysis/table1_main.py --tags c3 g8 --judge unified --safeguard` and `python scripts/analysis/table1_main.py --tags oly --arm-tag oly2 --split data/ids/oly_split.json --judge unified --safeguard` |
| Tables 10 and 11 (Appendix J: hesitation-count groups and hesitation density) | the Table 1 / Table 2 runs | `scripts/calibrate_threshold.py ... --include-capped --dump outputs/<tag>_probe_all.npz` for each greedy run, then `python scripts/analysis/appendix_regimes.py --bins greedy --fine` (Table 10; written to `outputs/tables/TABLE_S1_greedy_fine.txt`) and `--bins density` (Table 11) |
| Table 2 (Llama-3.1-8B and Qwen2.5-Math-7B) | `configs/main/{llama8b,qwen7b}_{math500,gsm8k}_{busemann_t0.5,busemann_t0.25,radius_t0.25}.json` | `CX_MODEL=meta-llama/Llama-3.1-8B-Instruct CX_LAYER=13 ... --tag _l8` / `_l8g`; `CX_MODEL=Qwen/Qwen2.5-Math-7B-Instruct CX_LAYER=19 ... --tag _q7` / `_q7g`; then `python scripts/analysis/table2_models.py` |
| Table 3(a, b) and Appendix M (every token, fixed direction, reverse direction, step boundaries, theta scan, layers 9 / 25) | `configs/ablation/*.json` except `euclid_*` | 1.5B, `--tag _abl` (`--data gsm8k` for the `*_gsm8k` configs; `CX_LAYER=9` / `25` for `layer9.json` / `layer25.json`); then `python scripts/analysis/table3_ablation.py`; the step-size column: `CX_MODEL=... CX_LAYER=19 python scripts/analysis/appendix_step_size.py` |
| Appendix: gate positions vs reasoning-step boundaries | greedy `_c3` run and `outputs/c3_probe_all.npz` | `python scripts/analysis/appendix_boundaries.py` |
| Table 3(c) and Appendix N (Table 16): Euclidean probes, the limit c -> 0 of the probe | `configs/ablation/euclid_{busemann,radius}_{math500,gsm8k}.json` | probes: step 2 with `--geometry euclid` (and `--score busemann` / `radius`) on the MATH and GSM8K training states; step lengths: `python scripts/match_step_size.py` (needs the hyperbolic radius probes too; the resulting values are already in the configs); runs: 1.5B, `--tag _c3` for the `*_math500` configs and `--data gsm8k --tag _g8` for the `*_gsm8k` configs, next to the Table 1 rows; then `python scripts/analysis/appendix_euclid.py`; edit diversity: `CX_MODEL=Qwen/Qwen2.5-Math-1.5B-Instruct CX_LAYER=19 python scripts/analysis/appendix_euclid_diversity.py` |
| Appendix I (Table 9): CAA baseline (6 cells) and exposure comparison | `configs/caa/{qwen1.5b,llama8b,qwen7b}_{math500,gsm8k}.json` | `python scripts/build_caa.py`, then the six runs with the tags above, then `python scripts/analysis/appendix_caa.py` |

Probes needed: the released `data/checkpoints/entchart_busemann_L19.pt` and `entchart_busemann_gsm_L19.pt` cover the
Busemann rows of Table 1 on MATH-500 / GSM8K and all ablations except the layer ablation; the radius probes, the
OlympiadBench probes, the Llama and 7B probes, the layer-9 / layer-25 probes and the four Euclidean probes of the geometry
ablation (`--geometry euclid`) are trained with steps 1-2 above
(the configs reference them by their default names).  Every table script accepts `--results_dir` (default
`outputs/`).  Grading is done from scratch by `hest/evaluation/judge_unified.py` (a local cache
`outputs/judge_cache_unified.json` speeds up repeated runs).

## Glossary

| paper | code |
|---|---|
| hyperbolic probe | `chart` (`kind: chart`; files `data/checkpoints/entchart_*.pt`; class `ChartV3` in `hest/probe.py`) |
| Busemann readout | `target: ideal` (Busemann function to a learnable ideal point) |
| radius readout | `target: centre` (hyperbolic distance to the origin) |
| entropy gate with cache rollback | `mode: reroll` (`hest/decoding.py`) |
| gate threshold theta (nats) | `ent_thr` |
| geodesic step length t | `t` (negative = the reverse-direction ablation) |
| re-armable gate, refractory 1 | `retrigger: true`, `refractory: 1` |
| step-boundary trigger (ablation) | `trigger: boundary` |
| every-token ablation (no gate) | `mode: allgen` |
| fixed-direction ablation (CAA direction at the gate, 6% of the state norm) | `kind: vecgate`, `rel: 0.06` |
| CAA baseline (all-token steering, multiplier 1) | `kind: vec`, `mode: all`, `mult: 1` |
| greedy baseline | `kind: none` (`configs/greedy/greedy.json`) |
| Euclidean probe (c -> 0, geometry ablation) | `geometry: euclid` in the probe file (`scripts/train_probe.py --geometry euclid`; files `data/checkpoints/entchart_euclid_*.pt`); kappa = 0, step z' = z + (t/2) u_hat, linear Busemann readout -<xi, z>, radius readout 2 abs(z) |
| runaway safeguard / fallback | a steered decode that reaches the 3072-token cap reports the greedy answer (`--safeguard`; `fb` = number of such problems) |
| hesitation points of a solution | tokens of the greedy solution whose teacher-forced next-token entropy exceeds theta (`outputs/<tag>_probe_all.npz`) |
| unified grading protocol | `hest/evaluation/judge_unified.py` (`uni_score`: boxed / explicit marker / the model's own restated answer / last number; tiered equivalence) |
| restated answer (`preds` in the result rows) | the answer the model restates from its own solution at decode time (`judge_batch` in `hest/decoding.py`) |
