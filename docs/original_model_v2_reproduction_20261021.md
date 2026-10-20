# Original BurnResu under evaluation protocol v2 (2026-10-21)

This note re-scores the **submitted BurnResu checkpoint** (epoch 15) with the
frozen evaluation protocol v2. It is a reproduction/re-analysis, not a new model
result.

- Protocol: `2.0` / `849a59c17baf8815`, `one_step_observed_history`, `main`
- Checkpoint: `ckpts/tcn_film_best_f1_0.7203_mae_21.3136.pth` (epoch 15,
  `best_score=0.7146202386775634`)
- Cohort: 55 patients, 3,622 windows, historical 15% test split
- Evidence: [original_model_v2_reproduction_20261021.json](original_model_v2_reproduction_20261021.json)

The split and test-window count reproduce exactly. The old headline numbers map
to the pooled positive-flow `speed` head, not to the action-decoded rate; the
patient-clustered paired intervals below are the basis for the statements.

## 1. What the old headline metrics measured

`run_epoch.py:345` computes MAE/R²/RMSE from `flow_pred`, not from
`rate = prob_cls * speed`. The v2 protocol registers these as separate rate
definitions in `utils/eval_protocol.py`.

| Metric | Old paper/log | v2 `speed` | v2 `rate` |
|---|---:|---:|---:|
| Positive-flow MAE, pooled (mL/h) | **27.44** | **27.4488** | 81.38 |
| Positive-flow MAE, patient mean (mL/h) | — | 30.0868 | 90.8361 |
| Positive-flow R² | **0.820** | **0.8213** | 0.2950 |
| Positive-flow SMAPE | — | — | 59.50 |
| Activity Macro-F1 | 0.696 (corrected) | — | 0.6894 |
| Strict action F1 (exploratory) | — | — | 0.2397 |
| 72 h cumulative dose MAE (mL) | — | 26,076 | 2,699.7 |
| 72 h cumulative dose bias (mL) | — | +26,076 | +2,247.5 |

The old values are a conditional positive-flow fit, not evidence that the model
delivers the right amount of fluid.

## 2. P1: cumulative dose (mL, patient mean)

| Method | 24 h | 48 h | 72 h | 72 h bias |
|---|---:|---:|---:|---:|
| `rate` | 1,693.3 [1,144.4, 2,338.8] | 2,036.8 [1,456.9, 2,683.5] | **2,699.7 [1,954.1, 3,519.0]** | **+2,247.5 [1,382.1, 3,200.8]** |
| persistence | 330.3 [240.5, 421.9] | 208.5 [156.3, 265.9] | **164.1 [118.4, 210.8]** | -164.1 |
| moving average | 645.1 [497.7, 796.8] | 423.2 [323.2, 542.1] | 322.5 [242.5, 409.8] | -322.5 |

Paired patient-cluster bootstrap (model − baseline; positive = model worse):

- `rate − persistence`, 72 h: **+2,535.5 mL [1,786.0, 3,354.2]**
- `rate − moving_average`, 72 h: **+2,377.2 mL [1,620.2, 3,182.2]**
- 24 h and 48 h intervals also exclude zero.

Per-fluid bias at 72 h: **water +2,354.0 mL**, crystal +43.7 mL, colloid
-150.2 mL. The cumulative failure is driven by the water channel.

## 3. P2: tolerance and zero-dose control

| Method | 24 h joint | 72 h joint | 24 h zero-dose accuracy |
|---|---:|---:|---:|
| `rate` | 0.125 [0.042, 0.229] | **0.085 [0.021, 0.170]** | **0/6** |
| `action_rate_uncond` | 0.104 [0.021, 0.188] | 0.106 [0.021, 0.192] | 5/6 |
| persistence | 0.354 [0.229, 0.500] | **0.702 [0.575, 0.830]** | 6/6 |
| moving average | 0.125 [0.042, 0.229] | 0.340 [0.213, 0.468] | 6/6 |

Paired 72 h: `rate − persistence` **-0.617 [-0.766, -0.468]**;
`rate − moving_average` **-0.255 [-0.404, -0.106]**. At 24 h the difference
against the moving average is 0.000 [-0.125, +0.125].

`rate = prob × speed` rarely reaches exact zero. The threshold-gated
`action_rate_uncond` can stop (5/6 zero-dose patients), but its 72 h P1 is
7,049.5 mL, so the decoder trade-off is real and must be reported explicitly.

## 4. P3: activity F1 and strict action F1

| Method | Macro-F1 | Micro-F1 | Strict action F1 | Event F1 | Predicted/true events |
|---|---:|---:|---:|---:|---:|
| `rate` | **0.6894 [0.6548, 0.7188]** | 0.6986 [0.6645, 0.7284] | **0.2397** | 0.4899 | 5,362 / 2,897 |
| `action_rate_uncond` | 0.6894 | 0.6986 | 0.3004 | 0.3407 | 1,917 / 2,897 |
| persistence | 0.6688 [0.6265, 0.7045] | 0.6818 [0.6397, 0.7174] | 0 (by definition) | 0 | 0 / 2,897 |
| moving average | 0.6854 [0.6490, 0.7183] | 0.6988 [0.6625, 0.7332] | 0.2329 | 0.5069 | 3,818 / 2,897 |

Paired P3: `rate − persistence` macro **+0.0206 [+0.0034, +0.0395]**;
`rate − moving_average` macro **+0.0040 [-0.0072, +0.0148]** (crosses zero).
The activity advantage is small and not robust to the moving-average baseline.

Strict action confusion for `rate` (classes: hold/start/stop/up/down):

| True \ predicted | hold | start | stop | up | down |
|---|---:|---:|---:|---:|---:|
| hold 7,969 | 4,630 | 1,965 | 0 | 4 | 1,370 |
| start 1,331 | 495 | 836 | 0 | 0 | 0 |
| stop 1,270 | 303 | 0 | **0** | 1 | 966 |
| up 110 | 42 | 0 | 0 | 2 | 66 |
| down 186 | 34 | 0 | 0 | 0 | 152 |

The smooth `rate` decoder predicts **no stop** for any of the 1,270 true stop
events (966 called down, 303 hold). `action_rate_uncond` predicts 83 stops, of
which 59 are correct, raising strict F1 to 0.3004.

## 5. Secondary diagnostics

| Metric | `rate` | persistence | moving average |
|---|---:|---:|---:|
| All-hours MAE (mL/h) | 78.93 | **61.53** | 65.46 |
| Positive-flow MAE (mL/h) | 90.84 | 90.33 | 97.38 |
| Zero-flow MAE (mL/h) | 81.55 | **51.55** | 54.56 |
| Positive-flow R² | **0.2950** | -0.2982 | 0.0049 |
| Change-point F1 (exact) | 0.4063 | **0.4386** | 0.4376 |
| Change-point F1 (±1 h) | 0.7751 | **0.9948** | 0.6775 |
| Unmatched true changes | 684 | **30** | 898 |
| No-change-hour MAE | 33.88 | **2.33** | 14.38 |
| False-change rate | 7.53% | **0%** | 12.61% |
| Direction concordance | **0.9778** | 0 (no action) | 0.6256 |
| Macro-AUC / Brier / ECE | 0.8103 / 0.1779 / **0.0715** | — | — |

The model's only clear regression advantage is positive-flow R²/SMAPE; it is
not better than persistence on positive-flow MAE, and it is worse on zero-flow,
cumulative dose, and change timing.

## 6. The same pattern in the later retrain

The 2025-12-13 epoch-11 retrain (the default `ckpts/tcn_film_best.pth`) gives
P1 72 h `rate` 3,215.2 mL, bias +2,785.7, P3 macro F1 0.6875, positive-flow MAE
90.14, macro-AUC 0.8079. The failure is structural, not a checkpoint accident.

## 7. Caveats

- Single historical 15% test split; not nested and not a generalization estimate.
- The old input pickle `data/output_data_final.pkl` no longer carries `label`;
  this re-run uses `data/output_data.pkl` and a 0-filled baseline. Against the
  original 2025-11-03 log the residual is about 0.05% (MAE 27.431 vs 27.445).
- Strict action F1 is exploratory: it applies the Final model's action classes to
  the old rate labels. It is not the Final model's 0.198 endpoint.
- The old paper's F1 0.743 is a transcription error; the corrected activity
  macro F1 is 0.696 (`docs/table_provenance.md`, correction C1).

## 8. Reproduction

The aggregate evidence is in
[original_model_v2_reproduction_20261021.json](original_model_v2_reproduction_20261021.json).
Patient-level predictions, the evaluation bundle and the checkpoint stay local
because they contain patient identifiers; they are not part of this repository.

Local commands used:

```bash
# 0. fill missing baseline fields (None -> 0) in a local copy
python - <<'PY'
import pickle
b = pickle.load(open('data/baseline.pkl', 'rb'))
out = {k: [0.0 if x is None else float(x) for x in v] for k, v in b.items()}
pickle.dump(out, open('.audit/eval_v2/baseline_filled.pkl', 'wb'))
PY

# 1. export the original checkpoint's test predictions
python predict_test_set.py \
  --ckpt ckpts/tcn_film_best_f1_0.7203_mae_21.3136.pth \
  --data_pkl data/output_data.pkl \
  --base_pkl .audit/eval_v2/baseline_filled.pkl \
  --thresholds 0.35 0.35 0.45 \
  --out .audit/eval_v2/test_original_epoch15.npz

# 2. build the aligned v2 bundle
python make_eval_bundle.py \
  --preds .audit/eval_v2/test_original_epoch15.npz \
  --data_pkl data/output_data.pkl --base_pkl .audit/eval_v2/baseline_filled.pkl \
  --out .audit/eval_v2/bundle_original_epoch15.npz

# 3. run the frozen engine
python evaluate_v2.py \
  --bundle .audit/eval_v2/bundle_original_epoch15.npz \
  --base_pkl .audit/eval_v2/baseline_filled.pkl \
  --methods speed,rate,action_rate_uncond,persistence,moving_average \
  --model_id BurnResu-original-epoch15 --run_id original_epoch15_20261021 \
  --expect_protocol_hash 849a59c17baf8815 \
  --out .audit/eval_v2/rows_original_epoch15.json \
  --summary .audit/eval_v2/summary_original_epoch15.md
```

The baseline pickle used for this run replaces `None` with 0 for the 205
affected patients; the original run's cleaned baseline is no longer present.
