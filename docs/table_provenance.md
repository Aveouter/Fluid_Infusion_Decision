# Provenance Ledger for Tables 2–4

Every number printed in the manuscript's Tables 2–4, mapped to the artefact that
would have produced it and marked with an explicit status. Required by issue #8.

**Nothing here changes a published value.** Numbers that cannot be reproduced are
marked as such rather than adjusted, per the roadmap rule that failures and
non-significant results must be reported honestly.

## Status vocabulary

| Status | Meaning |
|---|---|
| `reconciled` | Shipped artefact exists and its recorded value is consistent with the manuscript (allowing a validation→test gap) |
| `val-only` | Only the validation metric survives in the artefact name; the test metric is not available |
| `inconsistent` | Shipped artefact exists but its value **cannot** be reconciled with the manuscript |
| `no-artefact` | No shipped checkpoint or record corresponds to this row |

## 1. Checkpoint naming convention

`train.py` writes `{model}_best_f1_{f1:.4f}_mae_{mae:.4f}.pth`, where `f1` and
`mae` are the **validation** metrics at the selected epoch. These are therefore
*validation* numbers, not the test numbers a results table reports. A test value
must come from a separate evaluation (`test_metrics.json`).

Consequence: a checkpoint filename **never** directly gives a table entry; the
gap between validation and test is exactly what several rows below turn on.

## 2. Full model (`tcn_film`)

Two files exist and they are **different runs**:

| File | epoch | best_f1 (val) | best_mae (val) |
|---|---:|---:|---:|
| `tcn_film_best.pth` | 11 | 0.718953 | 22.0863 |
| `tcn_film_best_f1_0.7203_mae_21.3136.pth` | 15 | 0.720300 | 21.3136 |

* `training_metadata.json` and `test_metrics.json` refer to the **first**.
* `inference.py`'s default `--ckpt` points at the **second**.
* The filename metrics belong to the **second**.

Which one produced the manuscript's results cannot be determined from the
artefacts alone and must be settled when the tables are regenerated (#26).

`test_metrics.json` evaluated the first file on the test split:

| Metric | manuscript (Table 2) | `test_metrics.json` | status |
|---|---:|---:|---|
| AUC | 0.816 | 0.8174 (macro) | `reconciled` |
| MAE | 27.44 | 27.5467 (mL/h, positive-flow) | `reconciled` |
| Acc | 0.742 | 0.7365 | `reconciled` |
| R² | 0.820 | 0.8129 | `reconciled` |
| RMSE | 60.04 | 61.268 | `reconciled` |
| Recall | 0.813 | 0.7758 (macro) | `val-only` |
| **F1** | **0.743** | **0.6902 (macro)** | **`inconsistent`** |

The F1 cannot be produced by the shipped code, which computes `macro` only, and
it contradicts the same row's own recall and AUC (both macro) and Table 4's
macro F1 (0.696). See `tests/test_metrics_protocol.py`.

## 3. Table 2 — comparative performance

| Row | Manuscript (Acc/Recall/F1/AUC/MAE/R²/RMSE) | Shipped artefact | Status |
|---|---|---|---|
| Regression | 0.723 / 0.743 / 0.670 / 0.763 / 73.03 / 0.233 / 124.0 | none dedicated (see SVM note) | `no-artefact` |
| Random Forest | 0.665 / 0.918 / 0.686 / 0.653 / 72.45 / 0.274 / 111.4 | none | `no-artefact` |
| SVM | 0.724 / 0.743 / 0.671 / 0.763 / 73.03 / 0.233 / 124.1 | `ml_svr_best_f1_0.6982_mae_60.8376.pth` (val) | `val-only` |
| XG-Boost | 0.694 / 0.890 / 0.699 / 0.645 / 71.45 / 0.326 / 107.6 | none | `no-artefact` |
| GBR | 0.727 / 0.761 / 0.672 / 0.763 / 30.62 / 0.759 / 69.42 | `ml_lr_gbr_best_f1_0.6993_mae_27.8438.pth` (val) | `val-only` |
| Salinas et al. | — / — / — / — / 76.8 / 0.18 / 129.7 | external work | `no-artefact` |
| LSTM | 0.736 / 0.768 / 0.687 / 0.798 / 73.86 / 0.286 / 119.6 | `lstm_new_best_f1_0.7136_mae_62.4062.pth` (val) | `val-only` |
| CNN | 0.747 / 0.781 / 0.688 / 0.803 / 27.78 / 0.810 / 61.59 | `cnn_best_f1_0.7213_mae_22.5112.pth` (val) | `val-only` |
| Transformer | 0.744 / 0.747 / 0.690 / 0.804 / 30.90 / 0.795 / 63.98 | `transformer_best_f1_0.7206_mae_25.0423.pth` (val) | `val-only` |
| BurnResu w/o Baseline Enc. | 0.499 / 0.921 / 0.652 / 0.735 / 74.32 / 0.272 / 120.8 | `ours_wo_baseline_best_...mae_27.9457.pth` (val) | **`inconsistent`** |
| BurnResu w/o Temporal Enc. | 0.633 / 0.908 / 0.629 / 0.703 / 72.42 / 0.392 / 110.5 | `ours_wo_temporal_best_...mae_55.1046.pth` (val) | `val-only` |
| BurnResu (Full) | 0.742 / 0.813 / 0.743 / 0.816 / 27.44 / 0.820 / 60.04 | see §2 | **`inconsistent`** (F1) |

Note on the ML rows: `train_for_ml.py` reports **micro** F1 while `run_epoch.py`
(used for every deep-learning row) reports **macro** F1. Unless the manuscript
converted them, the machine-learning and deep-learning rows of this table are not
on a common scale. This is the aggregation mismatch flagged by the reviewers and
is recorded in `tests/test_metrics_protocol.py`.

## 4. Table 3 — ablation

| Row | Manuscript (AUC/F1/MAE/RMSE) | Shipped checkpoint (val) | Status |
|---|---|---|---|
| w/o Temporal Enc. | 0.703 / 0.629 / 72.42 / 110.5 | f1 0.6471, mae 55.1046 | `val-only` |
| w/o Baseline Enc. | 0.735 / 0.652 / **74.32** / 120.8 | f1 0.7143, mae **27.9457** | **`inconsistent`** |
| w/o self-attention | 0.814 / 0.633 / **74.67** / 112.7 | f1 0.7217, mae **22.2084** | **`inconsistent`** |
| w/o TCN | 0.810 / 0.693 / 27.63 / 60.59 | f1 0.7207, mae 22.2542 | `val-only` |
| w/o FiLM | 0.806 / 0.683 / 28.88 / 64.12 | f1 0.7155, mae 22.5010 | `val-only` |
| Full | 0.816 / 0.743 / 27.44 / 60.04 | see §2 | **`inconsistent`** (F1) |

**The two `inconsistent` rows are the collapse that reviewer 3 queried.** For the
full model the manuscript's test MAE is 1.29× the checkpoint's validation MAE.
For `w/o self-attention` it is **3.36×** (22.21 → 74.67) and for
`w/o Baseline Enc.` **2.66×** (27.95 → 74.32), while `w/o TCN` is 1.24× and
`w/o Temporal Enc.` 1.31× — in line with the full model.

So on validation these two variants are indistinguishable from the full model,
yet on test they land in the 72–75 band that also contains the plain regression
baselines (Table 2: 73.03). Issue #16 must decide whether that is a genuine
generalisation collapse or a different run, and it is the reason the ablation
narrative cannot be defended as written.

## 5. Table 4 — per-class performance

| Class | Manuscript (P/R/F1/AUC) | `test_metrics.json` (same checkpoint as §2) | Status |
|---|---|---|---|
| Crystalloid | 0.685 / 0.836 / 0.752 / 0.794 | 0.705 / 0.772 / 0.737 / 0.795 | `reconciled` (AUC) |
| Colloid | 0.668 / 0.798 / 0.727 / 0.834 | 0.628 / 0.836 / 0.717 / 0.836 | `reconciled` (AUC) |
| Glucose | 0.492 / 0.801 / 0.609 / 0.820 | 0.539 / 0.719 / 0.616 / 0.822 | `reconciled` (AUC) |
| Macro Average | 0.615 / 0.812 / 0.696 / 0.816 | 0.624 / 0.776 / 0.690 / 0.817 | `reconciled` |

The per-class AUCs match to within 0.003 and the macro average to within 0.007,
so Table 4 is consistent with the shipped evaluation path. The precision/recall/F1
differences are consistent with the two runs having selected different decision
thresholds (the shipped run used `[0.45, 0.30, 0.55]`, and the colloid value
violated the declared `--min_thresholds` floor of 0.40 — see #6).

## 6. What has to be regenerated

Four fixes landed during this remediation and each changes results, so **every
number above must be regenerated before resubmission**:

| Fix | Issue |
|---|---|
| Training augmentation no longer permutes/warps the window (36% of training windows were affected, leaking target segments into the input) | #5 |
| `RobustScaler` now fits on un-augmented training windows, and no longer over just the first 2000 windows | #6 |
| `--min_thresholds` / `--prec_floor` are now enforced (previously dead configuration) | #6 |
| Per-patient preprocessing fitted on the full 72 h series — **unresolved**, see the local record | #5 |

The shipped checkpoints were produced under the old configuration. They remain
useful for provenance (as in §2 and §4) but cannot be used as the basis of a
resubmission.

## 7. Reproducing a row

```bash
pip install -r requirements-dev.txt

# validation metrics live in the checkpoint filename; test metrics need a run:
python main.py --model tcn_film --mode test --device cuda:0 \
    --pkl_ts_path data/output_data.pkl --pkl_base_path data/baseline.pkl
```

A run manifest (`utils/run_manifest.py`) records git SHA, data fingerprints,
seeds, split, hyper-parameters and parameter count so that the resulting numbers
can be traced back. `training_metadata.json` as shipped lacks git SHA, data
fingerprints and split information, so it does not satisfy the manifest schema
on its own.
