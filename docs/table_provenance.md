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
| `corrected` | The manuscript value was wrong and has been corrected; see the corrections log (§8) |

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

The manuscript's 2025-11-03 test log loaded an epoch-15 checkpoint with
`best_score=0.7146202386775634`; the second file carries exactly that epoch and
score. The second file is therefore the run that produced the manuscript's test
metrics; the first is a later retrain. The 2026-10-21 re-evaluation under
protocol v2 is in `docs/original_model_v2_reproduction_20261021.md`.

`test_metrics.json` evaluated the first file on the test split:

| Metric | manuscript (Table 2) | `test_metrics.json` | status |
|---|---:|---:|---|
| AUC | 0.816 | 0.8174 (macro) | `reconciled` |
| MAE | 27.44 | 27.5467 (mL/h, positive-flow) | `reconciled` |
| Acc | 0.742 | 0.7365 | `reconciled` |
| R² | 0.820 | 0.8129 | `reconciled` |
| RMSE | 60.04 | 61.268 | `reconciled` |
| Recall | 0.813 | 0.7758 (macro) | `val-only` |
| **F1** | ~~0.743~~ → **0.696** | **0.6902 (macro)** | **`corrected`** |

The F1 was a transcription error, corrected in the manuscript (see §8). It
contradicted the same row's own recall and AUC (both macro) and Table 4's macro
F1 (0.696); the shipped code computes `macro` only, so 0.743 could not be
produced by it. See `tests/test_metrics_protocol.py`.

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
| BurnResu (Full) | 0.742 / 0.813 / **0.696** / 0.816 / 27.44 / 0.820 / 60.04 | see §2 | **`corrected`** (F1, §8) |

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
| Full | 0.816 / **0.696** / 27.44 / 60.04 | see §2 | **`corrected`** (F1, §8) |

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

## 8. Corrections log

Every change made to a value that appeared in the submitted manuscript. Nothing
is altered silently; each entry states what was wrong, what it was changed to,
and why that is the correct value.

### C1 — BurnResu F1 in Tables 2 and 3: 0.743 → 0.696

**Where.** Table 2 (`tab:comparison`), BurnResu (Full) row, F1 column; and
Table 3 (`tab:ablation`), BurnResu (Full) row, F1 column. Both carried the same
value, so the error propagated to two tables.

**What was wrong.** 0.743 is not a macro F1. The manuscript's own Table 4
reports per-class F1 of 0.752 / 0.727 / 0.609, whose macro average is 0.696 —
and Table 4 states that macro average explicitly. The same Table 2 row already
reports macro values for its other classification columns (AUC 0.816 and recall
0.813 both match Table 4's macro row), so the F1 column was the only cell in the
row computed on a different basis. The shipped code computes `macro` only, so it
cannot produce 0.743 at all. 0.743 does, however, appear twice elsewhere in the
same table as a *recall* value (Regression and SVM rows), which is consistent
with a transcription slip.

**Corrected to.** **0.696**, the macro average of the manuscript's own Table 4,
which keeps Tables 2, 3 and 4 mutually consistent.

**Why not 0.690.** `ckpts/test_metrics.json` records a macro F1 of 0.6902, but
for a *different run*: its per-class figures (crystalloid P 0.705 / R 0.772,
glucose P 0.539 / R 0.719) do not match the manuscript's Table 4 (0.685 / 0.836
and 0.492 / 0.801). Adopting 0.6902 would replace one inconsistency with
another, between Table 2 and Table 4.

**Nature of the change.** The value decreases. It is recorded here rather than
absorbed silently, and it is the kind of correction the response letter must
state explicitly.

**Affected claims.** The qualitative claim that BurnResu leads the comparison
weakens: at 0.696 its F1 sits just above Transformer (0.690) and CNN (0.688),
and the (Acc − F1) gap becomes 0.046 — in line with the other deep-learning
rows rather than an order of magnitude smaller. Whether that margin is
significant is exactly what #13 must establish; the corrected value must not be
presented as superiority without it.

### C2 — Table 1: respiratory rate repeated the heart-rate row

**Where.** Table 1 (`tab:demographics`), Vital Signs block.

**What was wrong.** The respiratory rate row read `103, (89,120)` — identical
to the heart-rate row above it. Recomputed from
`data/data_speed/vital_signs` over all 26136 planned hour positions, the
respiratory rate is **median 20, IQR (19, 21.5)**. A respiratory rate of 103
per minute is not physiologically survivable; the row had been copied.

**Corrected to.** `20, (19,21.5)` with missing rate 48.38%.

### C3 — Table 1: age IQR was truncated

**Where.** Table 1, Demographics block.

**What was wrong.** `45, (28,6)` — the upper quartile read as a single digit.
Recomputed from `data/baseline_processed.csv` (n=363), the age IQR is
**(28, 55)**; the `6` was the first digit of `55` with the second lost.

**Corrected to.** `45, (28,55)`.

### C4 — Table 1: the missing-rate column did not match the data

**Where.** Table 1, Vital Signs block.

**What was wrong.** The column is headed `Missing rate(%)` and held values
between 0.55 and 0.97. Neither reading fits: as a percentage it means under 1%
missing, as a fraction it means 55–97%, and the measured rates are 48.55% to
97.37%. Only CVP reconciled under either reading, which is why the error
survived — one row looked right.

**Corrected to.** Measured values over 26136 planned hour positions, with the
header spaced as `Missing rate (%)`:

| Variable | Was | Now |
|---|---:|---:|
| Heart Rate | 0.56 | 48.55 |
| SBP | 0.59 | 50.05 |
| DBP | 0.59 | 50.05 |
| Oxygen Saturation | 0.56 | 49.93 |
| CVP | 0.97 | 97.37 |
| Respiratory Rate | 0.55 | 48.38 |
| Temperature | 0.73 | 63.25 |

Urine output and oral intake still carry `--` in that column although both are
17.90% missing; filling them needs the denominator convention settled first.

### C5 — Table 1: fluid-type counts were wrong and mislabelled

**Where.** Table 1, Fluid Management block.

**What was wrong.** The three fluid rows read glucose 4742 (18.7%),
crystalloid 8942 (35.4%) and colloid 8190 (32.4%) — sums to 86.5%, which is the
total the reviewers questioned. Recomputed over the 25267 labelled hours, and
consistent with the multilabel protocol in `docs/label_protocol.md`:

| Class | Was | Now |
|---|---:|---:|
| Crystalloid | 8942 (35.4%) | **13236 (52.4%)** |
| Colloid | 8190 (32.4%) | **10275 (40.7%)** |
| Glucose | 4742 (18.7%) | **7235 (28.6%)** |

The three now sum to 121.7%, which is correct for a multilabel task — the three
types are independent and all three are absent in 6464 hours (25.6%). The old
figures summed to 86.5%, and the manuscript's reading of the remaining 13.5% as
"no fluid" does not follow (see `docs/label_protocol.md` §4).

### C6 — Table 1: header spacing

`Missing rate(\%)` → `Missing rate (\%)`.

### Corrections not yet made

The following were reported by reviewers and are **not** fixed here, because
each needs source material or a decision this ticket cannot supply:

- `Colloid Volume [mL]` carries a volume unit while its neighbours carry rates
  (R3-9). The source column has to be identified first.
- `Oral Intake [mL·kg⁻¹·h⁻¹] & 150, (80,260)` is implausible as a per-kg-per-hour
  rate; the unit is likely wrong but the raw field must be checked.
- The `Effect` and `p-value` columns are still undefined (R2-3e, R4-15); no test
  in the repository computes them.
- Figure 1 and Figure 2 typos ("Gerenation", "Vitals sign data", "Embeddig")
  live inside the PNG files, not the LaTeX; they need the figure sources, which
  are not in the working tree.
