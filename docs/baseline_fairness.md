# Baseline Fairness Audit

Issue #14. Reviewers 2 and 3 questioned whether the baselines in Table 2 were
trained and evaluated on the same footing as BurnResu — the symptom being that
regression, SVM and LSTM cluster near MAE 73 while CNN, Transformer and GBR sit
near 28–30. This document records what each baseline actually is, and every
asymmetry found.

Nothing here is a proposal; each row states what the code does today.

## 1. What each baseline is

| Model | Built by | Head / loss | Inputs | Tuned on |
|---|---|---|---|---|
| Regression, SVM, RF, XGBoost, GBR | `train_for_ml.py` | estimator directly on the rate vector; no ZILN head | last timestep's temporal features concatenated with baseline features | val |
| LSTM, CNN, Transformer, `base`, `flim` | `models/*` via `run_epoch.py` | classification logits (BCE) + ZILN regression head | full H+P window | val |
| BurnResu (`tcn_film`) | `models/tcn_film.py` via `run_epoch.py` | as above | full H+P window | val |
| Salinas et al. | external | rule-based | 24 h volume per kg per %TBSA | n/a |

## 2. Asymmetries found

### A1 — Different quantity, and a different mask (severe)

`run_epoch.py:426` states it outright: *"regression evaluation uses speed
(conditional mean, and only y_true>0 participates in MAE/RMSE/R/R2)"*. So the
deep-learning rows report `speed` — the regression head's conditional output —
over positive-flow cells only.

`train_for_ml.py` does the opposite: `flatten_loader` takes the last timestep's
rate vector for **every** window with no mask, and `eval_regression` computes
MAE over all of it.

Consequently Table 2's MAE column mixes two quantities and two denominators.
Measured on the aligned test set (see `docs/table_provenance.md`):

| Definition | BurnResu | Cells |
|---|---:|---:|
| `speed`, positive-flow mask — what the manuscript prints | 27.73 | 4382 |
| `rate`, positive-flow mask — the actual prediction | 81.70 | 4382 |
| `rate`, all cells — the convention `train_for_ml.py` uses | 78.15 | 10866 |

Neither column is comparable to the other, and the deep-learning rows are
favoured by both choices at once.

### A2 — Different F1 aggregation, but a consistent AUC (severe)

The two paths do not disagree everywhere, and the pattern explains how the
mismatch survived review:

| Column | `train_for_ml.py` | `run_epoch.py` | Comparable? |
|---|---|---|---|
| F1 | **micro** | **macro** | **no** |
| AUC | macro | macro | yes |

So Table 2's AUC column is on one scale, which makes the table look internally
consistent at a glance, while its F1 column compares micro scores in the
machine-learning half against macro scores in the deep-learning half. The
manuscript also declares macro F1 to be the primary classification metric, so
the declared metric is the one that is not what the baseline half reports.

### A3 — Baselines have no ZILN head and cannot express zero flow

The classical estimators regress the rate vector directly. They have no
zero-inflation component and no classification head, so they cannot be gated by
P(infusion active). Any statement that BurnResu's ZILN head is an advantage must
say what the head is compared against, and a like-for-like comparison would
require giving the baselines an equivalent head — which has not been done.

### A4 — Feature access differs

`flatten_loader` feeds non-sequence models the **last timestep only**, flattened
together with the baseline features. Sequence models receive the whole H+P
window. With `history_length=1` this happens to coincide, but the two paths are
not guaranteed to stay aligned if the history length changes.

Note also that `x_last` is taken from `x_ts`, which includes the label-derived
cumulative-volume columns. Both families therefore see them; this is not an
asymmetry, but it does mean every model in the table is scored on inputs that
carry cumulative label information.

### A5 — Salinas et al. is not the same task

The Salinas row (MAE 76.8) estimates a 24-hour volume in mL/kg/%TBSA from
baseline variables. BurnResu predicts an hourly rate in mL/h from longitudinal
data. The units differ, the horizon differs and the information set differs;
the row should be presented as context, not as a ranked competitor. Any ordering
that places it numerically beside the others is misleading.

## 3. What a fair table requires

- [ ] One rate definition for every row, named in the caption (`flow_pred`,
      `speed` or `rate` — see #12).
- [ ] One mask for every row, named in the caption.
- [ ] One F1 aggregation for every row, named in the caption.
- [ ] Hyper-parameter search space, budget and best-validation configuration
      recorded per baseline (the manifest in `utils/run_manifest.py` carries
      these fields).
- [ ] Salinas presented separately or explicitly marked non-comparable.
- [ ] A statement of what the ZILN head is being compared against (A3).

## 4. Automated guards

`tests/test_baseline_fairness.py` pins the structural facts above so that a
future change to either evaluation path fails loudly rather than silently
re-introducing a mixed table.
