# Claim–Evidence Matrix

Issue #27. Every consequential claim in the manuscript, the evidence that
currently supports it, and what was done. Reviewers 1, 2, 3 and 4 all objected
to claims outrunning the evidence; this is the ledger that answers them.

Status vocabulary: `supported` (evidence in this repository backs it),
`corrected` (the claim was wrong and has been rewritten), `removed` (the claim
was withdrawn), `limited` (true only under conditions that must be stated).

## Abstract

| Claim | Status | Basis |
|---|---|---|
| "BurnResu **consistently outperformed** conventional sequence modeling approaches" | **removed** | #13: patient-level paired bootstrap gives ΔMAE +13.84 [−6.05, +32.02] against persistence and −1.44 [−11.88, +10.80] against the moving average, both crossing zero; on per-patient macro F1 persistence is **significantly better** (Δ −0.0281, CI [−0.0428, −0.0147]). There is no consistent outperformance to report. |
| "capture multi-scale physiological dynamics" | **corrected** | #16: `history_length=1`, so the model receives exactly one timestep. A dilated causal TCN has nothing to convolve over and self-attention degenerates to weighting a single token. |
| "zero-inflated log-normal output" | **limited** | #15: the ZILN likelihood is never called; `p0` is never trained as a probability and the head output never leaves [19.0, 887.8]. |
| "clinician-decision modeling study … not proven optimal treatment targets" | **supported** | Correctly framed, and the framing must be preserved in every revision. |

## Methods

| Claim | Status | Basis |
|---|---|---|
| "classification head … Tversky loss" | **corrected** | #17: no Tversky implementation exists anywhere in the codebase. The actual loss is BCE. |
| "positive-only log(1+x)–Huber loss" | **corrected** | #17: the actual loss is a plain L1 (`_reg_l1_pos_only`). |
| "change-point consistency constraints, total-variation penalties, pre-onset sparsity constraints" | **removed** | #17: none of the three exists — `L_cp`, `L_tv`, `L_pre` have zero occurrences. |
| "StepLR schedule (decay every 100 epochs)" | **limited** | True as written, but training ran 16 epochs so the rate never changed. The cosine warm restart listed under robustness was never active. |
| "temperature scaling … applied during validation" | **removed** | #23: `--temp_scale` is declared and never read. |
| "Focal loss" robustness analysis | **removed** | #17: `--use_focal` defaults to True, is written into the metadata, and is read by nothing. |
| "±200 mL and ±10% tolerance accuracy" as an endpoint | **corrected** | #12: the endpoint was never implemented (`--tol_abs_ml` default 50.0, never used). Now implemented in `utils/endpoints.py` and reported; on the open-loop rollout it is **3.3%**. |
| "All preprocessing was fitted exclusively on the training set" | **removed** | #19: per-patient imputation and scaling are fitted on each patient's full 72-hour series, which for test patients uses information after the prediction time. |
| "70%/15%/15% patient-level split" | **supported** | Verified against `main.py:439-441`; the split reproduces exactly (254/54/55, 3622 test windows). |

## Results and Discussion

| Claim | Status | Basis |
|---|---|---|
| "superior sensitivity–specificity balance and greater robustness to threshold selection" | **removed** | #13: no significant advantage on any endpoint. |
| "each component provides meaningful benefit" (ablation) | **removed** | #16: removing self-attention costs 1.53 MAE with the interval crossing zero; removing FiLM costs nothing measurable. Only the input-modality ablations matter. |
| "the temporal encoder contributes a modest but consistent improvement (AUC 0.816 vs 0.703)" | **limited** | #16: that comparison is between input modalities, not architectural components, and at `history_length=1` the temporal encoder has no sequence to encode. |
| "our Transformer-based approach" | **corrected** | The architecture is a TCN–Transformer hybrid; reviewer 2 flagged the mismatch. |
| "Clinical Effectiveness Evaluation" (section title) | **corrected** | Retitled "Agreement with Recorded Clinical Practice". Reviewer 2 asked for exactly this. |
| "within 40%" (formula comparison) | **limited** | #25: holds only for Day 1 crystalloid (35.8%) and colloid (40.4%). Total is 55.6% and 110.4%. The claim must name the period and the fluid. |
| Figures 5 and 6 evaluation mode | **limited** | #11: never stated. The two modes differ by 1.76× in MAE and the open-loop cumulative volume collapses to 40% of truth. |
| "the model adapts to changing haemodynamics" | **removed** | #10: 100% miss on the 1327 hours where the clinician stopped an infusion; #12: the predicted trajectory varies by 2% of the true trajectory. |

## Claims never made but required by reviewers

| Requirement | Status | Basis |
|---|---|---|
| Persistence / moving-average baselines | **supported** | #9: implemented, and the comparison is now in the record. |
| Patient-level confidence intervals | **supported** | #13. |
| Calibration | **supported** | #23: crystalloid and colloid well calibrated; glucose overconfident by 19.6 points. |
| Subgroup performance | **supported** | #22: larger burns do worse; the 21 patients with a missing inhalation-injury field carry the highest error. |
| Failure cases | **supported** | #22: 31% of test patients above 100 mL/h mean error. |
| Cohort accounting | **limited** | #24: 449 raw records, 363 in the pickle, 364 in the manuscript; 86 exclusions unexplained. |

## Reference audit (R4-16)

| Item | Finding |
|---|---|
| Tversky loss citation | Cites Tversky 1977 (*Features of similarity*). That is the origin of the Tversky **index**; the Tversky **loss** as used in segmentation is Salehi et al. 2017. Since the loss is not implemented at all (#17), the citation should be removed with the claim rather than corrected. |
| Time-series deep learning references | To be checked against the claims that remain after the edits above. |
| Related work coverage | Must add the AI Clinician line and offline RL for sepsis, plus recent burn-resuscitation ML and urine-output closed-loop work (R2-8). |
| Reference count | 22 entries; reviewer 2 called it short. |

## What is not claimed anywhere and must stay that way

The data support an evaluation of a model that reproduces recorded clinician
decisions. They do not support clinical effectiveness, optimal treatment,
improved outcomes, or superiority as a decision-support strategy. Reviewer 4
asked for this boundary a third time; it is the single most repeated request in
the review and the manuscript already states it in three places. Those
statements must survive every further edit.

## Manuscript edits made

Five edits, listed with the diff in the issue record:

1. Abstract: "consistently outperformed" replaced with the null result and the
   framing as a characterization study.
2. Methods: the loss description corrected to BCE + L1 + Huber-on-cumulative,
   with the unfitted zero-mass component stated as a limitation.
3. Methods section title: "Clinical Effectiveness Evaluation" → "Agreement with
   Recorded Clinical Practice".
4. Same section's opening: "validate the clinical effectiveness" → "characterise
   agreement with recorded practice".
5. Discussion: "Transformer-based" → "hybrid temporal-convolutional and
   attention-based".
