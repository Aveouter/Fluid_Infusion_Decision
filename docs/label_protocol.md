# Target Label Protocol (v1)

> Historical six-column protocol for the existing dataset. For the rebuilt
> `final_v2_20261014` data, use [volume-first model data](fluid_volume_model_data.md).
> A recorded setting is not a delivered hourly average; do not run the new
> source settings through the legacy rate-to-volume conversion below.

Single source of truth for what BurnResu predicts. Machine-checkable definitions
live in [`utils/label_protocol.py`](../utils/label_protocol.py); the invariants are
enforced by [`tests/test_label_protocol.py`](../tests/test_label_protocol.py).
Anything in the manuscript's Methods describing the target must agree with this
document.

## 1. Column layout

Each hourly step is a 6-vector, in this exact order:

| Index | Field | Meaning | Unit |
|---|---|---|---|
| 0 | `crystal` | crystalloid infusion active | 0 / 1 |
| 1 | `colloid` | colloid infusion active | 0 / 1 |
| 2 | `water` | water / glucose infusion active | 0 / 1 |
| 3 | `crystal_rate` | crystalloid infusion rate | mL/h |
| 4 | `colloid_rate` | colloid infusion rate | mL/h |
| 5 | `water_rate` | water / glucose infusion rate | mL/h |

Chinese source columns: 晶体 / 胶体 / 水 and 晶体速度 / 胶体速度 / 水速度.

## 2. Multilabel, not multiclass

The three types are **independent**. Any subset may be active in the same hour —
including all three simultaneously. The task is therefore **multilabel**, and:

- the classification head must not use a softmax over the three types;
- accuracy / AUC / F1 must be reported with an explicit averaging scheme
  (per-class, macro, micro, or weighted) — see issue #7;
- per-class proportions **sum to more than 100 %** and must never be presented
  as if they partitioned the hours.

## 3. Core invariant

```
bit = 1  ⟺  rate > 0
```

The **rate is the authoritative field**. This matches `data2pkl.py`, which
rewrites the status bit from the recorded speed, and therefore matches the
pipeline that produced the reported results. A row with `bit = 0` and `rate > 0`
is a protocol violation and is rejected by `label_protocol.validate()`.

## 4. Zero flow ("no infusion")

There is **no separate "no infusion" class**. An hour with all three bits zero is
a no-infusion (zero-flow) hour. Consequences for reporting:

- Zero-flow hours must be counted directly (`zero_flow_mask`), not inferred.
- The remainder `1 − Σ(per-class proportion)` is **not** a zero-infusion share.
  Because the task is multilabel, that remainder is generally negative. The
  manuscript's current 86.5 % total for the three fluid types does not license
  the reading "the remaining ~14 % is no fluid" (issue #4, R3-8).

Rates are never back-filled across a zero-flow hour. An interruption followed by
a restart stays an interruption; carrying the previous rate forward would turn
"no infusion" into "continuous infusion" and silently destroy the zero-inflation
signal the loss is designed around.

## 5. Rates and volumes

The time axis is strictly hourly (`DT_HOURS = 1.0`), so

```
volume in hour t (mL)   = rate (mL/h) × 1 h
cumulative volume of c  = Σ_t  active[c,t] × rate[c,t] × Δt
```

Committing volume and rate are the same quantity here; **MAE reported in mL/h and
MAE reported in mL are different endpoints** and the manuscript must say which one
each table uses (issues #7, #12).

## 6. Aggregating records into an hour

When several source records land in the same hour (`aggregate_infusion_records`):

1. Records of the **same type in the same hour are summed** into that hour's
   volume. Nothing is dropped and no "first record wins" rule is used.
2. The hourly rate is that summed volume divided by the step length.
3. Records of **different types** stay independent (see §2).
4. A **bolus is not distinguished from a continuous infusion**: both collapse to
   the equivalent mean hourly rate. This is a known limitation of the source
   records, recorded here rather than silently assumed to be absent.
5. Hours with no record are zero-flow hours — they are not interpolated.

## 7. Sentinel values

`-1` in a status column means *not recorded* and normalises to `0`. After
normalisation every row must satisfy §3. `lint_data()` in
[`utils/data_loader.py`](../utils/data_loader.py) applies this transformation and
returns new arrays — it never modifies its input.

## 8. Known limitations to state in the manuscript

- The raw status column and the recorded speed carry **different semantics**: the
  former may mark "an intake record exists this hour" while the latter is the
  pump rate. This protocol resolves the conflict in favour of the rate; the
  alternative reading is not implemented and should be disclosed.
- Bolus events are indistinguishable from fast continuous infusion in the source
  data (§6.4).
- Whether the recorded speed is the *ordered* rate or the *delivered* rate has
  not been established from the source records.

## 9. Reporting checklist for Methods

- [ ] State that the task is multilabel over three independent fluid types.
- [ ] State that the status bit is derived from the recorded rate.
- [ ] State the rate unit (mL/h) and the step length (1 h).
- [ ] Report the zero-flow hour count and share directly.
- [ ] Name the averaging scheme for every classification metric.
- [ ] State clearly whether each MAE is instantaneous (mL/h) or cumulative (mL).

## Version history

| Version | Change |
|---|---|
| 1.0 | Initial protocol. Rate authoritative; no back-fill across zero-flow hours; multilabel semantics fixed. |
