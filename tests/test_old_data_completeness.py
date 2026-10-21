"""Synthetic tests for raw-hour coverage and endpoint eligibility.

These tests do not touch the real cohort.  They build the same small CSV
surface that ``load_series`` reads and check the four failure modes named in
the review: an interior missing hour with continuous row numbers, a missing
last 24 h, missing data that must not become zero, and a partial interval that
must never enter a cumulative endpoint.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.old_fluid_data_v1 import HORIZONS, build, interval_eligible, load_series

VITALS = ("体温", "心率", "脉搏", "呼吸", "收缩压", "舒张压", "CVP", "氧饱和度", "尿量", "口服.1")
BASELINE = ("性别", "年龄", "身高", "体重", "体重指数", "烧伤总面积",
            "合计Ⅲ°", "合计深Ⅱ°", "合计浅Ⅱ°", "吸入性损伤")


def _row(pid, hour, rate=100.0, heart_rate=80.0):
    row = dict(PID=pid, 时间=hour, 晶体=float(rate > 0), 胶体=0.0, 水=0.0,
               晶体速度=rate, 胶体速度=0.0, 水速度=0.0,
               晶体每小时入量=rate, 胶体每小时入量=0.0, 水每小时入量=0.0)
    row.update({name: 0.0 for name in VITALS})
    row["心率"] = heart_rate
    return row


def _write_cohort(tmp_path, hours_by_patient):
    (tmp_path / "data").mkdir()
    rows = []
    for pid, hours in hours_by_patient.items():
        for hour in hours:
            rows.append(_row(pid, hour))
    # A row whose time is empty and whose rate is a sentinel is unrecorded,
    # not a zero-flow hour.  It must disappear from the series.
    rows.append(_row(1, np.nan, rate=999.0))
    # A documented zero and a missing vital are different from no record.
    for row in rows:
        if row["PID"] == 1 and row["时间"] == 10:
            row["晶体速度"] = row["晶体每小时入量"] = 0.0
            row["晶体"] = 0.0
        if row["PID"] == 2 and row["时间"] == 5:
            row["心率"] = np.nan
    pd.DataFrame(rows).to_csv(tmp_path / "data/all_combined_data.csv", index=False)
    pd.DataFrame([{**{"Unnamed: 0": str(pid)}, **{name: 1.0 for name in BASELINE}}
                  for pid in hours_by_patient]).to_csv(
        tmp_path / "data/baseline_processed.csv", index=False)
    pd.DataFrame({"PID": [1]}).to_csv(tmp_path / "data/train_split.csv", index=False)
    pd.DataFrame({"PID": [2]}).to_csv(tmp_path / "data/test_split.csv", index=False)


@pytest.fixture()
def cohort(tmp_path):
    # Patient 1: interior hour 36 absent; later row numbers are still 1..71.
    # Patient 2: only hours 1..48, i.e. the last 24 h are absent.
    _write_cohort(tmp_path, {1: list(range(1, 36)) + list(range(37, 73)),
                             2: list(range(1, 49))})
    return tmp_path


def test_interior_gap_keeps_raw_hour_not_row_position(cohort):
    series, _ = load_series(cohort, expected_patients=2)
    time = series["1"]["time"]
    assert len(time) == 71
    assert time[0] == 1 and time[-1] == 72
    assert 36 not in time
    assert np.any(np.diff(time) == 2)
    rows, _, _ = build(cohort, expected_patients=2)
    patient_rows = rows["patient"] == 0
    at_37 = np.flatnonzero(patient_rows & (rows["hour"] == 37))[0]
    assert rows["hour"][at_37] == 37
    assert not rows["contiguous"][at_37]
    assert not rows["action_mask"][at_37].any()
    # The target at hour 34 has a six-hour future that crosses hour 36.
    at_34 = np.flatnonzero(patient_rows & (rows["hour"] == 34))[0]
    assert not rows["cum_mask"][at_34].any()
    # A target wholly after the gap is contiguous again.
    at_38 = np.flatnonzero(patient_rows & (rows["hour"] == 38))[0]
    assert rows["contiguous"][at_38]
    assert rows["cum_mask"][at_38, 0]  # six-hour horizon


def test_last_24h_missing_is_not_a_72h_endpoint(cohort):
    series, _ = load_series(cohort, expected_patients=2)
    time = series["2"]["time"]
    assert interval_eligible(time, 1, 24)
    assert interval_eligible(time, 1, 48)
    assert not interval_eligible(time, 1, 72)
    assert interval_eligible(time, 25, 48)
    assert not interval_eligible(time, 49, 72)
    rows, _, _ = build(cohort, expected_patients=2)
    patient_rows = rows["patient"] == 1
    assert rows["hour"][patient_rows].max() == 48
    # At the last observed origin no future horizon can be complete.
    last = np.flatnonzero(patient_rows)[-1]
    assert not rows["cum_mask"][last].any()


def test_missing_is_not_zero(cohort):
    series, _ = load_series(cohort, expected_patients=2)
    item = series["1"]
    assert item["length"] == 71
    assert np.isfinite(item["time"]).all()
    assert 999.0 not in item["rate"]          # empty-time row dropped
    assert 36 not in item["time"]
    assert (item["rate"][item["time"] == 10] == 0).all()  # documented zero kept
    # Missing vital is NaN, not 0.
    assert np.isnan(series["2"]["vitals"][series["2"]["time"] == 5, 1]).all()
    # A missing vital does not invalidate a setting endpoint.
    assert interval_eligible(series["2"]["time"], 1, 48, series["2"]["rate"])


def test_partial_interval_is_excluded_from_cumulative(cohort):
    series, _ = load_series(cohort, expected_patients=2)
    assert not interval_eligible(series["1"]["time"], 1, 72)
    assert interval_eligible(series["1"]["time"], 37, 72)
    rows, _, _ = build(cohort, expected_patients=2)
    # No row may claim a complete horizon that runs past the record or a gap.
    for i in np.flatnonzero(rows["patient"] == 0):
        for h_index, horizon in enumerate(HORIZONS):
            if not rows["cum_mask"][i, h_index]:
                assert np.isnan(rows["cum_target"][i, h_index]).all()
            else:
                origin = rows["hour"][i]
                assert interval_eligible(series["1"]["time"], origin + 1, origin + horizon,
                                         series["1"]["rate"])
    # Duplicate and non-integer stamps are not eligible.
    assert not interval_eligible([1, 2, 2, 3], 1, 3)
    assert not interval_eligible([1, 2, 2.5, 3], 1, 3)


def test_physiology_masks_keep_valid_and_hour(cohort):
    """History must blank oral/vitals, not the valid marker or hour feature."""
    _, joint, _ = build(cohort, expected_patients=2)
    _, history, _ = build(cohort, expected_patients=2, group="history")
    _, missing, _ = build(cohort, expected_patients=2, group="missing")
    _, values, _ = build(cohort, expected_patients=2, group="values")
    rows, _, _ = build(cohort, expected_patients=2)
    # Patient 2, target hour 10: the 24 h history contains hour 5, whose heart
    # rate is missing.  The last history row is hour 9.
    index = np.flatnonzero((rows["patient"] == 1) & (rows["hour"] == 10))[0]
    for arm in (joint, history, missing, values):
        window = arm["sequence"][index, -1]
        assert window[44] == 1.0       # valid marker preserved
        assert window[45] > 0.0        # raw hour feature preserved
    assert np.isnan(joint["sequence"][index, -5, 26])   # heart-rate value
    assert joint["sequence"][index, -5, 35] == 1.0      # missing indicator
    assert np.all(history["sequence"][index, :, 24:44] == 0.0)
    assert missing["sequence"][index, -5, 26] == 0.0
    assert missing["sequence"][index, -5, 35] == 1.0
    assert np.isnan(values["sequence"][index, -5, 26])
    assert values["sequence"][index, -5, 35] == 0.0
