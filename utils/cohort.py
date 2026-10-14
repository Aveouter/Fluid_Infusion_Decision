"""队列入组、零时点与选择偏倚（issue #24，对应 R4-14）。

背景
----
审稿人 4 要求：明确定义**零时点**（受伤、入院、转诊/转院、院前输液），
清楚报告因**非 ICU 入院或记录不完整**而排除的患者并讨论选择偏倚，
说明**儿科**患者的纳排标准。

现状（可核实的事实）
--------------------
- 原始护理记录目录有 **449** 个患者文件；
- 进入建模的 pkl 有 **363** 位患者 —— **86 位在某一环节被排除，原因未记录**；
- 论文写的是 **364 名患者**，与 pkl 的 363 不符；
- 数据审计统计 **62 位患者 < 18 岁**，论文未说明儿科纳排标准。

本模块提供**人数守恒**的排除台账与零时点对齐，使这些数字可以被逐项核对，
而不是只留一个最终数。
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable

#: 允许的排除原因。新增原因必须显式登记，避免"其它"成为黑洞。
EXCLUSION_REASONS = (
    "non_icu_admission",
    "incomplete_records",
    "paediatric",
    "no_fluid_records",
    "duplicate_patient",
    "unparseable_time",
)

#: 论文声称的队列规模与 pkl 实际规模。
PAPER_COHORT_N = 364
PICKLE_COHORT_N = 363

#: 原始记录目录中的患者文件数。
RAW_RECORD_N = 449


def cohort_ledger(candidates: Iterable[str],
                  exclusions: Dict[str, str]) -> Dict[str, object]:
    """人数守恒的排除台账。

    ``exclusions`` 形如 ``{patient_id: reason}``。返回含各原因计数与最终人数的
    字典，并**校验守恒**：候选数 == 排除数 + 最终数。

    排除原因必须取自 :data:`EXCLUSION_REASONS`；未知原因直接报错，
    避免"其它"变成无法审计的黑洞。
    """
    cands = [str(c) for c in candidates]
    if len(set(cands)) != len(cands):
        raise ValueError("候选名单存在重复编号")

    unknown = sorted({r for r in exclusions.values() if r not in EXCLUSION_REASONS})
    if unknown:
        raise ValueError(f"未知排除原因：{unknown}；请显式登记到 EXCLUSION_REASONS")

    not_candidate = sorted(set(exclusions) - set(cands))
    if not_candidate:
        raise ValueError(f"被排除者不在候选名单中：{not_candidate[:5]}")

    kept = [c for c in cands if c not in exclusions]
    counts = {r: 0 for r in EXCLUSION_REASONS}
    for r in exclusions.values():
        counts[r] += 1

    assert len(cands) == len(exclusions) + len(kept), "人数不守恒"
    return {
        "n_candidates": len(cands),
        "n_excluded": len(exclusions),
        "n_final": len(kept),
        "by_reason": {k: v for k, v in counts.items() if v},
        "final_ids": kept,
    }


def time_zero_offsets(injury: datetime, admission: datetime = None,
                      referral: datetime = None,
                      prehospital_fluids: bool = False,
                      horizon_hours: int = 72) -> Dict[str, object]:
    """把各事件时间映射到统一的 72 小时轴（以**受伤时间**为零点）。

    返回每个事件的相对小时数与**是否落在观察窗内**。院前时段（负的小时）
    不应被当作已观察：若院前已给液，第 0 小时的真实累计量并非从零开始，
    这一点必须显式标记而不是默认忽略。
    """
    if admission is not None and admission < injury:
        raise ValueError("入院时间早于受伤时间，零时点定义需要复核")

    def _hours(t):
        return None if t is None else (t - injury).total_seconds() / 3600.0

    h_adm = _hours(admission)
    h_ref = _hours(referral)
    return {
        "zero_point": "injury_time",
        "admission_hour": h_adm,
        "referral_hour": h_ref,
        "horizon_hours": horizon_hours,
        "admission_in_window": (h_adm is not None and 0 <= h_adm < horizon_hours),
        "referral_in_window": (h_ref is not None and 0 <= h_ref < horizon_hours),
        "prehospital_fluids": bool(prehospital_fluids),
        "cumulative_starts_at_zero": not bool(prehospital_fluids),
        "note": "院前已给液时第 0 小时的累计量不为零，不得默认按零起算",
    }


def eligibility(age: float, n_missing_baseline: int = 0,
                paediatric_cutoff: float = 18.0,
                max_missing_baseline: int = 10) -> Dict[str, object]:
    """纳排判定：**确定性**边界，儿科与记录完整性阈值均可核对。

    年龄缺失（NaN）按**排除**处理而非猜测——否则纳排本身就带入了假设。
    """
    if age != age:                        # NaN
        return {"eligible": False, "reason": "paediatric", "note": "年龄缺失"}
    if age < paediatric_cutoff:
        return {"eligible": False, "reason": "paediatric",
                "note": f"年龄 {age} < {paediatric_cutoff}"}
    if n_missing_baseline > max_missing_baseline:
        return {"eligible": False, "reason": "incomplete_records",
                "note": f"基线缺失 {n_missing_baseline} > {max_missing_baseline}"}
    return {"eligible": True, "reason": None, "note": ""}


def reconcile_with_manuscript(final_n: int) -> Dict[str, object]:
    """把最终人数与论文声称的规模对账。"""
    return {
        "manuscript_n": PAPER_COHORT_N,
        "pickle_n": PICKLE_COHORT_N,
        "computed_n": int(final_n),
        "matches_manuscript": int(final_n) == PAPER_COHORT_N,
        "matches_pickle": int(final_n) == PICKLE_COHORT_N,
        "discrepancy": PAPER_COHORT_N - int(final_n),
        "note": "论文写 364 而 pkl 为 363；这一人的差异必须查清并说明",
    }
