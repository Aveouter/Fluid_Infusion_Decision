"""复苏公式对照的量纲与口径（issue #25，对应 R1-2/3、R2-7、R3-7、R4-7/8）。

问题
----
论文图 5 把 TMMU/Parkland 类公式与 BurnResu 做误差比较。但两者的**任务不同**：

- 公式输出的是 **24 小时总量**，单位 ``mL/kg/%TBSA``（每公斤体重、每百分点
  烧伤面积）；
- BurnResu 输出的是 **每小时速率**，单位 ``mL/h``。

直接把两者相减没有意义。本模块把量纲换算显式化，并强制对照在**同一物理量**
（某患者某时段的累计体积，mL）上进行。

同时必须写明：两类方法比的都是「与**记录下来的临床医生行为**的距离」，
不是「哪种方案对患者更好」。后者需要结局证据，本数据没有（R4-8）。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

#: 公式输出的单位。换算到 mL 需要体重与 TBSA 两个分母。
FORMULA_UNIT = "mL/kg/%TBSA"

#: 允许的报告单位。
REPORT_UNITS = ("ml", "ml_per_kg", "ml_per_kg_tbsa")


def formula_total_ml(rate_per_kg_tbsa: float, weight_kg: float,
                     tbsa_percent: float) -> float:
    """把 ``mL/kg/%TBSA`` 的公式输出换算为该患者的 24 小时总量（mL）。

    注意 ``tbsa_percent`` 按**百分数**传入（45 表示 45%），与数据字典一致。
    """
    if weight_kg <= 0:
        raise ValueError(f"体重必须为正，实际 {weight_kg}")
    if tbsa_percent < 0:
        raise ValueError(f"TBSA 不得为负，实际 {tbsa_percent}")
    return float(rate_per_kg_tbsa) * float(weight_kg) * float(tbsa_percent)


def to_report_unit(volume_ml: float, unit: str, weight_kg: float,
                   tbsa_percent: float) -> float:
    """把 mL 换算到指定报告单位；分母为 0 时报错而不是产生 inf。"""
    if unit not in REPORT_UNITS:
        raise ValueError(f"unit 必须是 {REPORT_UNITS} 之一，实际 {unit}")
    if unit == "ml":
        return float(volume_ml)
    if weight_kg <= 0:
        raise ValueError("按体重归一化需要正的体重")
    if unit == "ml_per_kg":
        return float(volume_ml) / float(weight_kg)
    if tbsa_percent <= 0:
        raise ValueError("按 TBSA 归一化需要正的 TBSA")
    return float(volume_ml) / (float(weight_kg) * float(tbsa_percent))


def absolute_and_relative_error(true_ml: float, pred_ml: float,
                                eps: float = 1e-9) -> Dict[str, float]:
    """绝对误差（mL）与相对误差。

    相对误差在真实总量为 0 时无定义——返回 NaN 并标注，**不静默置 0**
    （置 0 会把"完全没预测对"记成"完美"）。
    """
    ae = abs(float(pred_ml) - float(true_ml))
    if abs(float(true_ml)) <= eps:
        return {"ae_ml": ae, "re": float("nan"), "re_defined": 0.0}
    return {"ae_ml": ae, "re": ae / abs(float(true_ml)), "re_defined": 1.0}


def summarize_errors(errors) -> Dict[str, float]:
    """汇总一批 ``absolute_and_relative_error`` 的结果。

    逐项给出 AE 与 RE 的均值/分位数，并**单独报出 RE 未定义的样本数**。
    """
    ae = np.array([e["ae_ml"] for e in errors], dtype=float)
    re = np.array([e["re"] for e in errors], dtype=float)
    defined = np.isfinite(re)
    out = {
        "n": int(len(errors)),
        "n_re_defined": int(defined.sum()),
        "n_re_undefined": int((~defined).sum()),
        "ae_mean_ml": float(ae.mean()) if len(ae) else float("nan"),
        "ae_p50_ml": float(np.percentile(ae, 50)) if len(ae) else float("nan"),
        "ae_p90_ml": float(np.percentile(ae, 90)) if len(ae) else float("nan"),
        "re_mean": float(re[defined].mean()) if defined.any() else float("nan"),
        "re_p50": float(np.percentile(re[defined], 50)) if defined.any() else float("nan"),
        "re_p90": float(np.percentile(re[defined], 90)) if defined.any() else float("nan"),
        "unit": "ae in mL; re dimensionless",
    }
    return out


def within_threshold(errors, threshold: float = 0.40) -> Dict[str, float]:
    """相对误差落在给定阈值内的比例，并给出 AE/RE 两种口径下的构成。

    论文的「40% 以内」表述必须同时说明：分母是什么、哪些样本因分母为 0 被排除。
    """
    re = np.array([e["re"] for e in errors], dtype=float)
    defined = np.isfinite(re)
    if not defined.any():
        return {"threshold": threshold, "n": 0, "fraction": float("nan"),
                "n_undefined": int((~defined).sum())}
    ok = re[defined] <= float(threshold)
    return {
        "threshold": float(threshold),
        "n": int(defined.sum()),
        "n_within": int(ok.sum()),
        "fraction": float(ok.mean()),
        "n_undefined": int((~defined).sum()),
    }


def imitation_not_efficacy_note() -> str:
    """对照的解读边界，应逐字出现在图注与讨论中（R4-7）。"""
    return (
        "该对照衡量的是各方法**重现既往临床医生记录行为**的程度，"
        "不是临床疗效或最优策略的比较。公式只使用基线变量，"
        "而模型使用逐小时纵向信息，因此误差更小属于预期而非结论；"
        "任何关于改善预后或治疗优越性的表述都需要结局证据。"
    )
