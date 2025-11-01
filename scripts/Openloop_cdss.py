# -*- coding: utf-8 -*-
"""
Openloop_cdss.py — One-click train + inference + rich comparisons (3 liquids, cumulative-aware)

修复与改进
- 修复 per-patient AE 表在绘图阶段的 Series→float 报错（.values[0]）。
- 将“患者缩放因子 m”的回归从 logit(m) → 线性回归 log(m)，
  以支持 m>1 的实际情况并避免信息丢失；对应 modifier 改为 exp(a*TBSA+b*W+c)。
- 去除 m_hat 的不合理上界（原 20.0），仅做最小值保护，避免缩放畸变。
- 微调若干健壮性细节（排序、空检查、clip）。

用法同原版。
"""

from __future__ import annotations
import os
import glob
import argparse
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# =============================
# Core model (3 liquids)
# =============================

def exp_decay(t, A, k, B):
    return A * np.exp(-k * t) + B


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class DecayParams:
    A: float
    k: float
    B: float


@dataclass
class LiquidDecay:
    cryst: DecayParams   # 晶体
    colloid: DecayParams # 胶体
    water: DecayParams   # 水


@dataclass
class LogisticParams:  # 名称保留以兼容，但语义已改为 log-scale 线性模型
    a: float
    b: float
    c: float


@dataclass
class UOPParams:
    alpha: float
    beta: float
    gamma_crys: float
    gamma_col: float
    gamma_water: float


@dataclass
class CDSSModel:
    decay_liq: LiquidDecay
    logistic: LogisticParams  # 用于 exp(a*TBSA + b*W + c) 的乘性缩放
    uop: UOPParams

    def modifier(self, tbsa_pct: float, weight_kg: float) -> float:
        # 乘性缩放：>0，无上界，支持 m>1 的真实情况
        z = self.logistic.a * tbsa_pct + self.logistic.b * weight_kg + self.logistic.c
        return float(np.exp(z))

    def infusion_rate_liquid(self, t_h: float, tbsa_pct: float, weight_kg: float, liquid: str) -> float:
        d = getattr(self.decay_liq, liquid)
        base = exp_decay(t_h, d.A, d.k, d.B)
        return float(self.modifier(tbsa_pct, weight_kg) * base)

    def predict_uop(self, t_h: float, r_crys: float, r_col: float, r_water: float) -> float:
        return float(self.uop.alpha * t_h + self.uop.beta +
                     self.uop.gamma_crys * r_crys +
                     self.uop.gamma_col  * r_col  +
                     self.uop.gamma_water* r_water)


# =============================
# Fitting
# =============================

def _fit_decay_one_liquid(df: pd.DataFrame, liquid_col: str) -> Tuple[DecayParams, Dict[str, float]]:
    """Two-stage fit per liquid: per-patient scale -> log(m) ~ a*TBSA+b*W+c -> global A,k,B.
    修正点：回归 log(m) 而非 logit(m)，保留 m>1 的信息。
    """
    m_by_patient: Dict[str, float] = {}
    grouped = df.groupby('patient_id', sort=False)
    for pid, g in grouped:
        g = g.sort_values('time_h')
        t = g['time_h'].to_numpy(dtype=float)
        r = g[liquid_col].to_numpy(dtype=float)
        if len(t) < 3 or np.all(r <= 0):
            m_by_patient[pid] = 1.0
            continue
        A0 = max(1.0, float(r.max() - r.min()))
        k0 = 0.2
        B0 = max(0.0, float(np.percentile(r, 10)))
        try:
            popt, _ = curve_fit(exp_decay, t, r, p0=[A0, k0, B0], bounds=([0.0, 1e-4, 0.0], [1e6, 10.0, 1e5]))
            fit_vals = exp_decay(t, *popt)
            s = float(np.clip((r.mean() / (fit_vals.mean() + 1e-6)), 1e-2, 100.0))  # 放宽上下界
        except Exception:
            s = 1.0
        m_by_patient[pid] = s

    # 回归 log(m) = a*TBSA + b*W + c
    rows = []
    for pid, g in grouped:
        if len(g) == 0:
            continue
        tbsa = float(g['TBSA_pct'].iloc[0])
        w = float(g['weight_kg'].iloc[0])
        m = float(np.clip(m_by_patient.get(pid, 1.0), 1e-6, 1e6))
        log_m = np.log(m)
        rows.append([tbsa, w, log_m])
    if not rows:
        # 退化兜底
        a = b = 0.0; c = 0.0
    else:
        X = np.asarray([[r[0], r[1], 1.0] for r in rows], dtype=float)
        y = np.asarray([r[2] for r in rows], dtype=float)
        theta, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, b, c = theta.tolist()

    def m_hat(tbsa, w):
        return np.exp(a*tbsa + b*w + c)

    # 全局 A,k,B：对缩放后的速率做指数衰减拟合
    t_all, r_all = [], []
    for pid, g in grouped:
        if len(g) == 0:
            continue
        tbsa = float(g['TBSA_pct'].iloc[0])
        w    = float(g['weight_kg'].iloc[0])
        mh = float(max(m_hat(tbsa, w), 1e-6))
        t_all.append(g['time_h'].to_numpy(dtype=float))
        r_all.append(g[liquid_col].to_numpy(dtype=float) / mh)
    t_all = np.concatenate(t_all) if t_all else np.array([0.0, 1.0, 2.0], dtype=float)
    r_all = np.concatenate(r_all) if r_all else np.array([100.0, 80.0, 70.0], dtype=float)

    A0 = max(1.0, float(r_all.max() - r_all.min()))
    k0 = 0.2
    B0 = max(0.0, float(np.percentile(r_all, 10)))
    popt, _ = curve_fit(exp_decay, t_all, r_all, p0=[A0, k0, B0], bounds=([0.0, 1e-4, 0.0], [1e6, 10.0, 1e5]))
    decay = DecayParams(*map(float, popt))

    return decay, {'a': a, 'b': b, 'c': c}


def fit_all(df_train: pd.DataFrame, use_rate_coupling: bool = True) -> CDSSModel:
    decay_crys, aux = _fit_decay_one_liquid(df_train, 'rate_crys_ml_h')
    decay_col , _  = _fit_decay_one_liquid(df_train, 'rate_col_ml_h')
    decay_water, _ = _fit_decay_one_liquid(df_train, 'rate_water_ml_h')
    logistic = LogisticParams(a=float(aux['a']), b=float(aux['b']), c=float(aux['c']))

    work = df_train.dropna(subset=['time_h'])
    if use_rate_coupling:
        work2 = work.dropna(subset=['uop_ml_kg_h'])
        if len(work2) < 3:
            uop = UOPParams(alpha=0.0, beta=float(np.nanmean(work['uop_ml_kg_h'])), gamma_crys=0.0, gamma_col=0.0, gamma_water=0.0)
        else:
            X = np.column_stack([
                work2['time_h'].values, np.ones(len(work2)),
                work2['rate_crys_ml_h'].values, work2['rate_col_ml_h'].values, work2['rate_water_ml_h'].values
            ])
            y = work2['uop_ml_kg_h'].values
            theta, *_ = np.linalg.lstsq(X, y, rcond=None)
            alpha, beta, g1, g2, g3 = theta.tolist()
            uop = UOPParams(alpha=float(alpha), beta=float(beta), gamma_crys=float(g1), gamma_col=float(g2), gamma_water=float(g3))
    else:
        work2 = work.dropna(subset=['uop_ml_kg_h'])
        if len(work2) < 3:
            uop = UOPParams(alpha=0.0, beta=float(np.nanmean(work['uop_ml_kg_h'])), gamma_crys=0.0, gamma_col=0.0, gamma_water=0.0)
        else:
            X = np.column_stack([work2['time_h'].values, np.ones(len(work2))])
            y = work2['uop_ml_kg_h'].values
            theta, *_ = np.linalg.lstsq(X, y, rcond=None)
            alpha, beta = theta.tolist()
            uop = UOPParams(alpha=float(alpha), beta=float(beta), gamma_crys=0.0, gamma_col=0.0, gamma_water=0.0)

    return CDSSModel(decay_liq=LiquidDecay(decay_crys, decay_col, decay_water), logistic=logistic, uop=uop)


# =============================
# Data ingestion (build rates + cumulative)
# =============================

def _norm_pid(pid: Any) -> str:
    s = str(pid).strip()
    if s.endswith('.csv'):
        s = s[:-4]
    return s


def _label_process(df: pd.DataFrame) -> pd.DataFrame:
    liquids = ['水', '晶体', '胶体']
    speeds = {'水': '水速度', '晶体': '晶体速度', '胶体': '胶体速度'}
    for col in list(speeds.values()) + liquids:
        if col not in df.columns:
            df[col] = 0
    # 按行清洗（保留原语义）
    for idx, row in df.iterrows():
        for liq in liquids:
            if pd.notnull(row[liq]) and row[liq] == -1:
                df.loc[idx, liq] = 0
            spd = speeds[liq]
            lval = df.loc[idx, liq]
            sval = df.loc[idx, spd]
            if pd.notnull(lval) and pd.notnull(sval):
                if lval == 0 and sval != 0:
                    df.loc[idx, liq] = 1
                if lval == 1 and sval == 0:
                    df.loc[idx, liq] = 0
    return df[['晶体', '胶体', '水', '晶体速度', '胶体速度', '水速度']]


def _parse_baseline_entry(entry: Any) -> Dict[str, Any]:
    if isinstance(entry, (list, tuple)) and len(entry) >= 10:
        sex, age, height, weight, bmi, tbsa, deg3, deep2, shallow2, inhal = entry[:10]
        return dict(sex=sex, age=age, height=height, weight=weight, bmi=bmi,
                    tbsa=tbsa, deg3=deg3, deep2=deep2, shallow2=shallow2, inhalation=inhal)
    if isinstance(entry, (dict, pd.Series)):
        sex_val = entry.get('基线_性别', entry.get('性别', None))
        if isinstance(sex_val, str):
            sex = 1 if sex_val == '男' else (0 if sex_val == '女' else None)
        else:
            sex = sex_val
        return dict(
            sex=sex,
            age=entry.get('年龄', entry.get('age', None)),
            height=entry.get('身高', entry.get('height', None)),
            weight=entry.get('体重', entry.get('weight', None)),
            bmi=entry.get('体重指数', entry.get('BMI', None)),
            tbsa=entry.get('烧伤总面积', entry.get('TBSA', 0)) or 0,
            deg3=entry.get('合计Ⅲ°', entry.get('III', 0)) or 0,
            deep2=entry.get('合计深Ⅱ°', entry.get('深II', 0)) or 0,
            shallow2=entry.get('合计浅Ⅱ°', entry.get('浅II', 0)) or 0,
            inhalation=entry.get('吸入性损伤', entry.get('inhalation', 0)) or 0
        )
    return dict(sex=None, age=None, height=None, weight=None, bmi=None,
                tbsa=0, deg3=0, deep2=0, shallow2=0, inhalation=0)


def _extract_ts(entry: Any) -> pd.DataFrame:
    def _coerce_df(x):
        return x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)
    if isinstance(entry, dict):
        parts = []
        if 'label' in entry and entry['label'] is not None:
            parts.append(_coerce_df(entry['label']))
        if 'tdata' in entry and entry['tdata'] is not None:
            parts.append(_coerce_df(entry['tdata']))
        if not parts:
            return pd.DataFrame()
        base = parts[0].copy()
        for extra in parts[1:]:
            for c in extra.columns:
                if c not in base.columns:
                    base[c] = extra[c]
        df = base
    elif isinstance(entry, pd.DataFrame):
        df = entry.copy()
    else:
        return pd.DataFrame()

    proc = _label_process(df.copy())
    if 'time_h' not in df.columns:
        df['time_h'] = np.arange(len(df), dtype=float)
    out = pd.concat([df[['time_h']], proc], axis=1)

    for c in ['uop_ml_kg_h', 'UOP_ml_kg_h', '尿量_ml_kg_h', '尿量', 'UOP']:
        if c in df.columns:
            out['uop_ml_kg_h'] = pd.to_numeric(df[c], errors='coerce')
            break
    if 'uop_ml_kg_h' not in out.columns:
        out['uop_ml_kg_h'] = np.nan

    out['rate_crys_ml_h']  = pd.to_numeric(out.get('晶体速度', 0), errors='coerce').fillna(0)
    out['rate_col_ml_h']   = pd.to_numeric(out.get('胶体速度', 0), errors='coerce').fillna(0)
    out['rate_water_ml_h'] = pd.to_numeric(out.get('水速度', 0), errors='coerce').fillna(0)

    # 假设 1 小时采样，rate 的累积近似体积
    out['cum_crys_ml']  = out['rate_crys_ml_h'].cumsum()
    out['cum_col_ml']   = out['rate_col_ml_h'].cumsum()
    out['cum_water_ml'] = out['rate_water_ml_h'].cumsum()

    return out


def build_training_df_from_pkls(pkl_ts_path: str, pkl_base_path: str) -> pd.DataFrame:
    if not os.path.exists(pkl_ts_path):
        raise FileNotFoundError(f"时间序列 pkl 未找到: {pkl_ts_path}")
    if not os.path.exists(pkl_base_path):
        raise FileNotFoundError(f"基线特征 pkl 未找到: {pkl_base_path}")

    with open(pkl_ts_path, 'rb') as f:
        ts_obj_raw = pickle.load(f)
    with open(pkl_base_path, 'rb') as f:
        base_obj_raw = pickle.load(f)

    if not isinstance(ts_obj_raw, dict) or not isinstance(base_obj_raw, dict):
        raise ValueError("两个 PKL 都必须是 dict[pid] -> ... 结构")

    ts_obj  = {_norm_pid(k): v for k, v in ts_obj_raw.items()}
    base_obj = {_norm_pid(k): v for k, v in base_obj_raw.items()}

    pids = sorted(set(ts_obj.keys()) & set(base_obj.keys()))
    if len(pids) == 0:
        sample_ts = list(sorted(ts_obj.keys()))[:10]
        sample_base = list(sorted(base_obj.keys()))[:10]
        raise ValueError(
            """
两个 PKL 的患者编号没有交集。请检查PID格式。
示例（ts前10）：{}
示例（base前10）：{}
""".format(sample_ts, sample_base)
        )

    rows = []
    for pid in pids:
        b = _parse_baseline_entry(base_obj[pid])
        tbsa = 0.0 if b.get('tbsa') is None else float(b['tbsa'])
        weight = np.nan if b.get('weight') is None else float(b['weight'])
        ts_df = _extract_ts(ts_obj[pid])
        if ts_df.empty:
            continue
        ts_df = ts_df.sort_values('time_h')
        for _, r in ts_df.iterrows():
            rows.append(dict(
                patient_id=str(pid),
                time_h=float(r['time_h']),
                rate_crys_ml_h=float(r['rate_crys_ml_h']),
                rate_col_ml_h=float(r['rate_col_ml_h']),
                rate_water_ml_h=float(r['rate_water_ml_h']),
                cum_crys_ml=float(r['cum_crys_ml']),
                cum_col_ml=float(r['cum_col_ml']),
                cum_water_ml=float(r['cum_water_ml']),
                uop_ml_kg_h=(np.nan if pd.isna(r['uop_ml_kg_h']) else float(r['uop_ml_kg_h'])),
                TBSA_pct=float(tbsa),
                weight_kg=(np.nan if pd.isna(weight) else float(weight)),
            ))

    df_train = pd.DataFrame(rows)
    if df_train.empty:
        raise ValueError("构建的训练 DataFrame 为空，请检查 PKL 内容。")

    for col in ['time_h','rate_crys_ml_h','rate_col_ml_h','rate_water_ml_h','uop_ml_kg_h','TBSA_pct','weight_kg','cum_crys_ml','cum_col_ml','cum_water_ml']:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
    for col in ['rate_crys_ml_h','rate_col_ml_h','rate_water_ml_h','cum_crys_ml','cum_col_ml','cum_water_ml']:
        df_train[col] = df_train[col].clip(lower=0)
    df_train['TBSA_pct'] = df_train['TBSA_pct'].fillna(0.0)
    if df_train['weight_kg'].isna().any():
        df_train['weight_kg'] = df_train['weight_kg'].fillna(df_train['weight_kg'].median())
    return df_train


def train_from_pkls(pkl_ts_path: str,
                    pkl_base_path: str,
                    out_model_path: Optional[str] = None,
                    prefer_rate_coupling: bool = True) -> Tuple[CDSSModel, pd.DataFrame]:
    df_train = build_training_df_from_pkls(pkl_ts_path, pkl_base_path)
    has_uop_ratio = df_train['uop_ml_kg_h'].notna().mean()
    use_rate_coupling = prefer_rate_coupling and (has_uop_ratio > 0.3)
    model = fit_all(df_train, use_rate_coupling=use_rate_coupling)
    if out_model_path:
        with open(out_model_path, 'wb') as f:
            pickle.dump(model, f)
    print(f"[INFO] 训练样本数: {len(df_train)} | 患者数: {df_train['patient_id'].nunique()} | UOP可用: {has_uop_ratio:.1%} | 耦合: {use_rate_coupling}")
    return model, df_train


# =============================
# Inference + plotting + errors
# =============================

@dataclass
class PolicyConfig:
    uop_lower: float = 0.5
    uop_upper: float = 1.0
    max_step_change_pct: float = 0.25
    min_rate: float = 0.0
    max_rate: float = 600.0
    nudge_gain: float = 0.6


def recommend_rates(model: CDSSModel, t_h: float, tbsa: float, w: float, cfg: PolicyConfig, prev_rates: Tuple[float,float,float] | None) -> Tuple[float,float,float]:
    rC = model.infusion_rate_liquid(t_h, tbsa, w, 'cryst')
    rL = model.infusion_rate_liquid(t_h, tbsa, w, 'colloid')
    rW = model.infusion_rate_liquid(t_h, tbsa, w, 'water')
    u_pred = model.predict_uop(t_h, rC, rL, rW)

    # 仅对晶体做 nudging（与论文常见实践相符，可按需扩展到胶体/水）
    if u_pred < cfg.uop_lower:
        gap = (cfg.uop_lower - u_pred) / max(1e-6, cfg.uop_lower)
        rC = rC * (1.0 + cfg.nudge_gain * gap)
    elif u_pred > cfg.uop_upper:
        gap = (u_pred - cfg.uop_upper) / max(1e-6, cfg.uop_upper)
        rC = rC * (1.0 - cfg.nudge_gain * gap)

    if prev_rates is not None:
        vals = [rC, rL, rW]
        out_vals = []
        for i, val in enumerate(vals):
            up = prev_rates[i] * (1.0 + cfg.max_step_change_pct)
            dn = prev_rates[i] * (1.0 - cfg.max_step_change_pct)
            out_vals.append(float(np.clip(val, dn, up)))
        rC, rL, rW = out_vals

    rC = float(np.clip(rC, cfg.min_rate, cfg.max_rate))
    rL = float(np.clip(rL, cfg.min_rate, cfg.max_rate))
    rW = float(np.clip(rW, cfg.min_rate, cfg.max_rate))
    return rC, rL, rW


def _format_mean_std(mean_v: float, std_v: float) -> str:
    if np.isnan(mean_v) or np.isnan(std_v):
        return "NA"
    return f"{mean_v:.1f} ± {std_v:.1f}"


def _make_error_tables_per_patient(t: np.ndarray,
                                   recC: np.ndarray, recL: np.ndarray, recW: np.ndarray,
                                   obsC: np.ndarray, obsL: np.ndarray, obsW: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    eps = 1e-6
    segments = {
        "0-24h": (t >= 0) & (t < 24),
        "24-48h": (t >= 24) & (t < 48)
    }
    out_rows = []
    pretty = []
    for seg_name, m in segments.items():
        for liq, (rec, obs) in {
            "Crystalloid": (recC, obsC),
            "Colloid": (recL, obsL),
            "Water": (recW, obsW),
        }.items():
            if m.sum() == 0:
                ae_mean = ae_std = re_mean = re_std = np.nan
            else:
                diff = rec[m] - obs[m]
                ae = np.abs(diff)
                re = diff / np.maximum(obs[m], eps)
                ae_mean, ae_std = float(np.mean(ae)), float(np.std(ae))
                re_mean, re_std = float(np.mean(re)), float(np.std(re))
            out_rows.append({'Segment': seg_name,'Liquid': liq,'AE_mean': ae_mean,'AE_std': ae_std,'RE_mean': re_mean,'RE_std': re_std})
            pretty.append({'Segment': seg_name,'Liquid': liq,'AE (mL/h)': _format_mean_std(ae_mean, ae_std),'RE (ratio)': _format_mean_std(re_mean, re_std)})
    return pd.DataFrame(out_rows), pd.DataFrame(pretty)


def _make_error_tables_all_patients(all_records: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    eps = 1e-6
    rows = []
    for rec in all_records:
        t = np.asarray(rec['t'], dtype=float)
        recC, recL, recW = rec['recC'], rec['recL'], rec['recW']
        obsC, obsL, obsW = rec['obsC'], rec['obsL'], rec['obsW']
        for seg_name, m in {"0-24h": (t >= 0) & (t < 24), "24-48h": (t >= 24) & (t < 48)}.items():
            if m.sum() == 0:
                continue
            for liq, (r, o) in {"Crystalloid": (recC, obsC), "Colloid": (recL, obsL), "Water": (recW, obsW)}.items():
                diff = r[m] - o[m]
                ae = np.abs(diff)
                re = diff / np.maximum(o[m], eps)
                for a, b in zip(ae, re):
                    rows.append({'Segment': seg_name,'Liquid': liq,'AE': float(a),'RE': float(b),'pid': rec['pid']})
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(rows)
    num_df = df.groupby(['Segment', 'Liquid']).agg(AE_mean=('AE', 'mean'), AE_std=('AE', 'std'), RE_mean=('RE', 'mean'), RE_std=('RE', 'std')).reset_index()
    # Pretty AE table like mean±std per liquid per segment
    def fmt(m, s):
        return f"{m:.1f} ± {s:.1f}" if pd.notnull(m) else "NA"
    pretty_rows = []
    for seg in ["0-24h", "24-48h"]:
        row = {'Segment': seg}
        for liq in ["Crystalloid", "Colloid", "Water"]:
            sub = num_df[(num_df['Segment'] == seg) & (num_df['Liquid'] == liq)]
            if len(sub):
                row[f'{liq} AE'] = fmt(sub['AE_mean'].values[0], sub['AE_std'].values[0])
        pretty_rows.append(row)
    ae_pretty = pd.DataFrame(pretty_rows)
    return num_df, ae_pretty


def _collect_patient_record(model: CDSSModel, pid: str, ts_obj: dict, base_obj: dict, uop_lower: float, uop_upper: float):
    ts_df = _extract_ts(ts_obj[pid])
    base = _parse_baseline_entry(base_obj[pid])
    tbsa = float(base.get('tbsa', 0) or 0)
    weight = base.get('weight', np.nan)
    weight = float(weight) if pd.notnull(weight) else 70.0
    cfg = PolicyConfig(uop_lower=uop_lower, uop_upper=uop_upper)
    recC, recL, recW, predU = [], [], [], []
    prev = None
    for tt in ts_df.sort_values('time_h')['time_h'].values.astype(float):
        rC, rL, rW = recommend_rates(model, float(tt), tbsa, weight, cfg, prev)
        recC.append(rC); recL.append(rL); recW.append(rW)
        predU.append(model.predict_uop(float(tt), rC, rL, rW))
        prev = (rC, rL, rW)
    ts_df = ts_df.sort_values('time_h')
    return {
        'pid': pid,
        't': ts_df['time_h'].values,
        'recC': np.array(recC), 'recL': np.array(recL), 'recW': np.array(recW),
        'obsC': ts_df['rate_crys_ml_h'].values,
        'obsL': ts_df['rate_col_ml_h'].values,
        'obsW': ts_df['rate_water_ml_h'].values,
        'predU': np.array(predU),
        'ts_df': ts_df,
    }


def plot_inference(model: CDSSModel,
                   ts_obj: dict,
                   base_obj: dict,
                   pid: Optional[str],
                   out_png: Optional[str],
                   uop_lower: float = 0.5,
                   uop_upper: float = 1.0,
                   extra_plots: bool = True):
    # normalize keys
    ts_obj = {_norm_pid(k): v for k, v in ts_obj.items()}
    base_obj = {_norm_pid(k): v for k, v in base_obj.items()}

    # choose pid
    if pid is None:
        inter = sorted(set(ts_obj.keys()) & set(base_obj.keys()))
        if not inter:
            raise ValueError('No common patient IDs between time-series and baseline pkls.')
        pid = inter[0]
        print(f"[INFO] --infer_pid 未指定，自动选择: {pid}")
    else:
        pid = _norm_pid(pid)
        if pid not in ts_obj or pid not in base_obj:
            raise ValueError(f"PID 不存在或不匹配: {pid}")

    rec = _collect_patient_record(model, pid, ts_obj, base_obj, uop_lower, uop_upper)
    ts_df = rec['ts_df']
    t     = rec['t']
    recC, recL, recW = rec['recC'], rec['recL'], rec['recW']
    predU            = rec['predU']

    # === Figure 1: rates + predicted UOP (model vs observed) ===
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(t, recC, label='Rec Crystalloid (mL/h)')
    ax1.plot(t, recL, label='Rec Colloid (mL/h)')
    ax1.plot(t, recW, label='Rec Water (mL/h)')
    ax1.plot(t, ts_df['rate_crys_ml_h'].values, linestyle='--', label='Obs Crystalloid (mL/h)')
    ax1.plot(t, ts_df['rate_col_ml_h'].values, linestyle='--', label='Obs Colloid (mL/h)')
    ax1.plot(t, ts_df['rate_water_ml_h'].values, linestyle='--', label='Obs Water (mL/h)')
    ax1.set_xlabel('Time (h)'); ax1.set_ylabel('Infusion rate (mL/h)')
    ax1.grid(True, alpha=0.3); ax1.legend(loc='upper left', ncol=2)

    ax2 = ax1.twinx()
    ax2.plot(t, predU, label='Predicted UOP (mL/kg/h)')
    ax2.axhspan(uop_lower, uop_upper, alpha=0.12)
    ax2.set_ylabel('UOP (mL/kg/h)'); ax2.legend(loc='upper right')
    fig.tight_layout()

    if out_png:
        os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
        plt.savefig(out_png, dpi=160)
        print(f"[OK] Saved figure -> {out_png}")
    else:
        plt.show()

    if not extra_plots:
        return

    base_stem = None
    if out_png:
        base_dir = os.path.dirname(out_png) or '.'
        stem = os.path.splitext(os.path.basename(out_png))[0]
        base_stem = os.path.join(base_dir, stem)

    # === Figure 2: cumulative volume (model vs observed) ===
    recC_cum = np.cumsum(recC); recL_cum = np.cumsum(recL); recW_cum = np.cumsum(recW)
    fig2, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, recC_cum, label='Rec Crystalloid (cum mL)')
    ax.plot(t, recL_cum, label='Rec Colloid (cum mL)')
    ax.plot(t, recW_cum, label='Rec Water (cum mL)')
    ax.plot(t, ts_df['cum_crys_ml'].values, linestyle='--', label='Obs Crystalloid (cum mL)')
    ax.plot(t, ts_df['cum_col_ml'].values,  linestyle='--', label='Obs Colloid (cum mL)')
    ax.plot(t, ts_df['cum_water_ml'].values,linestyle='--', label='Obs Water (cum mL)')
    ax.set_xlabel('Time (h)'); ax.set_ylabel('Cumulative Volume (mL)')
    ax.grid(True, alpha=0.3); ax.legend(loc='best', ncol=2)
    fig2.tight_layout()
    if base_stem:
        path2 = f"{base_stem}_cum_compare.png"; plt.savefig(path2, dpi=160); print(f"[OK] Saved cumulative compare figure -> {path2}")
    else:
        plt.show()

    # === Figure 3: total volume comparison bars (per liquid & total) ===
    totals_rec = np.array([recC_cum[-1], recL_cum[-1], recW_cum[-1]])
    totals_obs = np.array([float(ts_df['cum_crys_ml'].values[-1]), float(ts_df['cum_col_ml'].values[-1]), float(ts_df['cum_water_ml'].values[-1])])
    labels = ['Crystalloid','Colloid','Water','Total']
    rec_all = np.append(totals_rec, totals_rec.sum()); obs_all = np.append(totals_obs, totals_obs.sum())
    x = np.arange(len(labels)); width = 0.38
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.bar(x - width/2, obs_all, width, label='Observed'); ax3.bar(x + width/2, rec_all, width, label='Model (Rec)')
    ax3.set_xticks(x); ax3.set_xticklabels(labels); ax3.set_ylabel('Volume (mL)'); ax3.set_title('Total Volumes: Observed vs Model')
    ax3.grid(True, axis='y', alpha=0.3); ax3.legend(); fig3.tight_layout()
    if base_stem:
        path3 = f"{base_stem}_totals_bar.png"; plt.savefig(path3, dpi=160); print(f"[OK] Saved totals bar figure -> {path3}")
    else:
        plt.show()

    # === Figure 4: UOP diagnostics — pred vs obs & residuals ===
    fig4, ax4 = plt.subplots(figsize=(11, 5))
    ax4.plot(t, predU, label='Predicted UOP (mL/kg/h)')
    if 'uop_ml_kg_h' in ts_df.columns and ts_df['uop_ml_kg_h'].notna().any():
        obs_uop = ts_df['uop_ml_kg_h'].values
        ax4.plot(t, obs_uop, linestyle='--', label='Observed UOP (mL/kg/h)')
        res = np.array(predU) - obs_uop
        ax4_2 = ax4.twinx(); ax4_2.plot(t, res, label='Residual (Pred-Obs)', linestyle=':'); ax4_2.set_ylabel('Residual'); ax4_2.legend(loc='lower right')
    ax4.axhspan(uop_lower, uop_upper, alpha=0.12)
    ax4.set_xlabel('Time (h)'); ax4.set_ylabel('UOP (mL/kg/h)')
    ax4.grid(True, alpha=0.3); ax4.legend(loc='upper left'); fig4.tight_layout()
    if base_stem:
        path4 = f"{base_stem}_uop_compare.png"; plt.savefig(path4, dpi=160); print(f"[OK] Saved UOP compare figure -> {path4}")
    else:
        plt.show()

    # === Figure 5: Error summaries (per-patient) + bar chart ===
    num_tbl_p, ae_pretty_p = _make_error_tables_per_patient(
        t=np.asarray(t, dtype=float), recC=recC, recL=recL, recW=recW,
        obsC=ts_df['rate_crys_ml_h'].values, obsL=ts_df['rate_col_ml_h'].values, obsW=ts_df['rate_water_ml_h'].values,
    )
    if base_stem:
        num_path = f"{base_stem}_error_table_numeric.csv"; pretty_path = f"{base_stem}_error_table_AE_pretty.csv"
        num_tbl_p.to_csv(num_path, index=False); ae_pretty_p.to_csv(pretty_path, index=False)
        print(f"[OK] Saved error numeric table -> {num_path}")
        print(f"[OK] Saved error AE pretty table -> {pretty_path}")

    segs = ["0-24h", "24-48h"]; liqs = ["Crystalloid","Colloid","Water"]
    ae_means = []; ae_stds = []
    for seg in segs:
        sub = num_tbl_p[num_tbl_p['Segment'] == seg]
        means = []
        stds = []
        for liq in liqs:
            sel = sub.loc[sub['Liquid'] == liq, 'AE_mean']
            sel_s = sub.loc[sub['Liquid'] == liq, 'AE_std']
            means.append(float(sel.values[0]) if len(sel) else np.nan)
            stds.append(float(sel_s.values[0]) if len(sel_s) else np.nan)
        ae_means.append(means)
        ae_stds.append(stds)
    ae_means = np.array(ae_means); ae_stds = np.array(ae_stds)
    x = np.arange(len(liqs)); width = 0.35
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    ax5.bar(x - width/2, ae_means[0], width, yerr=ae_stds[0], label='AE 0-24h')
    ax5.bar(x + width/2, ae_means[1], width, yerr=ae_stds[1], label='AE 24-48h')
    ax5.set_xticks(x); ax5.set_xticklabels(liqs); ax5.set_ylabel('AE (mL/h)'); ax5.set_title('Average Absolute Error per Segment')
    ax5.grid(True, axis='y', alpha=0.3); ax5.legend(); fig5.tight_layout()
    if base_stem:
        path5 = f"{base_stem}_error_bar_AE.png"; plt.savefig(path5, dpi=160); print(f"[OK] Saved AE bar figure -> {path5}")
    else:
        plt.show()


# =============================
# CLI + GLOBAL aggregation
# =============================

def _resolve_from_data_dir(data_dir: str) -> Tuple[str, str]:
    ts_candidates = [os.path.join(data_dir, 'labels.pkl'), os.path.join(data_dir, 'output_data_final.pkl')] + sorted(glob.glob(os.path.join(data_dir, '*label*.pkl')))
    base_candidates = [os.path.join(data_dir, 'baseline.pkl')] + sorted(glob.glob(os.path.join(data_dir, '*base*.pkl')))
    ts = next((p for p in ts_candidates if os.path.exists(p)), None)
    base = next((p for p in base_candidates if os.path.exists(p)), None)
    if not ts or not base:
        raise FileNotFoundError(f"在 {data_dir} 未找到合适的 PKL：时间序列候选 {ts_candidates}；基线候选 {base_candidates}")
    print(f"[INFO] 选用时间序列: {ts}")
    print(f"[INFO] 选用基线: {base}")
    return ts, base


def main():
    ap = argparse.ArgumentParser(description='Train + Inference + Global AE/RE (3-liquids, cumulative-aware)')
    ap.add_argument('--pkl_ts_path', default=None, help='时间序列 PKL 路径（dict[pid]->label/tdata）')
    ap.add_argument('--pkl_base_path', default=None, help='基线 PKL 路径（dict[pid]->baseline）')
    ap.add_argument('--data_dir', default=None, help='如提供，将在该目录下自动寻找 PKL')
    ap.add_argument('--out', required=True, help='模型输出路径，例如 ./data/model.pkl')
    ap.add_argument('--export_csv', default=None, help='可选：导出训练明细 CSV（含三液体速率与累计）')
    ap.add_argument('--no_rate_coupling', action='store_true', help='禁用 UOP 与速率耦合（gamma=0）')
    ap.add_argument('--only_infer', action='store_true', help='仅推理画图：跳过训练，直接从 --out 载入模型')
    ap.add_argument('--infer_pid', default=None, help='推理的患者ID，不填则自动选择交集中的第一个')
    ap.add_argument('--plot_out', default=None, help='保存推理主图路径，不填则弹窗显示')
    ap.add_argument('--uop_lower', type=float, default=0.5)
    ap.add_argument('--uop_upper', type=float, default=1.0)
    args = ap.parse_args()

    # Resolve PKL paths
    if args.pkl_ts_path and args.pkl_base_path:
        ts_path = args.pkl_ts_path; base_path = args.pkl_base_path
    elif args.data_dir:
        ts_path, base_path = _resolve_from_data_dir(args.data_dir)
    else:
        ts_path, base_path = _resolve_from_data_dir('./data')

    # Load PKLs for inference / aggregation later
    with open(ts_path, 'rb') as f:
        ts_obj_raw = pickle.load(f)
    with open(base_path, 'rb') as f:
        base_obj_raw = pickle.load(f)
    ts_obj  = {_norm_pid(k): v for k, v in ts_obj_raw.items()}
    base_obj = {_norm_pid(k): v for k, v in base_obj_raw.items()}

    # Train or load
    if args.only_infer:
        if not os.path.exists(args.out):
            raise FileNotFoundError(f"--only_infer 模式需要已存在的模型: {args.out}")
        with open(args.out, 'rb') as f:
            model = pickle.load(f)
        df_train = None
    else:
        model, df_train = train_from_pkls(ts_path, base_path, out_model_path=args.out, prefer_rate_coupling=(not args.no_rate_coupling))
        if args.export_csv:
            out_csv = args.export_csv
            if os.path.isdir(out_csv):
                out_csv = os.path.join(out_csv, 'train_for_debug.csv')
            df_train.to_csv(out_csv, index=False)
            print(f"[INFO] 训练明细已导出: {out_csv}")

    # Single-patient plots (if requested)
    if args.infer_pid is not None or args.plot_out is not None:
        plot_inference(model, ts_obj, base_obj, args.infer_pid, args.plot_out, args.uop_lower, args.uop_upper)

    # ===== GLOBAL aggregation across ALL patients =====
    inter = sorted(set(ts_obj.keys()) & set(base_obj.keys()))
    all_records: List[Dict[str, Any]] = []
    for pid in inter:
        try:
            rec = _collect_patient_record(model, pid, ts_obj, base_obj, args.uop_lower, args.uop_upper)
            all_records.append(rec)
        except Exception as e:
            print(f"[WARN] 聚合时跳过 PID={pid}: {e}")

    if all_records:
        num_tbl_all, ae_pretty_all = _make_error_tables_all_patients(all_records)
        # Save & print
        out_dir = os.path.dirname(args.out) or '.'
        num_path = os.path.join(out_dir, 'GLOBAL_error_table_numeric.csv')
        pretty_path = os.path.join(out_dir, 'GLOBAL_error_table_AE_pretty.csv')
        num_tbl_all.to_csv(num_path, index=False)
        ae_pretty_all.to_csv(pretty_path, index=False)
        print(f"[OK] GLOBAL numeric AE/RE -> {num_path}")
        print(f"[OK] GLOBAL AE pretty -> {pretty_path}")

        # Console pretty print
        print("[INFO] === 全样本 AE/RE 统计（每小时） ===")
        for seg in ["0-24h", "24-48h"]:
            sub = num_tbl_all[num_tbl_all['Segment'] == seg]
            for liq in ["Crystalloid", "Colloid", "Water"]:
                row = sub[sub['Liquid'] == liq]
                if len(row) == 1:
                    ae_m, ae_s = row['AE_mean'].values[0], row['AE_std'].values[0]
                    re_m, re_s = row['RE_mean'].values[0], row['RE_std'].values[0]
                    print(f"{seg:<8s} {liq:<12s}  AE={ae_m:.2f}±{ae_s:.2f}  RE={re_m:.2f}±{re_s:.2f}")
        print("[INFO] ===============================")
    else:
        print("[WARN] 未能构建全样本误差统计（无可用记录）。")


if __name__ == '__main__':
    main()
