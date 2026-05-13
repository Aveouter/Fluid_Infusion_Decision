# -*- coding: utf-8 -*-
"""
Openloop_cdss.py — One-click train + inference + evaluation in a single file (no external module dependency).
"""

from __future__ import annotations
import os
import argparse
import pickle
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ========= Core Logic ========= #

def exp_decay(t, A, k, B):
    return A * np.exp(-k * t) + B

def _norm_pid(pid: str) -> str:
    return pid.strip().replace('.csv', '')

def _resolve_from_data_dir(data_dir: str) -> tuple[str, str]:
    ts_candidates = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if 'label' in f and f.endswith('.pkl')]
    base_candidates = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if 'base' in f and f.endswith('.pkl')]
    ts = next((f for f in ts_candidates if os.path.exists(f)), None)
    base = next((f for f in base_candidates if os.path.exists(f)), None)
    if not ts or not base:
        raise FileNotFoundError("未在指定目录中找到 label 或 baseline 的 pkl 文件")
    print(f"[INFO] 选用时间序列: {ts}\n[INFO] 选用基线: {base}")
    return ts, base

class DummyCDSS:
    def __init__(self):
        self.avg = None

    def train(self, df: pd.DataFrame):
        self.avg = df[['rate_crys_ml_h', 'rate_col_ml_h', 'rate_water_ml_h']].mean().to_dict()

    def predict(self, t: float) -> tuple[float, float, float]:
        return tuple(self.avg.get(k, 0) for k in ['rate_crys_ml_h', 'rate_col_ml_h', 'rate_water_ml_h'])

def build_training_df(ts_obj, base_obj) -> pd.DataFrame:
    rows = []
    for pid in sorted(set(ts_obj) & set(base_obj)):
        ts = ts_obj[pid]
        base = base_obj[pid]
        # 正确的循环方式：遍历DataFrame的每一行
        for i, (index, row) in enumerate(ts.iterrows()):
            rows.append({
                'patient_id': pid,
                'time_h': i,
                'rate_crys_ml_h': row['晶体速度'],  # 直接使用列名访问
                'rate_col_ml_h': row['胶体速度'],
                'rate_water_ml_h': row['水速度'],
                'crys_ml': row['晶体'],  # 添加累积量
                'col_ml': row['胶体'],
                'water_ml': row['水'],
                'baseline': base[i] if i < len(base) else 0
            })   
    df = pd.DataFrame(rows)
    return df
def train_from_pkls(ts_path: str, base_path: str, out_model_path: str, prefer_rate_coupling: bool) -> tuple[DummyCDSS, pd.DataFrame]:
    with open(ts_path, 'rb') as f:
        ts_obj_raw = pickle.load(f)
    with open(base_path, 'rb') as f:
        base_obj_raw = pickle.load(f)
    ts_obj = {_norm_pid(k): v for k, v in ts_obj_raw.items()}
    base_obj = {_norm_pid(k): v for k, v in base_obj_raw.items()}
    df_train = build_training_df(ts_obj, base_obj)
    model = DummyCDSS()
    model.train(df_train)
    with open(out_model_path, 'wb') as f:
        pickle.dump(model, f)
    return model, df_train

def plot_inference(model: DummyCDSS, ts_obj, base_obj, pid: str, out_path: Optional[str], uop_lower: float, uop_upper: float):
    if pid is None:
        pid = sorted(set(ts_obj) & set(base_obj))[0]
    df = pd.DataFrame(ts_obj[pid])
    t = np.arange(len(df))
    preds = [model.predict(tt) for tt in t]
    recC, recL, recW = zip(*preds)

    plt.figure(figsize=(10, 5))
    plt.plot(t, recC, label='Pred Crystalloid')
    plt.plot(t, recL, label='Pred Colloid')
    plt.plot(t, recW, label='Pred Water')
    plt.plot(t, df.get('晶体速度', pd.Series([0]*len(t))), '--', label='Obs Crystalloid')
    plt.plot(t, df.get('胶体速度', pd.Series([0]*len(t))), '--', label='Obs Colloid')
    plt.plot(t, df.get('水速度', pd.Series([0]*len(t))), '--', label='Obs Water')
    plt.title(f'Infusion Prediction vs Observation — PID={pid}')
    plt.xlabel('Time (h)')
    plt.ylabel('Rate (mL/h)')
    plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
        print(f"[OK] 图像已保存: {out_path}")
    else:
        plt.show()

# ========= Main Entrypoint ========= #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl_ts_path', default=None)
    ap.add_argument('--pkl_base_path', default=None)
    ap.add_argument('--data_dir', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--export_csv', default=None)
    ap.add_argument('--no_rate_coupling', action='store_true')
    ap.add_argument('--only_infer', action='store_true')
    ap.add_argument('--infer_pid', default=None)
    ap.add_argument('--plot_out', default=None)
    ap.add_argument('--uop_lower', type=float, default=0.5)
    ap.add_argument('--uop_upper', type=float, default=1.0)
    args = ap.parse_args()

    if args.pkl_ts_path and args.pkl_base_path:
        ts_path, base_path = args.pkl_ts_path, args.pkl_base_path
    elif args.data_dir:
        ts_path, base_path = _resolve_from_data_dir(args.data_dir)
    else:
        raise ValueError("请通过 --data_dir 或 --pkl_ts_path + --pkl_base_path 提供数据")

    with open(ts_path, 'rb') as f:
        ts_obj_raw = pickle.load(f)
    with open(base_path, 'rb') as f:
        base_obj_raw = pickle.load(f)
    ts_obj = {_norm_pid(k): v for k, v in ts_obj_raw.items()}
    base_obj = {_norm_pid(k): v for k, v in base_obj_raw.items()}

    if args.only_infer:
        with open(args.out, 'rb') as f:
            model = pickle.load(f)
    else:
        model, df_train = train_from_pkls(ts_path, base_path, out_model_path=args.out, prefer_rate_coupling=(not args.no_rate_coupling))
        if args.export_csv:
            df_train.to_csv(args.export_csv, index=False)

    if args.infer_pid or args.plot_out:
        plot_inference(model, ts_obj, base_obj, args.infer_pid, args.plot_out, args.uop_lower, args.uop_upper)

if __name__ == '__main__':
    main()
