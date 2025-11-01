# -*- coding: utf-8 -*-
"""
数据加载与预处理（增强版，兼容你现有工程）：
- 按病人/窗口划分；支持窗口步长 window_stride（降相关、提速）。
- 时间重采样（按体积守恒）：bit=OR_k，speed=Σ(bit*speed)/k。
- 重新计算 (inst_vol, cum_vol) 保证累计体积单调。
- 多目标类别过采样采样器（默认仅 water）。
- RobustScaler（基于训练集的中位数/IQR）统一特征规范化。
- lint_data() 一键体检（体积漂移、长度不一致等）。

返回：(train_loader, val_loader, test_loader)

用法示例：
train_loader, val_loader, test_loader = data_loader(
    Ts_data, Base_data,
    history_length=11, pred_length=1, batch_size=32,
    split_mode="patient", splits=(0.8, 0.1, 0.1),
    oversample_water=True, water_pos_ratio=0.5,
    target_step_hours=2.0, orig_step_hours=1.0,
    window_stride=2,  # 新增：每隔2步取一个滑窗
)
"""

import os
import re
import random
import pickle
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

# 仅在你项目存在时可用；没有也不报错
try:
    from utils.ts_augmentation_toolkit import TSAugmenter  # type: ignore
except Exception:  # 兜底一个空增强器
    class TSAugmenter:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, x, b, y):
            return x, b, y


# =========================
# 体检器（可选）：不改数据，只打印统计
# =========================

def lint_data(Ts_data: Dict, base_data: Dict, verbose: bool = True) -> None:
    """对原始序列做基础体检：长度、体积漂移、非法数。
    - 若 bit==0 但 speed>0，将被计入体积漂移（理论应为0）。
    - 仅打印统计，不修改 Ts_data。
    """
    def _as_np(a):
        if isinstance(a, pd.DataFrame):
            a = a.to_numpy()
        return np.asarray(a, dtype=np.float32)

    n_rec, n_len_mismatch = 0, 0
    vol_drifts: List[float] = []

    for key, rec in Ts_data.items():
        n_rec += 1
        tdata = _as_np(rec.get('tdata'))
        label = rec.get('label')
        label = label.to_numpy(dtype=np.float32) if isinstance(label, pd.DataFrame) else _as_np(label)

        if len(tdata) != len(label):
            n_len_mismatch += 1
            L = min(len(tdata), len(label))
            tdata = tdata[:L]
            label = label[:L]

        bits = label[:, :3]
        spd = np.clip(label[:, 3:6], 0.0, np.float32(1e9))
        inst_vol_raw = (bits * spd).sum(axis=1)

        # 清洗后：强制 speed=0 当 bit=0
        spd_clean = spd * (bits > 0.5)
        inst_vol_clean = (bits * spd_clean).sum(axis=1)

        s_raw = float(inst_vol_raw.sum())
        s_clean = float(inst_vol_clean.sum())
        drift = abs(s_clean - s_raw) / (s_raw + 1e-6)
        vol_drifts.append(drift)

    drift_avg = float(np.mean(vol_drifts)) if vol_drifts else 0.0
    if verbose:
        print("=" * 70)
        print(f"[LINT] records: {n_rec} | len mismatch: {n_len_mismatch}")
        print(f"[LINT] avg volume drift after bit-masking: {drift_avg*100:.3f}%")
        print("=" * 70)


# =========================
# Robust 特征规范化（基于训练集）
# =========================

class RobustScaler:
    """对 X(13列) 做 robust z-score: (x - median) / IQR。"""
    def __init__(self, eps: float = 1e-6):
        self.median: Optional[np.ndarray] = None
        self.iqr: Optional[np.ndarray] = None
        self.eps = eps

    def fit(self, X_list: Iterable[np.ndarray]):
        Xcat = np.concatenate([np.asarray(x, dtype=np.float32) for x in X_list if len(x) > 0], axis=0)
        self.median = np.nanmedian(Xcat, axis=0)
        q75 = np.nanpercentile(Xcat, 75, axis=0)
        q25 = np.nanpercentile(Xcat, 25, axis=0)
        self.iqr = np.maximum(q75 - q25, self.eps)

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.median is not None and self.iqr is not None, "Call fit() first"
        return (x - self.median) / self.iqr

    def __call__(self, x: torch.Tensor, b: torch.Tensor, y: torch.Tensor):
        x2 = self.transform(x.cpu().numpy())
        return torch.as_tensor(x2, dtype=torch.float32), b, y


# =========================
# Dataset
# =========================

class CustomDataset(Dataset):
    """
    每个样本是一个滑窗：
      - x: (H+P, 13) 时序特征（9列基础特征 + 4列累计体积）
      - b: (B,) baseline 向量（同病人相同）
      - y: (H+P, >=6) 标签，前3列为类型0/1，后3列为速度
    transform（如 TSAugmenter/RobustScaler）作用在 (x,b,y)。
    """
    def __init__(self, data, labels, baseline_data, history_length: int, transform=None):
        self.history_length = int(history_length)
        self.data = data
        self.labels = labels
        if isinstance(baseline_data, pd.DataFrame):
            self.baseline_data = baseline_data.to_numpy()
        else:
            self.baseline_data = baseline_data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)      # (H+P, 13)
        b = torch.tensor(self.baseline_data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)    # (H+P, >=6)
        if self.transform is not None:
            x, b, y = self.transform(x, b, y)
        return x, b, y


# =========================
# 采样器：多目标类别过采样（默认 water）
# =========================

def _make_multi_target_sampler(labels_list: List[np.ndarray], pred_length: int,
                               targets: Tuple[int, ...] = (2,), pos_ratio: float = 0.5):
    """
    labels_list: 与训练集中样本一一对应的 label 窗口 (H+P, >=6)
    targets: 需要过采样的类别索引（如 (2,) 表示 water；(0,2) 表示 crystal+water）
    pos_ratio: 目标阳性采样比例（相对于训练样本总数）
    阳性判定：预测窗内任一时刻该类 bit==1
    """
    if len(labels_list) == 0:
        return None

    N = len(labels_list)
    is_pos = np.zeros(N, dtype=np.int32)
    for i, lab in enumerate(labels_list):
        arr = np.asarray(lab, dtype=np.float32)
        win_bits = arr[-pred_length:, :3]
        is_pos[i] = 1 if (win_bits[:, list(targets)] > 0.5).any() else 0

    N_pos = int(is_pos.sum())
    N_neg = N - N_pos
    if N_pos == 0 or N_neg == 0:
        weights = np.ones(N, dtype=np.float32)
    else:
        target = float(min(max(pos_ratio, 0.05), 0.95))
        w_pos = (target / (1.0 - target)) * (N_neg / max(N_pos, 1))
        w_neg = 1.0
        weights = np.where(is_pos == 1, w_pos, w_neg).astype(np.float32)

    return WeightedRandomSampler(weights=torch.as_tensor(weights, dtype=torch.float),
                                 num_samples=len(weights), replacement=True)


# =========================
# Data Loader Builder
# =========================

def data_loader(
    Ts_data: Dict,
    base_data: Dict,
    history_length: int,
    pred_length: int,
    classes: int = 3,
    batch_size: int = 32,
    split_mode: str = "patient",  # "patient" | "window"
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 12,
    patient_map: Optional[Dict[str, str]] = None,
    patient_regex: Optional[str] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    leak_ratio: float = 0.0,
    # 仅训练集 water 过采样（向后兼容）；也可用 targets 参数来自定义
    oversample_water: bool = False,
    water_pos_ratio: float = 0.5,
    oversample_targets: Optional[Tuple[int, ...]] = None,  # None 表示用默认 (2,)
    # 时间步重采样
    resample_stride: Optional[int] = None,
    target_step_hours: Optional[float] = 1.0,
    orig_step_hours: float = 1.0,
    # 新增：滑窗步长，降低样本间相关性
    window_stride: int = 1,
    # 是否在构造后启用 RobustScaler（只拟合训练集）
    use_robust_scaler: bool = True,
    # 体检开关
    run_lint: bool = False,
):
    """构建 train/val/test 三个 DataLoader。
    - 若 oversample_water=True，将对 oversample_targets(默认water) 的“预测窗阳性样本”做加权采样。
    - window_stride>1 时，以更大步长滑窗，降低样本冗余与泄漏风险。
    - use_robust_scaler=True 时，使用训练集拟合中位数/IQR，对三分割统一规范化。
    """
    assert abs(sum(splits) - 1.0) < 1e-6, "splits 必须和为 1.0"
    rng = random.Random(seed)

    if run_lint:
        lint_data(Ts_data, base_data)

    # ---------- 决定重采样 stride ----------
    if resample_stride is None:
        if target_step_hours is not None and orig_step_hours > 0:
            resample_stride = max(1, int(round(float(target_step_hours) / float(orig_step_hours))))
        else:
            resample_stride = 1
    resample_stride = int(max(1, resample_stride))

    # --------- 工具函数 ---------
    def extract_patient_id(key: str, rec: dict) -> str:
        if isinstance(rec, dict) and ('patient_id' in rec):
            return str(rec['patient_id'])
        if patient_map is not None and key in patient_map:
            return str(patient_map[key])
        if patient_regex:
            m = re.search(patient_regex, key)
            if m:
                return m.group(1) if m.groups() else m.group(0)
        return key.split('_')[0].split('.')[0]

    def _safe_numeric_nd(x):
        if isinstance(x, pd.DataFrame):
            x = x.select_dtypes(include=[np.number]).to_numpy()
        elif isinstance(x, pd.Series):
            x = x.to_numpy()
        else:
            x = np.asarray(x)
        x = x.astype(np.float32, copy=False)
        return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

    def _has_nan_baseline(b):
        try:
            arr = _safe_numeric_nd(b)
            return np.isnan(arr).any()
        except Exception:
            return True

    def _downsample_blocks_mean(arr: np.ndarray, k: int) -> np.ndarray:
        if k <= 1:
            return arr.astype(np.float32, copy=False)
        T = len(arr)
        T2 = (T // k) * k
        if T2 == 0:
            return np.zeros((0, arr.shape[1]), dtype=np.float32)
        arr2 = arr[:T2].reshape(T2 // k, k, arr.shape[1])
        return arr2.mean(axis=1).astype(np.float32, copy=False)

    def _downsample_labels_by_volume(label: np.ndarray, k: int) -> np.ndarray:
        if k <= 1:
            return label.astype(np.float32, copy=False)
        T = len(label)
        T2 = (T // k) * k
        if T2 == 0:
            return np.zeros((0, label.shape[1]), dtype=np.float32)
        lab = label[:T2].reshape(T2 // k, k, label.shape[1]).astype(np.float32, copy=False)
        bits = lab[..., :3]
        spd  = lab[..., 3:6]
        block_inst_vol = (bits * spd).sum(axis=1)          # (B, 3)
        block_bits = (bits.max(axis=1) > 0.5).astype(np.float32)
        block_speed = (block_inst_vol / float(k)) * (block_bits > 0.5)
        if label.shape[1] > 6:
            extra = lab[..., 6:].mean(axis=1)
            out = np.concatenate([block_bits, block_speed, extra], axis=-1)
        else:
            out = np.concatenate([block_bits, block_speed], axis=-1)
        return out.astype(np.float32, copy=False)

    def _recompute_inst_cum_from_label(label_ds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(label_ds) == 0:
            z = np.zeros((0, 4), dtype=np.float32)
            return z, z
        bits = label_ds[:, :3]
        spd  = label_ds[:, 3:6]
        inst = np.stack([
            bits[:, 0] * spd[:, 0],
            bits[:, 1] * spd[:, 1],
            bits[:, 2] * spd[:, 2],
            bits[:, 0] * spd[:, 0] + bits[:, 1] * spd[:, 1] + bits[:, 2] * spd[:, 2],
        ], axis=1).astype(np.float32)
        cum = inst.cumsum(axis=0)
        return inst, cum

    # ---------- DataLoader 工具 ----------
    def make_loader(dataset, shuffle_flag: bool, sampler=None):
        if dataset is None:
            return None
        try:
            n = len(dataset)
        except Exception:
            n = 0
        if n == 0:
            return None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None) and shuffle_flag,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    # ---------- 时序增强器 ----------
    augmenter = TSAugmenter(
        p_timewarp=0.2, max_stretch=0.15,
        p_scale=0.2,   scale_sigma=0.05,
        p_jitter=0.2,  jitter_sigma=0.01,
        p_perm=0.2,    n_perm_segs=3,
        p_magwarp=0.2, magwarp_sigma=0.15,
    )

    # ---------- 样本构建 ----------
    def make_one_series_windows(key, rec, count_vec, bucket_lists):
        data_raw = rec['tdata']
        label_src = rec['label']
        label = label_src.values.tolist() if isinstance(label_src, pd.DataFrame) else label_src
        if len(label) != len(data_raw):
            L = min(len(label), len(data_raw))
            data_raw = data_raw[:L]
            label = label[:L]
        label = np.asarray(label, dtype=np.float32)

        # 统计 bit 组合（仍基于原始 1h）
        for l in label:
            count_vec[int(l[0] + l[1]*2 + l[2]*4)] += 1

        # 构造原始特征矩阵 -> 先拼 [t_norm] + 原特征
        data_mat = np.hstack([
            np.linspace(0, 1, len(data_raw), dtype=np.float32).reshape(-1, 1),
            np.asarray(data_raw, dtype=np.float32)
        ]).astype(np.float32)
        # 选 9 列
        data_sel = data_mat[:, [0, 1, 2, 3, 4, 5, 7, 8, 9]].astype(np.float32)

        # 重采样（按体积守恒）
        if resample_stride > 1:
            label_ds = _downsample_labels_by_volume(label, resample_stride)
            data_ds  = _downsample_blocks_mean(data_sel, resample_stride)
        else:
            label_ds = label.astype(np.float32, copy=False)
            data_ds  = data_sel

        # 重设时间归一化列
        if len(data_ds) > 0:
            data_ds[:, 0] = np.linspace(0, 1, len(data_ds), dtype=np.float32)

        # 再算 (inst, cum) 保证一致
        inst_ds, cum_ds = _recompute_inst_cum_from_label(label_ds)
        data_final = np.hstack([data_ds, cum_ds]).astype(np.float32)  # (T2, 13)

        # 体积保真提示（可选）
        vol_raw = float((label[:, :3] * label[:, 3:6]).sum())
        vol_ds = float(inst_ds.sum())
        if vol_raw > 0:
            drift = abs(vol_ds - vol_raw) / (vol_raw + 1e-6)
            if drift > 0.02:  # >2% 提示
                print(f"[WARN][{key}] volume drift after resample: {drift*100:.2f}%")

        T = len(data_final)
        key_base = key.split('.')[0]
        if key_base not in base_data or _has_nan_baseline(base_data[key_base]):
            print(f"[WARN] Skip {key_base}: baseline missing/NaN")
            return

        datas_lst, labels_lst, base_lst = bucket_lists
        step = max(1, int(window_stride))
        for i in range(0, T - history_length - pred_length + 1, step):
            x_win = data_final[i:i + history_length + pred_length]
            y_win = label_ds[i:i + history_length + pred_length]
            if (len(x_win) != history_length + pred_length or
                len(y_win) != history_length + pred_length):
                continue
            if np.isnan(x_win).any() or np.isnan(y_win).any():
                continue
            datas_lst.append(x_win)
            labels_lst.append(y_win)
            base_lst.append(base_data[key_base])

    # =============================================================================
    # A) 按“病人”划分（推荐）
    # =============================================================================
    if split_mode.lower() == "patient":
        patient_to_keys: Dict[str, List[str]] = {}
        for key in Ts_data.keys():
            pid = extract_patient_id(key, Ts_data[key])
            patient_to_keys.setdefault(pid, []).append(key)

        patients = list(patient_to_keys.keys())
        rng.shuffle(patients)
        n_pat = len(patients)
        n_train = int(round(splits[0] * n_pat))
        n_val   = int(round(splits[1] * n_pat))
        n_test  = max(0, n_pat - n_train - n_val)
        train_p = set(patients[:n_train])
        val_p   = set(patients[n_train:n_train+n_val])
        test_p  = set(patients[n_train+n_val:])

        datas_train, labels_train, base_train = [], [], []
        datas_val,   labels_val,   base_val   = [], [], []
        datas_test,  labels_test,  base_test  = [], [], []
        count_total = [0]*8
        count_train = [0]*8
        count_val   = [0]*8
        count_test  = [0]*8

        for pid in patients:
            keys = patient_to_keys[pid]
            if pid in train_p:
                for key in keys:
                    make_one_series_windows(key, Ts_data[key], count_train, (datas_train, labels_train, base_train))
            elif pid in val_p:
                for key in keys:
                    make_one_series_windows(key, Ts_data[key], count_val, (datas_val, labels_val, base_val))
            else:
                for key in keys:
                    make_one_series_windows(key, Ts_data[key], count_test, (datas_test, labels_test, base_test))

        for i in range(8):
            count_total[i] = count_train[i] + count_val[i] + count_test[i]

        train_dataset = CustomDataset(datas_train, labels_train, base_train, history_length, transform=augmenter)
        val_dataset   = CustomDataset(datas_val,   labels_val,   base_val,   history_length, transform=None)
        test_dataset  = CustomDataset(datas_test,  labels_test,  base_test,  history_length, transform=None)

        # 过采样（默认 targets=None -> (2,)）
        train_sampler = None
        if oversample_water and len(labels_train) > 0:
            targets = oversample_targets if oversample_targets is not None else (2,)
            train_sampler = _make_multi_target_sampler(labels_train, pred_length, targets=targets, pos_ratio=water_pos_ratio)

        train_loader = make_loader(train_dataset, shuffle_flag=(len(train_dataset) > 0), sampler=train_sampler)
        val_loader   = make_loader(val_dataset,   shuffle_flag=False)
        test_loader  = make_loader(test_dataset,  shuffle_flag=False)

        # RobustScaler（只拟合训练集）
        if use_robust_scaler and train_dataset is not None and len(train_dataset) > 0:
            sample_n = min(len(train_dataset), 2000)
            X_train_list = [np.array(train_dataset[i][0]) for i in range(sample_n)]
            scaler = RobustScaler(); scaler.fit(X_train_list)

            def _compose_transform(*fns):
                def f(x,b,y):
                    for fn in fns:
                        x,b,y = fn(x,b,y)
                    return x,b,y
                return f

            # 训练：增强 + 规范化（可按需调整顺序）
            if isinstance(train_dataset, CustomDataset):
                train_dataset.transform = _compose_transform(scaler, train_dataset.transform) if train_dataset.transform else scaler
            # 验证/测试：只做规范化
            for D in (val_dataset, test_dataset):
                if isinstance(D, CustomDataset):
                    D.transform = _compose_transform(scaler, D.transform) if D.transform else scaler

        # 打印信息
        print("="*70)
        print(f"[Split: PATIENT] 病人数: {n_pat}  ->  train/val/test: {len(train_p)}/{len(val_p)}/{len(test_p)}")
        print(f"resample_stride={resample_stride} (orig={orig_step_hours}h → target≈{resample_stride*orig_step_hours:.2f}h)")
        print(f"window_stride={window_stride}")
        print(f"类别分布（总，0..7）：{count_total}")
        print(f"类别分布（train）：{count_train}")
        print(f"类别分布（val）  ：{count_val}")
        print(f"类别分布（test） ：{count_test}")
        n_tr = len(train_dataset) if train_loader is not None else 0
        n_va = len(val_dataset)   if val_loader   is not None else 0
        n_te = len(test_dataset)  if test_loader  is not None else 0
        print(f"训练样本数: {n_tr} | 验证: {n_va} | 测试: {n_te}")
        if oversample_water and train_sampler is not None:
            print(f"[Oversample] targets={oversample_targets if oversample_targets else (2,)} pos_ratio={water_pos_ratio:.2f} (WeightedRandomSampler)")
        if train_loader is not None and n_tr > 0:
            x0, b0, y0 = train_dataset[0]
            print(f"单个样本 shape: X={tuple(x0.shape)}, B={tuple(b0.shape)}, Y={tuple(y0.shape)}")
        print("="*70)

        return train_loader, val_loader, test_loader

    # =============================================================================
    # B) 按“窗口”划分
    # =============================================================================
    else:
        datas, labels, baseline = [], [], []
        count = [0]*8

        for key in Ts_data.keys():
            make_one_series_windows(key, Ts_data[key], count, (datas, labels, baseline))

        dataset     = CustomDataset(datas, labels, baseline, history_length, transform=None)
        dataset_aug = CustomDataset(datas, labels, baseline, history_length, transform=augmenter)

        total_size = len(dataset)
        n_train = int(splits[0] * total_size)
        n_val   = int(splits[1] * total_size)
        n_test  = total_size - n_train - n_val

        g = torch.Generator().manual_seed(seed)
        _train, _val, _test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test], generator=g)

        # 训练用增强
        train_dataset = Subset(dataset_aug, _train.indices)
        val_dataset   = Subset(dataset,     _val.indices)
        test_dataset  = Subset(dataset,     _test.indices)

        # 验证泄漏回流训练（默认0.0）
        if leak_ratio > 0 and len(_val.indices) > 0:
            leak_sz = int(round(len(_val.indices) * leak_ratio))
            if leak_sz > 0:
                leak_idx = _val.indices[:leak_sz]
                val_dataset = Subset(dataset, _val.indices[leak_sz:])
                train_dataset = Subset(dataset_aug, list(train_dataset.indices) + list(leak_idx))

        # 过采样（窗口划分按索引）
        train_sampler = None
        if oversample_water and len(_train.indices) > 0:
            labels_train = [labels[i] for i in _train.indices]
            targets = oversample_targets if oversample_targets is not None else (2,)
            train_sampler = _make_multi_target_sampler(labels_train, pred_length, targets=targets, pos_ratio=water_pos_ratio)

        train_loader = make_loader(train_dataset, shuffle_flag=(len(train_dataset) > 0), sampler=train_sampler)
        val_loader   = make_loader(val_dataset,   shuffle_flag=False)
        test_loader  = make_loader(test_dataset,  shuffle_flag=False)

        # RobustScaler
        if use_robust_scaler and train_dataset is not None and len(train_dataset) > 0:
            # 注意：Subset 里 dataset_aug 才有 transform
            base_ds: CustomDataset = dataset_aug
            sample_idx = _train.indices[:min(len(_train.indices), 2000)]
            X_train_list = [np.array(base_ds[i][0]) for i in sample_idx]
            scaler = RobustScaler(); scaler.fit(X_train_list)

            def _compose_transform(*fns):
                def f(x,b,y):
                    for fn in fns:
                        x,b,y = fn(x,b,y)
                    return x,b,y
                return f

            # 给底层 dataset_aug/dataset 设置 transform
            base_ds.transform = _compose_transform(scaler, base_ds.transform) if base_ds.transform else scaler
            dataset.transform = scaler

        # 打印信息
        print("=" * 70)
        print(f"[Split: WINDOW] 类别分布（0..7）：{count}")
        print(f"resample_stride={resample_stride} (orig={orig_step_hours}h → target≈{resample_stride*orig_step_hours:.2f}h)")
        print(f"window_stride={window_stride}")
        n_tr = len(train_dataset) if train_loader is not None else 0
        n_va = len(val_dataset)   if val_loader   is not None else 0
        n_te = len(test_dataset)  if test_loader  is not None else 0
        print(f"总样本数: {len(dataset)}")
        print(f"训练集样本数: {n_tr} | 验证: {n_va} | 测试: {n_te}")
        if oversample_water and train_sampler is not None:
            print(f"[Oversample] targets={oversample_targets if oversample_targets else (2,)} pos_ratio={water_pos_ratio:.2f} (WeightedRandomSampler)")
        if train_loader is not None and n_tr > 0:
            # dataset_aug + Subset: 需要通过 indices 访问
            idx0 = _train.indices[0]
            x0, b0, y0 = dataset_aug[idx0]
            print(f"单个样本 shape: X={tuple(x0.shape)}, B={tuple(b0.shape)}, Y={tuple(y0.shape)}")
        print("=" * 70)

        return train_loader, val_loader, test_loader


# =========================
# Quick self-test
# =========================
if __name__ == '__main__':
    # 仅示例；工程中可忽略
    try:
        with open('data/processed_data.pkl', 'rb') as f:
            Ts_data = pickle.load(f)
        with open('data/processed_basedata.pkl', 'rb') as f:
            Base_data = pickle.load(f)
        print("Pickle data loaded successfully. Keys(Ts):", list(Ts_data.keys())[:5])
        print("Pickle data loaded successfully. Keys(Base):", list(Base_data.keys())[:5])

        # 可先体检
        lint_data(Ts_data, Base_data)

        train_loader, val_loader, test_loader = data_loader(
            Ts_data, Base_data,
            history_length=11, pred_length=1, batch_size=32,
            split_mode="patient", splits=(0.8, 0.1, 0.1),
            oversample_water=True, water_pos_ratio=0.5,
            target_step_hours=2.0, orig_step_hours=1.0,  # 自测两小时步长
            window_stride=2, use_robust_scaler=True,
            run_lint=False,
        )

        def _safe_len(loader):
            return 0 if loader is None else (len(loader.dataset) if hasattr(loader, 'dataset') else 0)

        print(f"Train dataset size: {_safe_len(train_loader)}")
        print(f"Validation dataset size: {_safe_len(val_loader)}")
        print(f"Test dataset size: {_safe_len(test_loader)}")
    except Exception as e:
        print(f"[SELF-TEST WARN] {e}")
