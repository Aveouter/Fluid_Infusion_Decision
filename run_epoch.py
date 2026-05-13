# -*- coding: utf-8 -*-
import logging
import math
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, roc_auc_score,
    mean_absolute_error, roc_curve
)
import matplotlib.pyplot as plt

# ---------------- 分布式标识（可选） ----------------
try:
    from dist_utils import is_main_process
except Exception:
    def is_main_process(): return True

# ---------------- 你的工具函数 ----------------
from utils.util import (
    _safe_div, fbeta_opt_thresh_per_class, smooth_probs_over_time,
    smape, rmse, tolerance_accuracy_abs, tolerance_accuracy_pct,
)

# =============================================================================
# 动态 Loss 权重器：EMA / DWA / 不确定性加权（Kendall & Gal）
# =============================================================================
class LossBalancer:
    """
    动态任务权重：
      mode = "ema" | "dwa" | "uncertainty"
    - ema: 逐 step 用 EMA 平衡子损失，使 w_i * EMA(L_i) ≈ target_ratio_i
    - dwa: 逐 epoch 用 DWA 公式按最近两轮损失变化分配权重
    - uncertainty: Kendall&Gal 不确定性加权（需要把 log_sigma 参数加入优化器）
    """
    def __init__(self,
                 keys=("cls","reg","cum"),
                 base_lambda=dict(cls=1.0, reg=1.0, cum=0.01),
                 target_ratio=dict(cls=1.0, reg=1.0, cum=0.02),
                 mode="ema",
                 ema_beta=0.98,
                 dwa_T=2.0,
                 clamp=(1e-4, 1e2)):
        self.keys = list(keys)
        self.base = {k: float(base_lambda.get(k, 1.0)) for k in self.keys}
        self.target = {k: float(target_ratio.get(k, 1.0)) for k in self.keys}
        self.mode = str(mode)
        self.ema_beta = float(ema_beta)
        self.clamp = clamp
        # EMA 状态
        self.ema = {k: None for k in self.keys}
        # DWA 历史（按 epoch）
        self.hist = {k: deque(maxlen=2) for k in self.keys}
        self.dwa_T = float(dwa_T)
        # 不确定性：可学习 log_sigma（默认未加入到任何优化器）
        self.log_sigma = {k: torch.nn.Parameter(torch.zeros(1)) for k in self.keys}

    def step_weights(self, raw_losses: dict, is_epoch_end: bool=False):
        """
        输入：子损失当前数值（float 或 0D tensor）
        输出：用于本步/本轮组合的权重字典（与 base 同量级）
        """
        vals = {k: float(raw_losses.get(k, 0.0)) for k in self.keys}
        lo, hi = self.clamp

        if self.mode == "ema":
            # 更新 EMA
            for k, v in vals.items():
                if not math.isfinite(v):
                    continue
                v = max(v, 1e-12)
                if self.ema[k] is None:
                    self.ema[k] = v
                else:
                    self.ema[k] = self.ema_beta * self.ema[k] + (1 - self.ema_beta) * v
            # 令 w_k ∝ target_k / EMA(L_k)
            raw_w = {k: (self.target[k] / max(self.ema[k] or 1e-12, 1e-12)) for k in self.keys}
            # 归一到与 base 的 L1 相同量级，避免初期爆
            base_sum = sum(abs(self.base[k]) for k in self.keys)
            w_sum = sum(max(v, 0.0) for v in raw_w.values())
            if w_sum <= 0:
                lam = self.base.copy()
            else:
                lam = {k: max(lo, min(hi, raw_w[k] * base_sum / w_sum)) for k in self.keys}
            return lam

        elif self.mode == "dwa":
            # 建议在 epoch 末尾推进历史（调用时传 is_epoch_end=True）
            if is_epoch_end:
                for k in self.keys:
                    self.hist[k].append(max(vals.get(k, 0.0), 1e-12))
            # 需要至少两轮才生效
            r = {}
            for k in self.keys:
                if len(self.hist[k]) < 2:
                    r[k] = 1.0
                else:
                    l_t, l_t_1 = self.hist[k][-1], self.hist[k][-2]
                    r[k] = l_t / max(l_t_1, 1e-12)
            # softmax 分配
            exps = {k: math.exp(r[k] / self.dwa_T) for k in self.keys}
            denom = sum(exps.values())
            raw_w = {k: exps[k] / denom for k in self.keys}
            base_sum = sum(abs(self.base[k]) for k in self.keys)
            lam = {k: max(lo, min(hi, raw_w[k] * base_sum)) for k in self.keys}
            return lam

        elif self.mode == "uncertainty":
            # 仅用于日志展示；真正组合应调用 weigh_uncertainty()
            lam = {k: float(torch.exp(-self.log_sigma[k]).item()) for k in self.keys}
            lam = {k: max(lo, min(hi, lam[k])) for k in self.keys}
            return lam

        else:
            return self.base.copy()

    def weigh_uncertainty(self, loss_dict: dict):
        """
        仅在 mode='uncertainty' 时使用：
          Σ ( L_k * exp(-s_k) + s_k )
        """
        s = 0.0
        for k in self.keys:
            lk = loss_dict.get(k, None)
            if lk is None:
                continue
            s += lk * torch.exp(-self.log_sigma[k]) + self.log_sigma[k]
        return s

# ---------------- 低阶工具 ----------------
def _dtw_distance_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Classic DTW for 1D sequences (O(T^2))."""
    T1, T2 = len(x), len(y)
    D = np.full((T1 + 1, T2 + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, T1 + 1):
        xi = x[i - 1]
        for j in range(1, T2 + 1):
            cost = abs(xi - y[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[T1, T2])

def _pearsonr_numpy(y_true_1d: np.ndarray, y_pred_1d: np.ndarray) -> float:
    mask = np.isfinite(y_true_1d) & np.isfinite(y_pred_1d)
    y1 = y_true_1d[mask]; y2 = y_pred_1d[mask]
    if y1.size < 2: return 0.0
    s1, s2 = np.std(y1), np.std(y2)
    if np.isclose(s1, 0.0) or np.isclose(s2, 0.0): return 0.0
    r = np.corrcoef(y1, y2)[0, 1]
    return float(r) if np.isfinite(r) else 0.0

def _r2_numpy(y_true_1d: np.ndarray, y_pred_1d: np.ndarray) -> float:
    mask = np.isfinite(y_true_1d) & np.isfinite(y_pred_1d)
    y1 = y_true_1d[mask]; y2 = y_pred_1d[mask]
    if y1.size < 2: return 0.0
    ss_tot = np.sum((y1 - y1.mean())**2)
    if np.isclose(ss_tot, 0.0): return 0.0
    ss_res = np.sum((y1 - y2)**2)
    r2 = 1.0 - ss_res / ss_tot
    return float(r2) if np.isfinite(r2) else 0.0

def _safe_mean(x: torch.Tensor, default: float = 0.0) -> torch.Tensor:
    if not isinstance(x, torch.Tensor) or x.numel() == 0:
        return torch.tensor(default)
    x2 = torch.nan_to_num(x, nan=0., posinf=0., neginf=0.)
    m = x2.mean()
    return m if torch.isfinite(m) else torch.tensor(default, device=x.device, dtype=x.dtype)

def _safe_item(x: torch.Tensor, default: float = 0.0) -> float:
    if not isinstance(x, torch.Tensor) or x.numel() == 0:
        return default
    v = torch.nan_to_num(x.detach().reshape(-1)[0], nan=0., posinf=0., neginf=0.)
    return float(v.item())

def huber(x, delta: float):
    a = torch.abs(x)
    inside = 0.5 * (a * a) / max(delta, 1e-8)
    outside = a - 0.5 * delta
    h = torch.where(a <= delta, inside, outside)
    return torch.nan_to_num(h, nan=0., posinf=0., neginf=0.)

def _bce_logits(logits, targets, reduction='mean'):
    return F.binary_cross_entropy_with_logits(logits, targets, reduction=reduction)

def _reg_l1_pos_only(pred: torch.Tensor, target: torch.Tensor, zero_eps: float = 1e-6) -> torch.Tensor:
    mask = (target > zero_eps)
    if not mask.any(): return pred.new_tensor(0.0)
    return F.l1_loss(pred[mask], target[mask])

def _zero_penalty(pred: torch.Tensor, target: torch.Tensor, zero_eps: float = 1e-6) -> torch.Tensor:
    mask0 = (target <= zero_eps)
    if not mask0.any(): return pred.new_tensor(0.0)
    return torch.mean(torch.clamp(pred[mask0], 0.0, zero_eps * 5))


# ============================== 主循环 ==============================
def run_epoch(
    model,
    data_loader,
    optimizer,                   # 训练传优化器，验证/测试传 None
    criterion_classification,    # 占位
    criterion_regression,        # 占位
    device,
    args,
    is_train: bool = True,
    epoch: int = 0,
    draw_roc: bool = True,      # 新增：是否保存 ROC 曲线（一般 test 时 True）
):
    model.train() if is_train else model.eval()
    torch.backends.cudnn.benchmark = True

    name     = str(getattr(args, "model", "")).lower()
    is_lstm  = (name == "lstm")
    C        = int(getattr(args, "num_classes", 3))
    P        = int(getattr(args, "pred_length", 1))
    idx_map  = [("crystal", 0), ("colloid", 1), ("water", 2)][:C]

    # 物理/损失相关
    max_speed       = float(getattr(args, "max_speed", 1200.0))
    zero_target_eps = float(getattr(args, "zero_target_eps", 1e-6))
    delta_ml        = float(getattr(args, "delta_ml", 1.0))
    cum_tol_ml      = float(getattr(args, "cum_tol_epsilon_ml", 300.0))

    # 验证指标相关
    smooth_win      = int(getattr(args, "smooth_win", 5))
    smooth_idx      = list(getattr(args, "smooth_classes", [C-1 if C>0 else 0]))
    tune_thr        = bool(getattr(args, "tune_thresholds", True))
    fixed_thr       = list(getattr(args, "fixed_thresholds", [0.5]*C))
    fbeta_list      = list(getattr(args, "fbeta_per_class", [1.0]*C))

    # AE/RE 归一化用：体重、TSBA 所在列（在 base_data 中）
    weight_col = getattr(args, "weight_col", 3)
    tsba_col   = getattr(args, "tsba_col", 5)

    # AMP
    use_amp = bool(getattr(args, "use_amp", False))
    scaler  = getattr(args, "amp_scaler", None)

    # ---------------- 动态权重器（初始化一次并复用到 args） ----------------
    if not hasattr(args, "loss_balancer"):
        args.loss_balancer = LossBalancer(
            keys=("cls","reg","cum"),
            base_lambda=dict(
                cls=float(getattr(args, "lambda_cls", 1.0)),
                reg=float(getattr(args, "lambda_reg", 1.0)),
                cum=float(getattr(args, "lambda_cum", 0.01)),
            ),
            # 你希望的"有效贡献比例"，可按需要调整
            target_ratio=dict(cls=1.0, reg=1.0, cum=0.02),
            mode=str(getattr(args, "loss_balance_mode", "ema")),  # "ema"|"dwa"|"uncertainty"
            ema_beta=float(getattr(args, "ema_beta", 0.98)),
            dwa_T=float(getattr(args, "dwa_T", 2.0)),
            clamp=(1e-4, 1e2),
        )
    balancer = args.loss_balancer

    # 累计器
    sums_raw = {"cls": 0.0, "reg": 0.0, "cum": 0.0}
    total_sum = 0.0
    correct_bits = 0
    total_bits   = 0

    # per-class 训练期统计
    cls_loss_elem_sum = np.zeros(C, dtype=np.float64)
    cls_loss_elem_cnt = np.zeros(C, dtype=np.int64)
    cls_correct_cnt   = np.zeros(C, dtype=np.int64)
    cls_total_cnt     = np.zeros(C, dtype=np.int64)

    # 验证/测试缓存
    all_labels_ids = []
    all_prob_raw_list = []
    all_prob_smooth_list = []
    all_flow_preds, all_flow_labels = [], []

    # 为 nDTW：缓存总瞬时序列（各类求和）与最终累计真值
    all_rate_seq_total = []   # list of (B,P)  —— rate = prob*speed
    all_true_seq_total = []   # list of (B,P)  —— inst_true = y_type*y_flow

    # 为 AE/RE 表：缓存每个样本各类最终累计量
    all_true_cum_cls = []   # list of (B,C)
    all_pred_cum_cls = []   # list of (B,C)

    # 体重 / TSBA（如果有）
    all_weight = []
    all_tsba   = []

    collect_eval = (not is_train)
    desc = f"{'Train' if is_train else 'Val/Test'} Ep{epoch}"
    pbar = tqdm(data_loader, total=len(data_loader), desc=desc, leave=False, disable=not is_main_process())

    with torch.set_grad_enabled(is_train):
        for step, (ts_data, base_data, label_data) in enumerate(pbar):
            ts  = torch.nan_to_num(ts_data.to(device, non_blocking=True), nan=0., posinf=0., neginf=0.)
            bas = torch.nan_to_num(base_data.to(device, non_blocking=True), nan=0., posinf=0., neginf=0.)
            lab = torch.nan_to_num(label_data.to(device, non_blocking=True), nan=0., posinf=0., neginf=0.)

            # 输入构造
            if is_lstm:
                inp_T, inp_B = ts, bas
            else:
                if ts.size(1) <= P or lab.size(1) <= P:
                    inp_T = torch.cat((ts[:, :-1, :], lab[:, :-1, :]), dim=-1)
                else:
                    inp_T = torch.cat((ts[:, :-P, :], lab[:, :-P, :]), dim=-1)
                inp_T = torch.nan_to_num(inp_T, nan=0., posinf=0., neginf=0.)
                inp_B = bas

            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            # forward
            ctx = torch.amp.autocast('cuda', enabled=(use_amp and device.type == "cuda"), dtype=torch.float16)
            with ctx:
                outs = model(inp_T, inp_B)
                if isinstance(outs, dict):
                    logits = outs.get("class_logits", None)  # (B,P,C)
                    flow_p = outs.get("flow_pred",   None)   # (B,P,C)
                else:
                    flow_p, logits, *_ = outs if isinstance(outs, (list, tuple)) else (None, None)
                if logits is None or flow_p is None or logits.size(-1) != C:
                    raise RuntimeError("model 必须输出 (class_logits[B,P,C], flow_pred[B,P,C])，且 C 一致")

                # 标签窗口对齐
                if is_lstm:
                    y_type = lab[:, :, :C]
                    y_flow = lab[:, :, C:]
                else:
                    y_type = lab[:, -P:, :C]
                    y_flow = lab[:, -P:, C:]

                prob   = torch.sigmoid(logits)                     # (B,P,C)
                speed  = F.softplus(flow_p).clamp(0.0, max_speed)  # (B,P,C)
                rate   = torch.clamp(prob * speed, 0.0, max_speed) # (B,P,C)
                inst_true = torch.clamp(y_type * y_flow, 0.0, max_speed)

                # 子损失
                L_cls = _bce_logits(logits, y_type.float(), reduction='mean')
                L_reg = _reg_l1_pos_only(speed, y_flow, zero_eps=zero_target_eps)
                L_zero = _zero_penalty(speed, y_flow, zero_eps=zero_target_eps)
                L_reg = L_reg + 0.05 * L_zero

                cum_true  = torch.cumsum(inst_true * delta_ml, dim=1).sum(-1)  # (B,P) -> sum C
                cum_pred  = torch.cumsum(rate      * delta_ml, dim=1).sum(-1)
                L_cum     = _safe_mean(huber(cum_pred - cum_true, delta=cum_tol_ml), 0.0)

                # ---------------- 组合损失：动态权重 ----------------
                if balancer.mode == "uncertainty":
                    L = balancer.weigh_uncertainty({"cls": L_cls, "reg": L_reg, "cum": L_cum})
                    lam_view = balancer.step_weights({"cls": _safe_item(L_cls), "reg": _safe_item(L_reg), "cum": _safe_item(L_cum)})
                else:
                    lam_view = balancer.step_weights({"cls": _safe_item(L_cls), "reg": _safe_item(L_reg), "cum": _safe_item(L_cum)})
                    L = lam_view["cls"] * L_cls + lam_view["reg"] * L_reg + lam_view["cum"] * L_cum

                L = torch.nan_to_num(L, nan=0., posinf=0., neginf=0.)

            # backward
            if is_train and optimizer is not None:
                if use_amp and scaler is not None and device.type == "cuda":
                    scaler.scale(L).backward()
                    if getattr(args, "grad_clip", None):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                    scaler.step(optimizer); scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    L.backward()
                    if getattr(args, "grad_clip", None):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                    optimizer.step()

            # 统计
            sums_raw["cls"] += _safe_item(L_cls, 0.0)
            sums_raw["reg"] += _safe_item(L_reg, 0.0)
            sums_raw["cum"] += _safe_item(L_cum, 0.0)
            total_sum += _safe_item(L, 0.0)

            with torch.no_grad():
                bce_all = F.binary_cross_entropy_with_logits(logits, y_type.float(), reduction='none')  # (B,P,C)
                bce_sum = bce_all.sum(dim=(0, 1))
                for _, i in idx_map:
                    cls_loss_elem_sum[i] += float(bce_sum[i].item())
                    cls_loss_elem_cnt[i] += int(bce_all[:,:,i].numel())
                pred_bits_05 = (prob > 0.5)
                correct_c = (pred_bits_05 == y_type).sum(dim=(0,1))
                for _, i in idx_map:
                    cls_correct_cnt[i] += int(correct_c[i].item())
                    cls_total_cnt[i]   += int(y_type[:,:,i].numel())

                bit = (prob > 0.5).long()
                correct_bits += int((bit == y_type.long()).sum().item())
                total_bits   += int(y_type.numel())

                # 评估期缓存
                if collect_eval:
                    all_labels_ids.append(y_type.detach().cpu())
                    all_prob_raw_list.append(prob.detach().cpu())
                    probs_for_pred = prob
                    if smooth_win > 1:
                        probs_for_pred = smooth_probs_over_time(probs_for_pred, win=smooth_win, only_idx=smooth_idx)
                    all_prob_smooth_list.append(probs_for_pred.detach().cpu())

                    # 回归评估使用 speed（条件均值，且仅 y_true>0 参与 MAE/RMSE/R/R2）
                    flow_eval = speed
                    all_flow_preds.append(flow_eval.detach().cpu().reshape(-1))
                    all_flow_labels.append(y_flow.detach().cpu().reshape(-1))

                    # nDTW：汇总各类 -> (B,P) 序列：rate 与 inst_true
                    total_rate = rate.sum(dim=-1)       # (B,P)
                    total_true = inst_true.sum(dim=-1)  # (B,P)
                    all_rate_seq_total.append(total_rate.detach().cpu())
                    all_true_seq_total.append(total_true.detach().cpu())

                    # —— 用于 AE/RE 表：各类的最终累计量（按 delta_ml 积分）——
                    true_cum_cls = (inst_true * delta_ml).sum(dim=1)  # (B,C)
                    pred_cum_cls = (rate      * delta_ml).sum(dim=1)  # (B,C)
                    all_true_cum_cls.append(true_cum_cls.detach().cpu())
                    all_pred_cum_cls.append(pred_cum_cls.detach().cpu())

                    # 体重 / TSBA（如果定义了列）
                    if weight_col is not None:
                        w = bas[..., int(weight_col)]
                        all_weight.append(w.detach().cpu().reshape(-1))
                    if tsba_col is not None:
                        t = bas[..., int(tsba_col)]
                        all_tsba.append(t.detach().cpu().reshape(-1))


            pbar.set_postfix({
                "tot": f"{_safe_item(L):.4f}",
                "cls": f"{_safe_item(L_cls):.4f}",
                "acc": f"{(correct_bits/max(1,total_bits)):.3f}",
            })

        # DWA 在 epoch 末推进历史（可选）
        if balancer.mode == "dwa":
            _ = balancer.step_weights({"cls": sums_raw["cls"], "reg": sums_raw["reg"], "cum": sums_raw["cum"]}, is_epoch_end=True)

    # ---- epoch 聚合 ----
    denom = max(1, len(data_loader))
    avg_raw = {k: v / denom for k, v in sums_raw.items()}
    avg_total = total_sum / denom
    bit_acc   = (correct_bits / total_bits) if total_bits > 0 else 0.0

    per_class = {name: {} for name, _ in idx_map}
    for name_c, i in idx_map:
        per_class[name_c]["cls_loss"] = _safe_div(cls_loss_elem_sum[i], cls_loss_elem_cnt[i], default=0.0)
        per_class[name_c]["cls_acc"]  = _safe_div(cls_correct_cnt[i],   cls_total_cnt[i],   default=0.0)
        per_class[name_c].update(dict(precision=0.0, recall=0.0, f1=0.0, auc=0.0))

    # ---- 评估指标 ----
    best_thr = list(fixed_thr)
    f1_macro = prec_macro = rec_macro = auc_macro = 0.0
    mae_all = mse_all = rmse_all = 0.0
    r_all = r2_all = 0.0
    mae_per_h = [0.0]*P
    rmse_per_h = [0.0]*P

    # nDTW + AE/RE
    dtw_avg = 0.0
    ndtw_mean = 0.0
    ndtw_p50  = 0.0
    ndtw_p90  = 0.0
    cum_true_mean = 0.0
    ae_mean = ae_p50 = ae_p90 = 0.0
    re_mean = re_p50 = re_p90 = 0.0

    # 三个单位下的 AE/RE 表
    ae_re_tables = {
        "ml": None,
        "ml_per_kg": None,
        "ml_per_kg_tsba": None,
        "by_day_type": None,
    }

    if collect_eval and all_labels_ids:
        y_true_t         = torch.cat(all_labels_ids,       dim=0)
        prob_raw_t       = torch.cat(all_prob_raw_list,    dim=0)
        prob_smooth_t    = torch.cat(all_prob_smooth_list, dim=0)

        y_true         = y_true_t.numpy().reshape(-1, C)
        prob_raw_np    = prob_raw_t.numpy().reshape(-1, C)
        prob_smooth_np = prob_smooth_t.numpy().reshape(-1, C)

        if tune_thr:
            best_thr = fbeta_opt_thresh_per_class(y_true, prob_smooth_np, betas=fbeta_list)
        thr_arr = np.array(best_thr).reshape(1, -1)
        y_pred = (prob_smooth_np > thr_arr).astype(int)

        for name_c, i in idx_map:
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true[:, i], y_pred[:, i], average='binary', zero_division=0)
            auc_val = 0.0
            if np.unique(y_true[:, i]).size == 2:
                try:
                    auc_val = float(roc_auc_score(y_true[:, i], prob_raw_np[:, i]))
                except Exception:
                    auc_val = 0.0
            per_class[name_c].update(precision=float(prec), recall=float(rec), f1=float(f1), auc=auc_val)

        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        prec_macro, rec_macro, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0)
        aucs = [per_class[name]["auc"] for name, _ in idx_map]
        auc_macro = float(np.mean(aucs)) if len(aucs) else 0.0

        # ===== ROC 曲线保存（按类，多曲线在一张图上） =====
        if draw_roc & False:
            try:
                # 科研风格：适当设置图像大小、字体等
                plt.figure(figsize=(5, 5), dpi=300)

                # 设置全局样式（可根据需要调整）
                plt.rcParams.update({
                    "font.size": 10,
                    "axes.linewidth": 1.0,
                    "xtick.direction": "in",
                    "ytick.direction": "in",
                    "xtick.major.size": 4,
                    "ytick.major.size": 4,
                    "xtick.minor.size": 2,
                    "ytick.minor.size": 2,
                    "axes.unicode_minus": False,
                })

                # 红绿蓝配色循环
                color_cycle = ["#DE3826", "#05BF78", "#4083BA"]
                curve_idx = 0

                for name_c, i in idx_map:
                    # 只画二分类的 ROC
                    if np.unique(y_true[:, i]).size == 2:
                        try:
                            fpr, tpr, _ = roc_curve(y_true[:, i], prob_raw_np[:, i])
                            auc_val = per_class[name_c]["auc"]
                            color = color_cycle[curve_idx % len(color_cycle)]
                            curve_idx += 1
                            if name_c == "water":
                                name_c = "Glucose"
                            elif name_c == "crystal":
                                name_c = "Crystalloid"
                            elif name_c == "colloid":
                                name_c = "Colloid"
                            plt.plot(
                                fpr,
                                tpr,
                                # label=f"{name_c} (AUC = {auc_val:.3f})",
                                color=color,
                                linewidth=1.8,
                            )
                        except Exception:
                            continue

                # 对角线基线
                plt.plot(
                    [0, 1],
                    [0, 1],
                    linestyle="--",
                    linewidth=1.0,
                    color="0.6",
                )

                # 坐标轴设置
                plt.xlim(0.0, 1.0)
                plt.ylim(0.0, 1.0)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")

                # 按要求：不要图名，所以不加 title
                # plt.title(f"ROC Curve (epoch {epoch})")

                # 图例：科研风格一般去掉边框，放在右下角
                plt.legend(
                    loc="lower right",
                    frameon=False,
                    fontsize=8,
                    handlelength=1.8,
                )

                # 细网格，一般科研图会用浅灰色网格
                plt.grid(
                    True,
                    linestyle=":",
                    linewidth=0.6,
                    alpha=0.7,
                )

                plt.tight_layout()

                save_path = getattr(args, "roc_save_path", None)
                if save_path is None:
                    mode_tag = "train" if is_train else "val"
                    save_path = f"roc_{mode_tag}_{name}.png"

                plt.savefig(save_path, bbox_inches="tight")
                plt.close()
                logging.info(f"ROC curve saved to: {save_path}")
            except Exception as e:
                logging.warning(f"Failed to save ROC curve: {e}")

        # 回归（仅 y_true>0）
        y_pred_reg = torch.cat(all_flow_preds,  dim=0).numpy()
        y_true_reg = torch.cat(all_flow_labels, dim=0).numpy()
        mask = np.isfinite(y_true_reg) & (y_true_reg > zero_target_eps)
        if mask.any():
            mae_all   = float(mean_absolute_error(y_true_reg[mask], y_pred_reg[mask]))
            mse_all   = float(np.mean((y_true_reg[mask] - y_pred_reg[mask]) ** 2))
            rmse_all  = float(np.sqrt(mse_all))
            r_all     = _pearsonr_numpy(y_true_reg[mask], y_pred_reg[mask])
            r2_all    = _r2_numpy(y_true_reg[mask], y_pred_reg[mask])

        # —— nDTW + AE/RE 统计：逐样本 DTW/最终累计真值 —— #
        if len(all_rate_seq_total) > 0:
            rate_mat = torch.cat(all_rate_seq_total, dim=0).numpy()  # (N, P)
            true_mat = torch.cat(all_true_seq_total, dim=0).numpy()  # (N, P)
            dtw_list, ndtw_list, final_list = [], [], []
            eps = 1e-6
            for i in range(rate_mat.shape[0]):
                a = rate_mat[i]
                b = true_mat[i]
                dtw_val = _dtw_distance_1d(a, b)
                final   = float(np.sum(b))
                dtw_list.append(dtw_val)
                final_list.append(final)
                ndtw_list.append(dtw_val / (final + eps))

            if len(dtw_list) > 0:
                dtw_arr  = np.array(dtw_list, dtype=np.float64)
                ndtw_arr = np.array(ndtw_list, dtype=np.float64)
                fin_arr  = np.array(final_list, dtype=np.float64)
                dtw_avg       = float(dtw_arr.mean())
                ndtw_mean     = float(ndtw_arr.mean())
                ndtw_p50      = float(np.percentile(ndtw_arr, 50))
                ndtw_p90      = float(np.percentile(ndtw_arr, 90))
                cum_true_mean = float(fin_arr.mean())

                # ---- 最终累计量的 AE / RE（基于 sum(rate), sum(true)）---- #
                ae_list, re_list = [], []
                for i in range(rate_mat.shape[0]):
                    true_sum = float(np.sum(true_mat[i]))
                    pred_sum = float(np.sum(rate_mat[i]))
                    ae = abs(pred_sum - true_sum)
                    re = ae / (true_sum + eps) if true_sum > 0 else 0.0
                    ae_list.append(ae)
                    re_list.append(re)

                if len(ae_list) > 0:
                    ae_arr = np.array(ae_list, dtype=np.float64)
                    re_arr = np.array(re_list, dtype=np.float64)
                    ae_mean      = float(ae_arr.mean())
                    ae_p50       = float(np.percentile(ae_arr, 50))
                    ae_p90       = float(np.percentile(ae_arr, 90))
                    re_mean      = float(re_arr.mean())
                    re_p50       = float(np.percentile(re_arr, 50))
                    re_p90       = float(np.percentile(re_arr, 90))

        # ========= 根据最终累计量，构建按天和类型组织的 AE/RE 表 =========
        try:
            if len(all_true_cum_cls) > 0:
                eps = 1e-6
                true_cum_mat = torch.cat(all_true_cum_cls, dim=0).numpy()   # (N,C)
                pred_cum_mat = torch.cat(all_pred_cum_cls, dim=0).numpy()   # (N,C)

                # 2. 处理体重和表面积数据
                weight_vec = None
                tsba_vec = None

                if len(all_weight) > 0:
                    weight_vec = torch.cat(all_weight, dim=0).numpy()
                    weight_vec = np.maximum(weight_vec, 35)  # 假设体重不低于35kg

                if len(all_tsba) > 0:
                    tsba_vec = torch.cat(all_tsba, dim=0).numpy()
                    tsba_vec = np.maximum(tsba_vec, eps)  # 防止TSBA为零或负值

                # 3. 根据样本顺序推算天数，假设总共有2天
                day_ids = np.floor(np.arange(true_cum_mat.shape[0]) / (true_cum_mat.shape[0] / 2)) + 1
                logging.info("No day information found, using sample index to determine days (2 days in total)")

                # 4. 获取唯一的天数
                unique_days = np.unique(day_ids)

                # 5. 创建按天和类型组织的表格
                table_by_day_type = {}

                # helper: 将一维向量按 24 聚合（sum for t/p, nanmean for weight/tsba）
                def aggregate_24_sum(vec):
                    pad = (-len(vec)) % 24
                    if pad > 0:
                        vec_p = np.pad(vec, (0, pad), mode='constant', constant_values=0.0)
                    else:
                        vec_p = vec
                    return vec_p.reshape(-1, 24).sum(axis=1)

                def aggregate_24_nanmean(vec):
                    # 用 NaN 填充，最后用 nanmean（避免 pad=0 拉低平均）
                    pad = (-len(vec)) % 24
                    if pad > 0:
                        vec_p = np.pad(vec, (0, pad), mode='constant', constant_values=np.nan)
                    else:
                        vec_p = vec
                    # reshape 后 nanmean
                    with np.errstate(all='ignore'):
                        gm = np.nanmean(vec_p.reshape(-1, 24), axis=1)
                    # 对完全为 nan 的组，用 np.nanmean 会返回 nan，这里用最小合理值替代（比如35 或 eps）
                    gm = np.where(np.isfinite(gm), gm, np.nan)
                    return gm

                # 6. 按天处理
                for day in unique_days:
                    day_mask = (day_ids == day)
                    day_str = f"Day{int(day)}"
                    table_by_day_type[day_str] = {}

                    # 按流体类型处理
                    for name_c, j in idx_map:
                        t_raw = true_cum_mat[day_mask, j]   # shape (n_samples_in_day,)
                        p_raw = pred_cum_mat[day_mask, j]

                        if t_raw.size == 0:
                            continue

                        # 合并为每24个一组（sum）
                        t24 = aggregate_24_sum(t_raw)
                        p24 = aggregate_24_sum(p_raw)

                        # 对 weight/tsba 做组内平均（如果存在）
                        w24 = None
                        s24 = None
                        if weight_vec is not None:
                            w_day = weight_vec[day_mask]
                            if w_day.size > 0:
                                w24 = aggregate_24_nanmean(w_day)
                        if tsba_vec is not None:
                            s_day = tsba_vec[day_mask]
                            if s_day.size > 0:
                                s24 = aggregate_24_nanmean(s_day)

                        # 有效组掩码（基于合并后的 t24/p24）
                        mask_pos = np.isfinite(t24) & np.isfinite(p24) & (t24 > 0)
                        if not mask_pos.any():
                            continue

                        t_data = t24[mask_pos]
                        p_data = p24[mask_pos]

                        # ml/kg 单位计算（用 w24 的组平均）
                        if w24 is not None:
                            w_data = w24[mask_pos]
                            # 如果某些组因为全是 nan 导致 w_data 为 nan，使用一个合理下限替代以避免除零/无穷
                            w_data = np.where(np.isfinite(w_data), w_data, 35.0)
                            w_data = np.maximum(w_data, 1e-6)
                            t_perkg = t_data / w_data
                            p_perkg = p_data / w_data
                            ae_mlkg = np.abs(p_perkg - t_perkg)
                            re_mlkg = ae_mlkg / (t_perkg + eps)

                            table_by_day_type[day_str][name_c] = {
                                "AE_ml_kg_mean": float(ae_mlkg.mean()),
                                "AE_ml_kg_std": float(ae_mlkg.std()),
                                "RE_ml_kg_mean": float(re_mlkg.mean()),
                                "RE_ml_kg_std": float(re_mlkg.std()),
                            }

                        # ml/(kg*TSBA) 单位计算（用 w24 与 s24）
                        if (w24 is not None) and (s24 is not None):
                            w_data = w24[mask_pos]
                            s_data = s24[mask_pos]
                            # 处理 nan
                            w_data = np.where(np.isfinite(w_data), w_data, 35.0)
                            s_data = np.where(np.isfinite(s_data), s_data, eps)
                            denom = w_data * s_data
                            denom = np.maximum(denom, eps)
                            t_perkg_tsba = t_data / denom
                            p_perkg_tsba = p_data / denom
                            ae_mlkg_tsba = np.abs(p_perkg_tsba - t_perkg_tsba)
                            re_mlkg_tsba = ae_mlkg_tsba / (t_perkg_tsba + eps)

                            if name_c in table_by_day_type[day_str]:
                                table_by_day_type[day_str][name_c].update({
                                    "AE_ml_kg_tbsa_mean": float(ae_mlkg_tsba.mean()),
                                    "AE_ml_kg_tbsa_std": float(ae_mlkg_tsba.std()),
                                    "RE_ml_kg_tbsa_mean": float(re_mlkg_tsba.mean()),
                                    "RE_ml_kg_tbsa_std": float(re_mlkg_tsba.std()),
                                })
                            else:
                                table_by_day_type[day_str][name_c] = {
                                    "AE_ml_kg_tbsa_mean": float(ae_mlkg_tsba.mean()),
                                    "AE_ml_kg_tbsa_std": float(ae_mlkg_tsba.std()),
                                    "RE_ml_kg_tbsa_mean": float(re_mlkg_tsba.mean()),
                                    "RE_ml_kg_tbsa_std": float(re_mlkg_tsba.std()),
                                }

                # 7. 计算总计行（对全部样本也按 24 分组，保持一致）
                table_by_day_type["Total"] = {}

                for name_c, j in idx_map:
                    t_all = true_cum_mat[:, j]
                    p_all = pred_cum_mat[:, j]

                    if t_all.size == 0:
                        continue

                    t24_all = aggregate_24_sum(t_all)
                    p24_all = aggregate_24_sum(p_all)

                    w24_all = None
                    s24_all = None
                    if weight_vec is not None:
                        w24_all = aggregate_24_nanmean(weight_vec)
                    if tsba_vec is not None:
                        s24_all = aggregate_24_nanmean(tsba_vec)

                    mask_pos = np.isfinite(t24_all) & np.isfinite(p24_all) & (t24_all > 0)
                    if not mask_pos.any():
                        continue

                    t_data = t24_all[mask_pos]
                    p_data = p24_all[mask_pos]

                    if w24_all is not None:
                        w_data = w24_all[mask_pos]
                        w_data = np.where(np.isfinite(w_data), w_data, 35.0)
                        w_data = np.maximum(w_data, 1e-6)
                        t_perkg = t_data / w_data
                        p_perkg = p_data / w_data
                        ae_mlkg = np.abs(p_perkg - t_perkg)
                        re_mlkg = ae_mlkg / (t_perkg + eps)

                        table_by_day_type["Total"][name_c] = {
                            "AE_ml_kg_mean": float(ae_mlkg.mean()),
                            "AE_ml_kg_std": float(ae_mlkg.std()),
                            "RE_ml_kg_mean": float(re_mlkg.mean()),
                            "RE_ml_kg_std": float(re_mlkg.std()),
                        }

                    if (w24_all is not None) and (s24_all is not None):
                        w_data = w24_all[mask_pos]
                        s_data = s24_all[mask_pos]
                        w_data = np.where(np.isfinite(w_data), w_data, 35.0)
                        s_data = np.where(np.isfinite(s_data), s_data, eps)
                        denom = w_data * s_data
                        denom = np.maximum(denom, eps)
                        t_perkg_tsba = t_data / denom
                        p_perkg_tsba = p_data / denom
                        ae_mlkg_tsba = np.abs(p_perkg_tsba - t_perkg_tsba)
                        re_mlkg_tsba = ae_mlkg_tsba / (t_perkg_tsba + eps)

                        if name_c in table_by_day_type["Total"]:
                            table_by_day_type["Total"][name_c].update({
                                "AE_ml_kg_tbsa_mean": float(ae_mlkg_tsba.mean()),
                                "AE_ml_kg_tbsa_std": float(ae_mlkg_tsba.std()),
                                "RE_ml_kg_tbsa_mean": float(re_mlkg_tsba.mean()),
                                "RE_ml_kg_tbsa_std": float(re_mlkg_tsba.std()),
                            })
                        else:
                            table_by_day_type["Total"][name_c] = {
                                "AE_ml_kg_tbsa_mean": float(ae_mlkg_tsba.mean()),
                                "AE_ml_kg_tbsa_std": float(ae_mlkg_tsba.std()),
                                "RE_ml_kg_tbsa_mean": float(re_mlkg_tsba.mean()),
                                "RE_ml_kg_tbsa_std": float(re_mlkg_tsba.std()),
                            }

                # 8. 将结果存储到字典中
                ae_re_tables["by_day_type"] = table_by_day_type

        except Exception as e:
            logging.warning(f"Failed to build day-type AE/RE table: {e}")
            # 万一出错，不影响主流程
            pass
        # ========= AE/RE 表构建结束 =========


        # per-horizon（可选）：尝试恢复为 (N,P,C) 后按 t 统计 MAE/RMSE
        try:
            N_total = y_true_t.shape[0]
            y_pred_reg_3d = y_pred_reg.reshape(N_total, P, C)
            y_true_reg_3d = y_true_reg.reshape(N_total, P, C)
            for t in range(P):
                yt = y_true_reg_3d[:, t, :]
                yp = y_pred_reg_3d[:, t, :]
                m  = (yt > zero_target_eps) & np.isfinite(yt) & np.isfinite(yp)
                if m.any():
                    diff = (yp[m] - yt[m])
                    mae_per_h[t]  = float(np.mean(np.abs(diff)))
                    rmse_per_h[t] = float(np.sqrt(np.mean(diff**2)))
        except Exception:
            pass

    # 汇总结构
    per_class_stats = {
        "classification": per_class,
        "overall": {
            "model": str(getattr(args, "model", "")),
            "loss_total": float(avg_total),
            "bit_acc": float(bit_acc),
            "avg_cls_loss": float(avg_raw["cls"]),
            "avg_reg_loss": float(avg_raw["reg"] + avg_raw["cum"]),
            # 分类宏指标
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro),
            "auc_macro": float(auc_macro),
            # 回归整体（y_true>0）
            "mae_all_pos": float(mae_all),
            "rmse_all_pos": float(rmse_all),
            "mse_all_pos": float(mse_all),
            "pearson_r_pos": float(r_all),
            "r2_pos": float(r2_all),
            # 阈值/平滑信息
            "best_thresholds": list(map(float, best_thr)),
            "smooth_win": int(smooth_win),
            "raw_loss": {k: float(v) for k, v in avg_raw.items()},
            # Time Indicator
            "time_indicator": {
                "mae_per_h": [float(x) for x in mae_per_h],
                "rmse_per_h": [float(x) for x in rmse_per_h],
            },
            # nDTW（替代 DTW）+ AE/RE
            "ndtw_mean": float(ndtw_mean),
            "ndtw_p50": float(ndtw_p50),
            "ndtw_p90": float(ndtw_p90),
            "dtw_mean_raw": float(dtw_avg),          # 原始 DTW (参考)
            "cum_true_mean": float(cum_true_mean),   # 归一化分母的均值（解释 nDTW 时有用）
            # 最终累计量误差（总量维度）
            "ae_final_mean": float(ae_mean),
            "ae_final_p50": float(ae_p50),
            "ae_final_p90": float(ae_p90),
            "re_final_mean": float(re_mean),
            "re_final_p50": float(re_p50),
            "re_final_p90": float(re_p90),
            "note": "MAE/RMSE/R/R2 仅在 y_true>0 的点上统计；nDTW 为逐样本 DTW/(sum_true+eps) 的均值；"
                    "AE/RE 为最终累计量误差（绝对值相对误差）",
            # 三种单位下的 AE/RE 表
            "ae_re_tables": ae_re_tables,
        }
    }

    # ---------------- logging ----------------
    if is_main_process():
        prefix = 'train' if is_train else 'val/test'
        logging.info("=" * 72)
        logging.info(f"[{prefix.upper()}] Epoch {epoch}")
        logging.info(f" TotalLoss: {per_class_stats['overall']['loss_total']:.6f} | "
                     f"BitAcc@0.5: {per_class_stats['overall']['bit_acc']:.4f}")
        logging.info(" RawLoss  : "
                     f"cls={per_class_stats['overall']['raw_loss']['cls']:.6f}, "
                     f"reg={per_class_stats['overall']['raw_loss']['reg']:.6f}, "
                     f"cum={per_class_stats['overall']['raw_loss']['cum']:.6f}")

        # 打印实际使用的 lambda（EMA/DWA 不确定性）
        lam_print = balancer.step_weights({"cls": per_class_stats['overall']['raw_loss']['cls'],
                                           "reg": per_class_stats['overall']['raw_loss']['reg'],
                                           "cum": per_class_stats['overall']['raw_loss']['cum']})
        logging.info(" Lambda   (actual weights): cls={:.4f}, reg={:.4f}, cum={:.4f}".format(
            lam_print.get("cls", 1.0), lam_print.get("reg", 1.0), lam_print.get("cum", 0.01)
        ))

        if collect_eval:
            logging.info("  Thresh  | per-class optimal thresholds: "
                         f"{[f'{t:.2f}' for t in per_class_stats['overall']['best_thresholds']]}")

            for name_c, _ in idx_map:
                cls = per_class[name_c]
                logging.info(
                    f"  {name_c:>7s} | cls_loss(BCE-proxy):{cls.get('cls_loss',0.0):.6f} | "
                    f"acc@0.5:{cls.get('cls_acc',0.0):.4f} | "
                    f"P:{cls.get('precision',0.0):.4f} R:{cls.get('recall',0.0):.4f} "
                    f"F1:{cls.get('f1',0.0):.4f} | AUC(raw):{cls.get('auc',0.0):.4f}"
                )

            ov = per_class_stats["overall"]
            # —— 表格行（便于粘到你的"Model/Classification/Regression/TimeIndicator"表）
            logging.info("SummaryRow\tModel\tAcc\tPre\tRecall\tF1\tMAE\tR2\tRMSE\tnDTW")
            logging.info("SummaryRow\t{model}\t{acc:.4f}\t{pre:.4f}\t{rec:.4f}\t{f1:.4f}\t"
                         "{mae:.4f}\t{r2:.4f}\t{rmse:.4f}\t{ndtw:.4f}".format(
                model=ov["model"],
                acc=ov["bit_acc"],
                pre=ov["precision_macro"],
                rec=ov["recall_macro"],
                f1=ov["f1_macro"],
                mae=ov["mae_all_pos"],
                r2=ov["r2_pos"],
                rmse=ov["rmse_all_pos"],
                ndtw=ov["ndtw_mean"],
            ))

            # 人类可读综述（含 r 与分位数 + AE/RE）
            logging.info(
                "Overall | F1(macro):{f1:.4f} | Acc(bit):{acc:.4f} | Pre(macro):{pre:.4f} | Rec(macro):{rec:.4f} | "
                "MAE(>0):{mae:.4f} | RMSE(>0):{rmse:.4f} | R(>0):{r:.4f} | R2(>0):{r2:.4f} | "
                "nDTW(mean):{ndtw:.4f} | nDTW(P50/P90):{p50:.4f}/{p90:.4f} | "
                "AE_final(mean):{ae_m:.2f} P50/P90:{ae_p50:.2f}/{ae_p90:.2f} | "
                "RE_final(mean):{re_m:.3f} P50/P90:{re_p50:.3f}/{re_p90:.3f} | "
                "denom_mean(sum_true):{den:.1f}".format(
                    f1=ov['f1_macro'], acc=ov['bit_acc'],
                    pre=ov['precision_macro'], rec=ov['recall_macro'],
                    mae=ov['mae_all_pos'], rmse=ov['rmse_all_pos'],
                    r=ov['pearson_r_pos'], r2=ov['r2_pos'],
                    ndtw=ov['ndtw_mean'], p50=ov['ndtw_p50'], p90=ov['ndtw_p90'],
                    ae_m=ov['ae_final_mean'], ae_p50=ov['ae_final_p50'], ae_p90=ov['ae_final_p90'],
                    re_m=ov['re_final_mean'], re_p50=ov['re_final_p50'], re_p90=ov['re_final_p90'],
                    den=ov['cum_true_mean']
                )
            )
            logging.info(f"TimeIndicator | MAE_per_h: {ov['time_indicator']['mae_per_h']}")
            logging.info(f"TimeIndicator | RMSE_per_h: {ov['time_indicator']['rmse_per_h']}")

            # ===== 打印三个 AE/RE 表 =====
            tables = ov.get("ae_re_tables", {})

            def _log_aere_table(table, unit_label):
                if table is None:
                    return
                logging.info(f"AE/RE Table ({unit_label}) | type | AE_mean | AE_std | RE_mean | RE_std")
                for key in ["crystal", "colloid", "water", "total"]:
                    if key in table:
                        row = table[key]
                        logging.info(
                            f"  {key:>8s} | "
                            f"{row['AE_mean']:.2f} | {row['AE_std']:.2f} | "
                            f"{row['RE_mean']:.2f} | {row['RE_std']:.2f}"
                        )

            _log_aere_table(tables.get("ml"), "ml")
            _log_aere_table(tables.get("ml_per_kg"), "ml·kg^-1")
            _log_aere_table(tables.get("ml_per_kg_tsba"), "ml·kg^-1·TSBA^-1")

            # ===== 打印按天和类型组织的 AE/RE 表 =====
            day_type_table = tables.get("by_day_type")
            if day_type_table:
                logging.info("AE/RE Table (by Day and Type)")
                logging.info("day\tmetric\tAE_ml_kg_mean\tAE_ml_kg_std\tRE_ml_kg_mean\tRE_ml_kg_std\tAE_ml_kg_tbsa_mean\tAE_ml_kg_tbsa_std\tRE_ml_kg_tbsa_mean\tRE_ml_kg_tbsa_std")
                
                # 定义输出顺序
                days_order = []
                for day in day_type_table.keys():
                    if day.startswith("Day"):
                        days_order.append(day)
                days_order.sort()  # 按Day1, Day2...排序
                days_order.append("Total")  # 最后是Total
                
                fluid_types = ["crystal", "colloid", "water"]  # 根据你的idx_map调整
                
                for day in days_order:
                    if day in day_type_table:
                        for fluid_type in fluid_types:
                            if fluid_type in day_type_table[day]:
                                data = day_type_table[day][fluid_type]
                                # 安全获取所有字段，避免KeyError
                                ae_ml_kg_mean = data.get("AE_ml_kg_mean", 0.0)
                                ae_ml_kg_std = data.get("AE_ml_kg_std", 0.0)
                                re_ml_kg_mean = data.get("RE_ml_kg_mean", 0.0)
                                re_ml_kg_std = data.get("RE_ml_kg_std", 0.0)
                                ae_ml_kg_tbsa_mean = data.get("AE_ml_kg_tbsa_mean", 0.0)
                                ae_ml_kg_tbsa_std = data.get("AE_ml_kg_tbsa_std", 0.0)
                                re_ml_kg_tbsa_mean = data.get("RE_ml_kg_tbsa_mean", 0.0)
                                re_ml_kg_tbsa_std = data.get("RE_ml_kg_tbsa_std", 0.0)
                                
                                logging.info(
                                    f"{day}\t{fluid_type}\t"
                                    f"{ae_ml_kg_mean:.2f}\t{ae_ml_kg_std:.2f}\t"
                                    f"{re_ml_kg_mean:.3f}\t{re_ml_kg_std:.3f}\t"
                                    f"{ae_ml_kg_tbsa_mean:.2f}\t{ae_ml_kg_tbsa_std:.2f}\t"
                                    f"{re_ml_kg_tbsa_mean:.3f}\t{re_ml_kg_tbsa_std:.3f}"
                                )

        logging.info("=" * 72)

    # 返回（接口保持不变；最后一个 dict 内含 nDTW/r/r2/AE/RE 等富指标）
    ov = per_class_stats["overall"]
    return (
        float(ov['loss_total']),        # 0 total loss
        float(ov['bit_acc']),           # 1
        float(ov['avg_reg_loss']),      # 2
        float(ov['f1_macro']),          # 3
        float(ov['precision_macro']),   # 4
        float(ov['recall_macro']),      # 5
        float(ov['auc_macro']),         # 6
        float(ov['mae_all_pos']),       # 7
        float(ov['mse_all_pos']),       # 8
        [], [], [], [], 0.5,            # legacy 保持形状
        per_class_stats,                # -1: 富指标
    )