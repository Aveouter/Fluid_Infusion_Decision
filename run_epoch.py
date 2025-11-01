# -*- coding: utf-8 -*-
import logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, roc_auc_score,
    mean_absolute_error
)

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

    # AMP
    use_amp = bool(getattr(args, "use_amp", False))
    scaler  = getattr(args, "amp_scaler", None)

    # 损失权重
    lambda_cls = float(getattr(args, "lambda_cls", 1.0))
    lambda_reg = float(getattr(args, "lambda_reg", 0.01))
    lambda_cum = float(getattr(args, "lambda_cum", 0.001))
    base_lambda = {"cls": lambda_cls, "reg": lambda_reg, "cum": lambda_cum}

    # 累计器
    sums_raw = {k: 0.0 for k in base_lambda}
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

    collect_eval = (not is_train)
    desc = f"{'Train' if is_train else 'Val/Test'} Ep{epoch}"
    pbar = tqdm(data_loader, total=len(data_loader), desc=desc, leave=False, disable=not is_main_process())

    if epoch == 0 and is_main_process():
        logging.info(f"[DEBUG] Loss weights - cls: {lambda_cls}, reg: {lambda_reg}, cum: {lambda_cum}")

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

                # 损失
                L_cls = _bce_logits(logits, y_type.float(), reduction='mean')
                L_reg = _reg_l1_pos_only(speed, y_flow, zero_eps=zero_target_eps)
                L_zero = _zero_penalty(speed, y_flow, zero_eps=zero_target_eps)
                L_reg = L_reg + 0.05 * L_zero

                cum_true  = torch.cumsum(inst_true * delta_ml, dim=1).sum(-1)  # (B,P)->sum C
                cum_pred  = torch.cumsum(rate      * delta_ml, dim=1).sum(-1)
                L_cum     = _safe_mean(huber(cum_pred - cum_true, delta=cum_tol_ml), 0.0)

                L = (base_lambda["cls"] * L_cls +
                     base_lambda["reg"] * L_reg +
                     base_lambda["cum"] * L_cum)
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

            pbar.set_postfix({
                "tot": f"{_safe_item(L):.4f}",
                "cls": f"{_safe_item(L_cls):.4f}",
                "acc": f"{(correct_bits/max(1,total_bits)):.3f}",
            })

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

    # 新：nDTW
    dtw_avg = 0.0
    ndtw_mean = 0.0
    ndtw_p50  = 0.0
    ndtw_p90  = 0.0
    cum_true_mean = 0.0

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

        # —— nDTW 统计：逐样本 DTW/最终累计真值 —— #
        if len(all_rate_seq_total) > 0:
            rate_mat = torch.cat(all_rate_seq_total, dim=0).numpy()  # (N, P)
            true_mat = torch.cat(all_true_seq_total, dim=0).numpy()  # (N, P)
            dtw_list, ndtw_list, final_list = [], [], []
            eps = 1e-6
            for i in range(rate_mat.shape[0]):
                # 推荐口径：对累计曲线做 DTW 也可以；这里按瞬时 total 做 DTW（与前版口径一致），换成累计只需加 cumsum
                # 若你想对累计做 DTW：uncomment 两行
                # a = np.cumsum(rate_mat[i])
                # b = np.cumsum(true_mat[i])
                a = rate_mat[i]
                b = true_mat[i]
                dtw_val = _dtw_distance_1d(a, b)
                final   = float(np.sum(b))  # 也可用累计末值：float(np.cumsum(b)[-1])
                dtw_list.append(dtw_val)
                final_list.append(final)
                ndtw_list.append(dtw_val / (final + eps))

            if len(dtw_list) > 0:
                dtw_arr  = np.array(dtw_list, dtype=np.float64)
                ndtw_arr = np.array(ndtw_list, dtype=np.float64)
                fin_arr  = np.array(final_list, dtype=np.float64)
                dtw_avg      = float(dtw_arr.mean())
                ndtw_mean    = float(ndtw_arr.mean())
                ndtw_p50     = float(np.percentile(ndtw_arr, 50))
                ndtw_p90     = float(np.percentile(ndtw_arr, 90))
                cum_true_mean= float(fin_arr.mean())

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
            "lambda_fixed": {k: float(v) for k, v in base_lambda.items()},
            "raw_loss": {k: float(v) for k, v in avg_raw.items()},
            # Time Indicator
            "time_indicator": {
                "mae_per_h": [float(x) for x in mae_per_h],
                "rmse_per_h": [float(x) for x in rmse_per_h],
            },
            # nDTW（替代 DTW）
            "ndtw_mean": float(ndtw_mean),
            "ndtw_p50": float(ndtw_p50),
            "ndtw_p90": float(ndtw_p90),
            "dtw_mean_raw": float(dtw_avg),          # 原始 DTW (仅作参考，不用于表格)
            "cum_true_mean": float(cum_true_mean),   # 归一化分母的均值（解释 nDTW 时有用）
            "note": "MAE/RMSE/R/R2 仅在 y_true>0 的点上统计；nDTW 为逐样本 DTW/(sum_true+eps) 的均值",
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
                     f"reg={avg_raw['reg']:.6f}, "
                     f"cum={avg_raw['cum']:.6f}")
        logging.info(" Lambda   : "
                     f"cls={base_lambda['cls']:.4f}, reg={base_lambda['reg']:.4f}, cum={base_lambda['cum']:.4f}")

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
            # —— 表格行（便于粘到你的“Model/Classification/Regression/TimeIndicator”表）
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

            # 人类可读综述（包含 r 与分位数）
            logging.info(
                "Overall | F1(macro):{f1:.4f} | Acc(bit):{acc:.4f} | Pre(macro):{pre:.4f} | Rec(macro):{rec:.4f} | "
                "MAE(>0):{mae:.4f} | RMSE(>0):{rmse:.4f} | R(>0):{r:.4f} | R2(>0):{r2:.4f} | "
                "nDTW(mean):{ndtw:.4f} | nDTW(P50/P90):{p50:.4f}/{p90:.4f} | denom_mean(sum_true):{den:.1f}".format(
                    f1=ov['f1_macro'], acc=ov['bit_acc'],
                    pre=ov['precision_macro'], rec=ov['recall_macro'],
                    mae=ov['mae_all_pos'], rmse=ov['rmse_all_pos'],
                    r=ov['pearson_r_pos'], r2=ov['r2_pos'],
                    ndtw=ov['ndtw_mean'], p50=ov['ndtw_p50'], p90=ov['ndtw_p90'],
                    den=ov['cum_true_mean']
                )
            )
            logging.info(f"TimeIndicator | MAE_per_h: {ov['time_indicator']['mae_per_h']}")
            logging.info(f"TimeIndicator | RMSE_per_h: {ov['time_indicator']['rmse_per_h']}")
        logging.info("=" * 72)

    # 返回（接口保持不变；最后一个 dict 内含 nDTW）
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
        per_class_stats,                # -1: 富指标（包含 ndtw_mean / r / r2）
    )
