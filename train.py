# -*- coding: utf-8 -*-
import os
import json
import time
import glob
import logging
import math
import shutil
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from run_epoch import run_epoch


# ------------------------------ 统计参数工具 ------------------------------ #
def _count_params(model: nn.Module):
    """返回(总参数, 可训练参数)，单位：个"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _format_params(n: int) -> str:
    """将参数量格式化为 K/M/B 显示"""
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return str(n)


def _make_optimizer(model: nn.Module, args) -> optim.Optimizer:
    base_lr = float(getattr(args, "learning_rate", 1e-5))
    head_mul = float(getattr(args, "head_lr_mult", 1.0))
    weight_decay = float(getattr(args, "weight_decay", 0.0))

    params = []
    trunk_ids = set()

    # 若模型显式暴露 cls_head / reg_head，则 head 用更高 LR
    if hasattr(model, "cls_head") or hasattr(model, "reg_head"):
        if hasattr(model, "cls_head"):
            params.append({
                "params": model.cls_head.parameters(),
                "lr": base_lr * head_mul,
                "weight_decay": weight_decay
            })
        if hasattr(model, "reg_head"):
            params.append({
                "params": model.reg_head.parameters(),
                "lr": base_lr * head_mul,
                "weight_decay": weight_decay
            })
        for n, p in model.named_parameters():
            is_head = n.startswith("cls_head") or n.startswith("reg_head")
            if (not is_head) and p.requires_grad:
                trunk_ids.add(id(p))
        if trunk_ids:
            trunk = [p for p in model.parameters() if id(p) in trunk_ids]
            params.append({
                "params": trunk,
                "lr": base_lr,
                "weight_decay": weight_decay
            })
        logging.info(f"Optimizer: {len(params)} param groups, head_lr_mult={head_mul}")
    else:
        params = [{
            "params": model.parameters(),
            "lr": base_lr,
            "weight_decay": weight_decay
        }]
        logging.info("Optimizer: single param group")

    optimizer_type = str(getattr(args, "optimizer", "adam")).lower()
    if optimizer_type == "adamw":
        optimizer = optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(params, lr=base_lr, weight_decay=weight_decay)

    return optimizer


def _make_scheduler(optimizer: optim.Optimizer, args):
    kind = str(getattr(args, "lr_scheduler", "step")).lower()

    if kind == "cosine":
        t0 = int(getattr(args, "t0", 50))
        eta_min_ratio = float(getattr(args, "eta_min_ratio", 0.01))
        base_lrs = [g["lr"] for g in optimizer.param_groups]
        eta_min = min(base_lrs) * eta_min_ratio if base_lrs else 0.0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t0, T_mult=1, eta_min=eta_min
        )
        logging.info(f"Scheduler: CosineAnnealingWarmRestarts (T_0={t0})")

    elif kind == "plateau":
        patience = int(getattr(args, "lr_patience", 5))
        factor = float(getattr(args, "lr_factor", 0.5))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=patience, factor=factor, verbose=True
        )
        logging.info(f"Scheduler: ReduceLROnPlateau (patience={patience}, factor={factor})")

    else:  # step
        step_size = int(getattr(args, "lr_step", 100))
        gamma = float(getattr(args, "lr_gamma", 0.1))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
        logging.info(f"Scheduler: StepLR (step={step_size}, gamma={gamma})")

    return scheduler


def _calc_main_score(args, val_pack: Tuple) -> float:
    """
    run_epoch 返回：
      (loss_total, bit_acc, avg_reg_loss, f1_macro, precision_macro, recall_macro,
       auc_macro, mae_pos, mse_pos, ... , stats_dict)
    """
    (val_loss, val_acc, _, val_f1, val_prec, val_rec, val_auc, val_mae, val_mse, *_tail) = val_pack

    metric = str(getattr(args, "main_metric", "f1")).lower()

    if metric == "loss":
        return -float(val_loss)
    elif metric == "auc":
        return float(val_auc)
    elif metric == "acc":
        return float(val_acc)
    elif metric == "fbeta":
        beta = float(getattr(args, "fbeta", 1.0))
        denom = (beta ** 2) * val_prec + val_rec
        if denom <= 0:
            return 0.0
        return (1 + beta ** 2) * (val_prec * val_rec) / denom
    elif metric == "mae":
        return -float(val_mae)
    elif metric == "mse":
        return -float(val_mse)
    elif metric == "rmse":
        return -math.sqrt(float(val_mse)) if val_mse > 0 else 0.0
    else:
        return float(val_f1)


def _analyze_regression_performance(val_pack: Tuple, args):
    (val_loss, val_acc, val_reg_loss, val_f1, val_prec, val_rec, val_auc, val_mae, val_mse, *_tail) = val_pack
    val_rmse = math.sqrt(val_mse) if val_mse > 0 else 0.0

    logging.info("=== Regression Performance Analysis ===")
    logging.info(f"MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}")
    logging.info(f"Regression Loss: {val_reg_loss:.4f}")

    if val_mse > 10000:
        logging.warning("Very high MSE detected.")
        logging.info("Focus on classification metrics due to reduced regression weights.")
    elif val_mse > 1000:
        logging.info("High MSE detected; regression is secondary to classification.")


def _auto_find_latest_ckpt(output_path: str) -> Optional[str]:
    cands = sorted(glob.glob(os.path.join(output_path, "*.pth")))
    return cands[-1] if cands else None


def _save_best_thresholds(stats: Dict[str, Any], out_dir: str):
    try:
        best_thr = stats.get("overall", {}).get("best_thresholds", None)
        if best_thr is not None:
            p = os.path.join(out_dir, "best_thresholds.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(list(map(float, best_thr)), f, ensure_ascii=False, indent=2)
            logging.info(f"[VAL] Saved best thresholds -> {p} | {best_thr}")
    except Exception as e:
        logging.warning(f"[VAL] Save thresholds failed: {e}")


def _save_training_metadata(args, best_epoch: int, best_score: float, output_path: str,
                            best_f1: float = None, best_mae: float = None, best_ckpt_path: str = None):
    try:
        metadata = {
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_f1": best_f1,
            "best_mae": best_mae,
            "best_ckpt_path": best_ckpt_path,
            "main_metric": getattr(args, "main_metric", "f1"),
            "model": getattr(args, "model", ""),
            "training_completed": time.strftime('%Y-%m-%d %H:%M:%S'),
            "args": {k: str(v) for k, v in vars(args).items()}
        }
        metadata_path = os.path.join(output_path, "training_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Failed to save training metadata: {e}")


def train(model: nn.Module, train_loader, val_loader, args):
    device = args.device
    model = model.to(device)

    # 损失权重
    args.lambda_cls = float(getattr(args, "lambda_cls", 1.0))
    args.lambda_reg = float(getattr(args, "lambda_reg", 0.3))
    args.lambda_cum = float(getattr(args, "lambda_cum", 0.01))

    optimizer = _make_optimizer(model, args)
    scheduler = _make_scheduler(optimizer, args)

    best_score = float("-inf")
    best_epoch = -1
    best_ckpt_path = None

    # 双指标追踪（F1 大优、MAE 小优）
    best_f1 = -float("inf")
    best_mae = float("inf")
    improve_eps = float(getattr(args, "improve_eps", 1e-8))

    patience = int(getattr(args, "patience", 10))
    no_improve = 0
    epochs = int(getattr(args, "epochs", 100))

    # 历史
    train_history = {
        'epoch': [],
        'train_loss': [],
        'val_score': [],
        'val_f1': [],
        'val_mae': [],
        'val_mse': [],
        'learning_rate': []
    }

    logging.info(f"Start training for up to {epochs} epochs. main_metric={args.main_metric}")
    logging.info(f"Loss weights - cls: {args.lambda_cls}, reg: {args.lambda_reg}, cum: {args.lambda_cum}")
    if hasattr(args, 'fixed_thresholds'):
        logging.info(f"Fixed thresholds: {args.fixed_thresholds}")

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        logging.info(f"Epoch {epoch+1}/{epochs} started at {time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}")

        # ---- Train ----
        train_pack = run_epoch(
            model, train_loader, optimizer=optimizer,
            criterion_classification=None, criterion_regression=None,
            device=device, args=args, is_train=True, epoch=epoch
        )
        train_loss = float(train_pack[0])

        # ---- Val ----
        val_pack = run_epoch(
            model, val_loader, optimizer=None,
            criterion_classification=None, criterion_regression=None,
            device=device, args=args, is_train=False, epoch=epoch
        )
        val_stats = val_pack[-1] if isinstance(val_pack[-1], dict) else {}
        main_score = _calc_main_score(args, val_pack)

        (val_loss, val_acc, val_reg_loss, val_f1, val_prec, val_rec, val_auc, val_mae, val_mse, *_tail) = val_pack

        # ---- LR Step ----
        current_lr = optimizer.param_groups[0]['lr']
        try:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(main_score)
            else:
                scheduler.step()
        except Exception as e:
            logging.warning(f"Scheduler step failed: {e}")

        # ---- Log history ----
        train_history['epoch'].append(epoch + 1)
        train_history['train_loss'].append(train_loss)
        train_history['val_score'].append(main_score)
        train_history['val_f1'].append(val_f1)
        train_history['val_mae'].append(val_mae)
        train_history['val_mse'].append(val_mse)
        train_history['learning_rate'].append(current_lr)

        epoch_time = time.time() - epoch_start
        logging.info(
            f"[TRAIN] loss={train_loss:.6f}, "
            f"[VAL] {args.main_metric}={main_score:.4f}, "
            f"F1={val_f1:.4f}, Acc={val_acc:.4f}, "
            f"MAE={val_mae:.4f}, lr={current_lr:.2e}, time={epoch_time:.1f}s"
        )

        if epoch % 10 == 0:
            _analyze_regression_performance(val_pack, args)

        # ---- Save best: 任一改进（F1↑ 或 MAE↓）即保存 ----
        if not np.isfinite(main_score):
            logging.warning("[VAL] main_score is NaN, skip improvement check.")
            improved = False
        else:
            f1_better = (val_f1 > best_f1 + improve_eps)
            mae_better = (val_mae + improve_eps < best_mae)
            improved = f1_better or mae_better

        if improved:
            if f1_better:
                best_f1 = float(val_f1)
            if mae_better:
                best_mae = float(val_mae)

            best_epoch = epoch
            best_score = float(main_score)
            os.makedirs(args.output_path, exist_ok=True)

            ckpt_path = os.path.join(
                args.output_path,
                f"{args.model}_best_f1_{best_f1:.4f}_mae_{best_mae:.4f}.pth"
            )

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_score': best_score,
                'best_f1': best_f1,
                'best_mae': best_mae,
                'args': vars(args)
            }, ckpt_path)

            best_ckpt_path = ckpt_path
            _save_best_thresholds(val_stats, args.output_path)

            # 同步写 best 路径
            try:
                with open(os.path.join(args.output_path, "best_ckpt_path.txt"), "w", encoding="utf-8") as f:
                    f.write(best_ckpt_path)
            except Exception:
                pass

            logging.info(
                f"[VAL] New best by {'F1' if f1_better else ''}"
                f"{' & ' if f1_better and mae_better else ''}"
                f"{'MAE' if mae_better else ''}: "
                f"F1={val_f1:.4f} (best {best_f1:.4f}), MAE={val_mae:.4f} (best {best_mae:.4f}); "
                f"{args.main_metric}={best_score:.4f}. Saved -> {ckpt_path}"
            )
            no_improve = 0
        else:
            no_improve += 1
            logging.info(
                f"[VAL] No improvement for {no_improve} epoch(s). "
                f"best_f1={best_f1:.4f}, best_mae={best_mae:.4f}, "
                f"best_{args.main_metric}={best_score:.4f} @e{best_epoch+1}"
            )

        # ---- Save history periodically ----
        if epoch % 10 == 0:
            try:
                history_path = os.path.join(args.output_path, "training_history.json")
                with open(history_path, "w", encoding="utf-8") as f:
                    json.dump(train_history, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logging.warning(f"Failed to save training history: {e}")

        # ---- Early stopping ----
        if no_improve >= patience:
            logging.info(f"Early stopping at epoch {epoch+1} (patience={patience}).")
            break

    total_time = time.time() - start_time
    logging.info(f"Training completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")

    # ---- 训练结束兜底 ----
    os.makedirs(args.output_path, exist_ok=True)

    if best_ckpt_path is None:
        stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        best_ckpt_path = os.path.join(args.output_path, f"{args.model}_last_{stamp}.pth")
        torch.save(model.state_dict(), best_ckpt_path)
        logging.info(f"[WARN] No improvement observed; saved last weights -> {best_ckpt_path}")

    # 统一导出 *_best.pth —— 直接复制，避免反序列化触发 PyTorch 2.6 安全限制
    final_best_path = os.path.join(args.output_path, f"{args.model}_best.pth")
    try:
        if best_ckpt_path and os.path.exists(best_ckpt_path):
            shutil.copyfile(best_ckpt_path, final_best_path)
            best_ckpt_path = final_best_path
            logging.info(f"[FINAL] Best checkpoint exported -> {final_best_path}")
        else:
            logging.warning(f"[FINAL] No best_ckpt_path to export: {best_ckpt_path}")
    except Exception as e:
        logging.warning(f"[FINAL] Export best checkpoint failed: {e}")

    try:
        with open(os.path.join(args.output_path, "best_ckpt_path.txt"), "w", encoding="utf-8") as f:
            f.write(best_ckpt_path)
    except Exception:
        pass

    _save_training_metadata(
        args, best_epoch, best_score, args.output_path,
        best_f1=best_f1, best_mae=best_mae, best_ckpt_path=best_ckpt_path
    )

    try:
        history_path = os.path.join(args.output_path, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(train_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Failed to save final training history: {e}")

    logging.info("Training complete.")
    return best_ckpt_path, best_score, best_epoch


def _load_ckpt_forgiving(model: nn.Module, ckpt_path: str, device):
    """
    加强版加载：
    1) 先尝试 weights_only=True（PyTorch 2.6 默认安全模式）
    2) 尝试注册自定义安全全局（run_epoch.LossBalancer），再试一次 weights_only=True
    3) 最后回退到 weights_only=False（仅在你确认 checkpoint 来源可信时）
    """
    def _try_load(weights_only_flag):
        return torch.load(ckpt_path, map_location=device, weights_only=weights_only_flag)

    state = None

    # 1) 安全模式
    try:
        state = _try_load(weights_only_flag=True)
    except Exception as e1:
        logging.debug(f"Safe load (weights_only=True) failed: {e1}")

    # 2) 注册自定义全局再试
    if state is None:
        try:
            try:
                from run_epoch import LossBalancer  # 仅用于 allowlist
                torch.serialization.add_safe_globals([LossBalancer])
            except Exception:
                pass
            state = _try_load(weights_only_flag=True)
        except Exception as e2:
            logging.debug(f"Safe load with allowlist failed: {e2}")

    # 3) 明确不安全模式
    if state is None:
        logging.warning(
            "Safe load failed; falling back to weights_only=False. "
            "Only do this if you trust the checkpoint source."
        )
        state = _try_load(weights_only_flag=False)

    if isinstance(state, dict) and 'model_state_dict' in state:
        model_state = state['model_state_dict']
        logging.info(f"Loaded checkpoint from epoch {state.get('epoch', 'unknown')}, "
                     f"best_score: {state.get('best_score', 'unknown')}")
    else:
        model_state = state

    try:
        model.load_state_dict(model_state, strict=False)
        logging.info("Checkpoint loaded successfully with strict=False")
    except Exception as e:
        logging.warning(f"Non-strict loading failed: {e}, trying module prefix removal...")
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in model_state.items():
            new_k = k.replace("module.", "") if isinstance(k, str) and k.startswith("module.") else k
            new_state[new_k] = v
        model.load_state_dict(new_state, strict=False)
        logging.info("Checkpoint loaded successfully after module prefix removal")


# ------------------------------ 更稳的 JSON 导出 ------------------------------ #
def _to_py(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_py(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def test(model: nn.Module, test_loader, args, ckpt_path: Optional[str] = None, thresholds_path: Optional[str] = None):
    device = args.device
    model = model.to(device)
    model.eval()

    # ========= 在这里计算模型的参数量等数值 =========
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 简单估算模型大小（假设 float32 -> 4 字节）
    param_size_bytes = total_params * 4
    param_size_mb = param_size_bytes / (1024 ** 2)

    logging.info(
        f"[TEST] Model parameters: total={total_params:,}, "
        f"trainable={trainable_params:,}, "
        f"approx_size={param_size_mb:.2f} MB (fp32)"
    )
    # ========================================

    # 选择 checkpoint
    if ckpt_path is None:
        txtp = os.path.join(args.output_path, "best_ckpt_path.txt")
        if os.path.exists(txtp):
            with open(txtp, "r", encoding="utf-8") as f:
                ckpt_path = f.read().strip()

    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        candidate = os.path.join(args.output_path, f"{args.model}_best.pth")
        ckpt_path = candidate if os.path.exists(candidate) else _auto_find_latest_ckpt(args.output_path)

    if ckpt_path and os.path.exists(ckpt_path):
        _load_ckpt_forgiving(model, ckpt_path, device)
        logging.info(f"[TEST] Loaded checkpoint: {ckpt_path}")
    else:
        logging.warning("[TEST] No checkpoint found; evaluating current weights.")

    # 固定阈值（来自 VAL）
    args.tune_thresholds = False
    if thresholds_path is None:
        thresholds_path = os.path.join(args.output_path, "best_thresholds.json")

    if os.path.exists(thresholds_path):
        try:
            with open(thresholds_path, "r", encoding="utf-8") as f:
                best_thr = json.load(f)
            args.fixed_thresholds = [float(x) for x in best_thr][:3]
            logging.info(f"[TEST] Using fixed thresholds from VAL: {args.fixed_thresholds}")
        except Exception as e:
            logging.warning(f"[TEST] Failed to load thresholds; fallback to args.fixed_thresholds. Err={e}")
    else:
        logging.warning(
            f"[TEST] thresholds file not found: {thresholds_path}. "
            f"Fallback to args.fixed_thresholds={getattr(args, 'fixed_thresholds', None)}"
        )

    # ---------- 评测计时开始 ----------
    torch.cuda.synchronize(device) if torch.cuda.is_available() else None
    eval_start = time.time()

    logging.info("[TEST] Starting evaluation...")
    test_pack = run_epoch(
        model, test_loader, optimizer=None,
        criterion_classification=None, criterion_regression=None,
        device=device, args=args, is_train=False, epoch=-1
    )

    torch.cuda.synchronize(device) if torch.cuda.is_available() else None
    eval_dur = time.time() - eval_start
    # ---------- 评测计时结束 ----------

    # 基本统计
    stats = test_pack[-1] if isinstance(test_pack[-1], dict) else {}

    (test_loss, test_acc, test_reg_loss, test_f1, test_prec, test_rec,
     test_auc, test_mae, test_mse, *_tail) = test_pack
    test_rmse = math.sqrt(test_mse) if test_mse > 0 else 0.0

    overall = stats.get("overall", {})
    r_pos   = float(overall.get("pearson_r_pos", 0.0))
    r2_pos  = float(overall.get("r2_pos", 0.0))
    ndtw_mean = float(overall.get("ndtw_mean", 0.0))
    ndtw_p50  = float(overall.get("ndtw_p50", 0.0))
    ndtw_p90  = float(overall.get("ndtw_p90", 0.0))
    time_ind  = overall.get("time_indicator", {}) or {}
    mae_per_h = time_ind.get("mae_per_h", [])
    rmse_per_h = time_ind.get("rmse_per_h", [])

    # 估计样本与吞吐
    n_batches = None
    bs_hint = None
    n_samples = None
    try:
        n_batches = len(test_loader)
    except Exception:
        pass
    try:
        bs_hint = getattr(test_loader, "batch_size", None)
    except Exception:
        pass
    # 优先从 stats 里拿（如果 run_epoch 里有写入）
    n_samples = overall.get("n_samples") if isinstance(overall, dict) else None
    if n_samples is None:
        # 次优：dataset 长度
        try:
            n_samples = len(test_loader.dataset)
        except Exception:
            n_samples = None
    # 仍拿不到就按 batch 粗估
    if (n_samples is None) and (n_batches is not None) and (bs_hint is not None):
        n_samples = n_batches * bs_hint

    if eval_dur > 0:
        if n_samples is not None:
            throughput = n_samples / eval_dur
            logging.info(
                f"[TEST] Elapsed={eval_dur:.3f}s, Samples={n_samples}, "
                f"Throughput={throughput:.2f} samples/s"
            )
        else:
            throughput = None
            logging.info(f"[TEST] Elapsed={eval_dur:.3f}s (Samples=unknown)")
        if n_batches:
            logging.info(
                f"[TEST] Avg time per batch ≈ {eval_dur / n_batches:.4f}s "
                f"over {n_batches} batches (batch_size≈{bs_hint})"
            )
    else:
        logging.info("[TEST] Elapsed time too small to report throughput.")

    # 结果日志
    logging.info("=== Test Results ===")
    logging.info(f"Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
    logging.info(f"Regression - MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}")
    logging.info(
        f"R(>0): {r_pos:.4f}, R2(>0): {r2_pos:.4f}, "
        f"nDTW(mean): {ndtw_mean:.4f} (P50/P90: {ndtw_p50:.4f}/{ndtw_p90:.4f})"
    )

    # 持久化测试指标
    try:
        out_path = os.path.join(args.output_path, "test_metrics.json")
        payload = {
            "checkpoint": ckpt_path,
            "fixed_thresholds": getattr(args, "fixed_thresholds", None),
            "test_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "timing": {
                "elapsed_sec": float(eval_dur),
                "n_batches": int(n_batches) if n_batches is not None else None,
                "batch_size_hint": int(bs_hint) if bs_hint is not None else None,
                "n_samples": int(n_samples) if n_samples is not None else None,
                "throughput_sps": float(n_samples / eval_dur) if (n_samples is not None and eval_dur > 0) else None,
            },
            "model_params": {
                "total": int(total_params),
                "trainable": int(trainable_params),
                "total_fmt": f"{total_params:,}",
                "trainable_fmt": f"{trainable_params:,}",
                "approx_size_mb": float(param_size_mb),
            },
            "detailed_metrics": {
                "loss": float(test_loss),
                "accuracy": float(test_acc),
                "f1_score": float(test_f1),
                "precision": float(test_prec),
                "recall": float(test_rec),
                "auc": float(test_auc),
                "mae": float(test_mae),
                "mse": float(test_mse),
                "rmse": float(test_rmse),
                "pearson_r_pos": r_pos,
                "r2_pos": r2_pos,
                "ndtw_mean": ndtw_mean,
                "ndtw_p50": ndtw_p50,
                "ndtw_p90": ndtw_p90,
                "mae_per_h": _to_py(mae_per_h),
                "rmse_per_h": _to_py(rmse_per_h),
            },
            "overall": _to_py(overall),
            "classification": _to_py(stats.get("classification", {})),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logging.info(f"[TEST] Saved metrics -> {out_path}")
    except Exception as e:
        logging.warning(f"[TEST] Failed to save test metrics: {e}")

    return test_pack
