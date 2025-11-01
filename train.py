# train.py
# -*- coding: utf-8 -*-
import os
import json
import time
import glob
import logging
import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from run_epoch import run_epoch


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
            "best_score": best_score,           # 兼容 main_metric
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

    # 新增：双指标追踪（F1 大优、MAE 小优）
    best_f1 = -float("inf")
    best_mae = float("inf")
    improve_eps = float(getattr(args, "improve_eps", 1e-8))

    patience = int(getattr(args, "patience", 10))
    no_improve = 0
    epochs = int(getattr(args, "epochs", 100))

    # 历史记录
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
            best_score = float(main_score)  # 兼容 main_metric 记录
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

            # 同步写 best 路径（随时可被 test() 读取）
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

    # ---- 训练结束的兜底与“最终保存最好的” ----
    os.makedirs(args.output_path, exist_ok=True)

    # 若整个过程从未产生 best，则至少保存最后一次权重
    if best_ckpt_path is None:
        stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        best_ckpt_path = os.path.join(args.output_path, f"{args.model}_last_{stamp}.pth")
        torch.save(model.state_dict(), best_ckpt_path)
        logging.info(f"[WARN] No improvement observed; saved last weights -> {best_ckpt_path}")

    # 再次将“当前已知最优”的检查点写到一个稳定文件名，便于下游直接引用
    final_best_path = os.path.join(args.output_path, f"{args.model}_best.pth")
    try:
        # 读取 best_ckpt（无论是完整字典还是纯 state_dict），再以统一格式写到 *_best.pth
        state = torch.load(best_ckpt_path, map_location="cpu")
        if isinstance(state, dict) and 'model_state_dict' in state:
            # 直接再保存一次为稳定文件名
            torch.save(state, final_best_path)
        else:
            # 仅有权重，则封装为标准字典
            torch.save({
                'epoch': None,
                'model_state_dict': state if isinstance(state, dict) else state,
                'optimizer_state_dict': None,
                'scheduler_state_dict': None,
                'best_score': best_score,
                'best_f1': best_f1,
                'best_mae': best_mae,
                'args': vars(args)
            }, final_best_path)
        best_ckpt_path = final_best_path
        logging.info(f"[FINAL] Best checkpoint exported -> {final_best_path}")
    except Exception as e:
        logging.warning(f"[FINAL] Export best checkpoint failed: {e}")

    # 写 best 路径文本，供 test() 自动发现
    try:
        with open(os.path.join(args.output_path, "best_ckpt_path.txt"), "w", encoding="utf-8") as f:
            f.write(best_ckpt_path)
    except Exception:
        pass

    # 保存训练元数据（包含 best_f1/mae 与路径）
    _save_training_metadata(
        args, best_epoch, best_score, args.output_path,
        best_f1=best_f1, best_mae=best_mae, best_ckpt_path=best_ckpt_path
    )

    # 保存完整训练历史
    try:
        history_path = os.path.join(args.output_path, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(train_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Failed to save final training history: {e}")

    logging.info("Training complete.")
    return best_ckpt_path, best_score, best_epoch


def _load_ckpt_forgiving(model: nn.Module, ckpt_path: str, device):
    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict):
        if 'model_state_dict' in state:
            model_state = state['model_state_dict']
            logging.info(f"Loaded checkpoint from epoch {state.get('epoch', 'unknown')}, "
                         f"best_score: {state.get('best_score', 'unknown')}")
        else:
            model_state = state
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
            new_k = k.replace("module.", "") if k.startswith("module.") else k
            new_state[new_k] = v
        try:
            model.load_state_dict(new_state, strict=False)
            logging.info("Checkpoint loaded successfully after module prefix removal")
        except Exception as e2:
            logging.error(f"All loading attempts failed: {e2}")
            raise


def test(model: nn.Module, test_loader, args, ckpt_path: Optional[str] = None, thresholds_path: Optional[str] = None):
    device = args.device
    model = model.to(device)
    model.eval()

    # 选择 checkpoint
    if ckpt_path is None:
        txtp = os.path.join(args.output_path, "best_ckpt_path.txt")
        if os.path.exists(txtp):
            with open(txtp, "r", encoding="utf-8") as f:
                ckpt_path = f.read().strip()

    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        # 尝试 *_best.pth 或最新 .pth
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
        logging.warning(f"[TEST] thresholds file not found: {thresholds_path}. "
                        f"Fallback to args.fixed_thresholds={getattr(args,'fixed_thresholds', None)}")

    # 评估
    logging.info("[TEST] Starting evaluation...")
    test_pack = run_epoch(
        model, test_loader, optimizer=None,
        criterion_classification=None, criterion_regression=None,
        device=device, args=args, is_train=False, epoch=-1
    )
    stats = test_pack[-1] if isinstance(test_pack[-1], dict) else {}

    (test_loss, test_acc, test_reg_loss, test_f1, test_prec, test_rec,
     test_auc, test_mae, test_mse, *_tail) = test_pack
    test_rmse = math.sqrt(test_mse) if test_mse > 0 else 0.0

    logging.info("=== Test Results ===")
    logging.info(f"Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
    logging.info(f"Regression - MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}")

    # 持久化测试指标
    try:
        out_path = os.path.join(args.output_path, "test_metrics.json")
        payload = {
            "overall": stats.get("overall", {}),
            "classification": stats.get("classification", {}),
            "fixed_thresholds": getattr(args, "fixed_thresholds", None),
            "checkpoint": ckpt_path,
            "test_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "detailed_metrics": {
                "loss": float(test_loss),
                "accuracy": float(test_acc),
                "f1_score": float(test_f1),
                "precision": float(test_prec),
                "recall": float(test_rec),
                "auc": float(test_auc),
                "mae": float(test_mae),
                "mse": float(test_mse),
                "rmse": float(test_rmse)
            }
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logging.info(f"[TEST] Saved metrics -> {out_path}")
    except Exception as e:
        logging.warning(f"[TEST] Failed to save test metrics: {e}")

    return test_pack
