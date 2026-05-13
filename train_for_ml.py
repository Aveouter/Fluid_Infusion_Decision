# -*- coding: utf-8 -*-
"""
Refactored ML baselines for burn-fluid tasks
- Fixes data leakage (scaler/thresholds) and probability handling
- Adds robust metrics (AUPRC, Brier, ECE, operating-point PR, P50/P90 MAE, R2)
- Validates thresholds on val set and freezes them for test
- Persists scaler/thresholds/models per baseline
- 默认在 test 集上画 ROC 曲线（每个模型一张图）

Run example:
python ml_baselines_refactor.py \
  --ts_pkl /baksv/CIGIT/GXN_Liuxy/fluid/data/output_data.pkl \
  --base_pkl /baksv/CIGIT/GXN_Liuxy/fluid/data/baseline.pkl \
  --out ml_output_refined \
  --models RandomForest XGBoost Logistic SVM GBDT
"""

import os
import argparse
import json
import logging
import pickle
import numpy as np
from typing import Tuple, Dict, Any

from joblib import dump, load

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression

# 可选的 XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor  # noqa

    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# 你的工程里的数据加载器
from utils.data_loader import data_loader

np.random.seed(42)


# ----------------------------------------------------------
# Logging
# ----------------------------------------------------------
def setup_logger(output_path: str) -> logging.Logger:
    os.makedirs(output_path, exist_ok=True)
    logger = logging.getLogger("mlref")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(output_path, "train_ml.log"))
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


# ----------------------------------------------------------
# Model factory
# ----------------------------------------------------------
def get_model(name: str):
    name_l = name.lower()
    if name_l == "randomforest":
        clf = OneVsRestClassifier(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        )
        reg = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=500, random_state=42, n_jobs=-1
            )
        )
        scaler = None

    elif name_l == "xgboost":
        if not _HAS_XGB:
            raise RuntimeError("xgboost 未安装，请先安装：pip install xgboost")
        clf = OneVsRestClassifier(
            XGBClassifier(
                eval_metric="logloss",
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                n_estimators=400,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1,
            )
        )
        reg = MultiOutputRegressor(
            XGBRegressor(
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                n_estimators=600,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1,
            )
        )
        scaler = None

    elif name_l == "logistic":
        clf = OneVsRestClassifier(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
                n_jobs=-1,
            )
        )
        reg = MultiOutputRegressor(LinearRegression())
        # 仅对线性/核方法做标准化（避免泄漏：只 fit 在 train）
        scaler = StandardScaler()

    elif name_l == "svm":
        clf = OneVsRestClassifier(
            SVC(probability=True, class_weight="balanced", random_state=42)
        )
        reg = MultiOutputRegressor(SVR())
        scaler = StandardScaler()

    elif name_l in ("gbdt", "gbr", "gradientboosting"):
        clf = OneVsRestClassifier(
            GradientBoostingClassifier(random_state=42)
        )
        reg = MultiOutputRegressor(
            GradientBoostingRegressor(random_state=42)
        )
        scaler = None

    else:
        raise ValueError(f"Unsupported model: {name}")
    return clf, reg, scaler


# ----------------------------------------------------------
# Utilities
# ----------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def get_proba(clf, X: np.ndarray) -> np.ndarray:
    """
    统一从 OneVsRest / 基分类器中拿概率：
    - 优先 predict_proba
    - 其次 decision_function + sigmoid
    - 兼容列表返回 / (n_estimators, n_samples, 2) 的情况
    返回形状：(N, C)
    """
    if hasattr(clf, "predict_proba"):
        try:
            p = clf.predict_proba(X)
            if isinstance(p, list):
                # OneVsRest[List[np.ndarray (N,2)]] -> (N, C)
                p = np.column_stack([pi[:, 1] for pi in p])
            elif isinstance(p, np.ndarray) and p.ndim == 3:
                # (n_estimators, n_samples, 2) -> 取正类并平均/堆叠处理
                p = np.vstack([e[:, 1] for e in p]).T
            return p
        except Exception:
            pass
    if hasattr(clf, "decision_function"):
        s = clf.decision_function(X)
        if s.ndim == 1:
            s = s[:, None]
        return sigmoid(s)
    raise RuntimeError(
        "Classifier does not expose probability or decision scores."
    )


def ece_score(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """
    微平均 ECE（等宽分箱）。
    y_true: (N, C) in {0,1}
    y_prob: (N, C) in [0,1]
    """
    y_true_f = y_true.reshape(-1)
    y_prob_f = y_prob.reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (y_prob_f >= bins[i]) & (y_prob_f < bins[i + 1])
        if m.sum() == 0:
            continue
        conf = y_prob_f[m].mean()
        acc = y_true_f[m].mean()
        ece += (m.mean()) * abs(conf - acc)
    return float(ece)


def select_thresholds(
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    target: str = "f1",
    beta: float = 1.0,
    req_recall: float = 0.9,
    req_precision: float = 0.8,
) -> np.ndarray:
    """
    基于验证集逐类选阈值：
      target ∈ {'f1', 'precision_at_recall', 'recall_at_precision'}
    返回：(C,) 每类一个阈值
    """
    n_classes = y_true_val.shape[1]
    thr = np.zeros(n_classes, dtype=float)
    for c in range(n_classes):
        yt = y_true_val[:, c]
        yp = y_prob_val[:, c]
        grid = np.linspace(0.05, 0.95, 19)
        best, best_t = -1.0, 0.5
        for t in grid:
            pred = (yp >= t).astype(int)
            if target == "f1":
                prec = precision_score(yt, pred, zero_division=0)
                rec = recall_score(yt, pred, zero_division=0)
                num = (1 + beta**2) * prec * rec
                den = beta**2 * prec + rec
                score = (num / den) if den > 0 else 0.0
            elif target == "precision_at_recall":
                rec = recall_score(yt, pred, zero_division=0)
                score = (
                    precision_score(yt, pred, zero_division=0)
                    if rec >= req_recall
                    else -1.0
                )
            elif target == "recall_at_precision":
                prec = precision_score(yt, pred, zero_division=0)
                score = (
                    recall_score(yt, pred, zero_division=0)
                    if prec >= req_precision
                    else -1.0
                )
            else:
                raise ValueError("Unknown target")
            if score > best:
                best, best_t = score, t
        thr[c] = best_t
    return thr


def eval_regression(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    返回 MAE / RMSE / P50_MAE / P90_MAE / R2
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    ae = np.abs(y_true - y_pred).reshape(-1)
    p50 = float(np.percentile(ae, 50))
    p90 = float(np.percentile(ae, 90))
    r2 = float(r2_score(y_true, y_pred, multioutput="uniform_average"))
    return dict(MAE=mae, RMSE=rmse, P50_MAE=p50, P90_MAE=p90, R2=r2)


def flatten_loader(
    loader, num_classes: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 DataLoader 拉平成 (X, y_cls, y_reg)
    - X: 拼接 [最后一步的时序特征, baseline]
    - y_cls: 最后一步的 one-hot bits (C)
    - y_reg: 最后一步的连续值（如 3 路流速）
    """
    X, y_cls, y_reg = [], [], []
    if loader is None:
        return None, None, None
    for x_ts, x_base, y in loader:
        # x_ts: (B, T, D_ts), x_base: (B, D_base), y: (B, T, >= 6)
        x_last = x_ts[:, -1, :].cpu().numpy()
        b_np = x_base.cpu().numpy()
        X.append(np.concatenate((x_last, b_np), axis=1))
        y_np = y.cpu().numpy()
        y_cls.append(y_np[:, -1, :num_classes])
        y_reg.append(y_np[:, -1, num_classes:])
    if len(X) == 0:
        return None, None, None
    return np.vstack(X), np.vstack(y_cls), np.vstack(y_reg)


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument('--ts_pkl', type=str, default='data/output_data_final.pkl', help='时间序列 pkl 路径')
    ap.add_argument('--base_pkl', type=str, default='data/baseline.pkl', help='基线特征 pkl 路径')
    ap.add_argument("--out", default="ml_output_refined")
    ap.add_argument("--history_length", type=int, default=1)
    ap.add_argument("--pred_length", type=int, default=1)
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument(
        "--models",
        nargs="+",
        default=["RandomForest", "XGBoost", "Logistic", "SVM", "GBDT"],
    )
    ap.add_argument(
        "--thr_target",
        default="f1",
        choices=["f1", "precision_at_recall", "recall_at_precision"],
    )
    # 供 data_loader 选择性覆盖的参数（如需要可加更多）
    ap.add_argument("--target_step_hours", type=float, default=1.0)
    ap.add_argument("--orig_step_hours", type=float, default=1.0)
    ap.add_argument("--window_stride", type=int, default=1)
    ap.add_argument(
        "--split_mode", default="patient", choices=["patient", "window"]
    )
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    logger = setup_logger(args.out)

    # ---- Load pickles ----
    with open(args.ts_pkl, "rb") as f:
        Ts_data = pickle.load(f)
    with open(args.base_pkl, "rb") as f:
        Base_data = pickle.load(f)

    # ---- Build loaders（沿用你的 data_loader）----
    train_loader, val_loader, test_loader = data_loader(
        Ts_data,
        Base_data,
        history_length=args.history_length,
        pred_length=args.pred_length,
        classes=args.num_classes,
        batch_size=args.batch_size,
        split_mode=args.split_mode,
        target_step_hours=args.target_step_hours,
        orig_step_hours=args.orig_step_hours,
        window_stride=args.window_stride,
        use_robust_scaler=True,  # 让时序特征在构建数据时也走一次 robust 规范化
    )

    # ---- Flatten ----
    X_train, y_cls_train, y_reg_train = flatten_loader(
        train_loader, args.num_classes
    )
    X_val, y_cls_val, y_reg_val = flatten_loader(
        val_loader, args.num_classes
    )
    X_test, y_cls_test, y_reg_test = flatten_loader(
        test_loader, args.num_classes
    )

    if any(
        v is None
        for v in (
            X_train,
            y_cls_train,
            y_reg_train,
            X_val,
            y_cls_val,
            y_reg_val,
            X_test,
            y_cls_test,
            y_reg_test,
        )
    ):
        raise RuntimeError(
            "数据加载为空，请检查 data_loader 输出与参数。"
        )

    # ---- Train per model ----
    for name in args.models:
        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"===== Training {name} =====")
        clf, reg, scaler = get_model(name)

        # Scaler（只在TRAIN上fit），避免泄漏
        if scaler is not None:
            scaler.fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_val_s = scaler.transform(X_val)
            dump(scaler, os.path.join(out_dir, "scaler.joblib"))
        else:
            X_train_s, X_val_s = X_train, X_val

        # 训练
        clf.fit(X_train_s, y_cls_train)
        reg.fit(X_train_s, y_reg_train)
        dump(clf, os.path.join(out_dir, "classifier.joblib"))
        dump(reg, os.path.join(out_dir, "regressor.joblib"))

        # ---- 验证：概率与阈值 ----
        p_val = get_proba(clf, X_val_s)
        try:
            auc = float(
                roc_auc_score(y_cls_val, p_val, average="macro")
            )
        except Exception:
            auc = float("nan")
        try:
            aupr = float(
                average_precision_score(
                    y_cls_val, p_val, average="macro"
                )
            )
        except Exception:
            aupr = float("nan")

        brier = float(
            brier_score_loss(
                y_cls_val.reshape(-1), p_val.reshape(-1)
            )
        )
        ece = float(ece_score(y_cls_val, p_val))

        thr = select_thresholds(
            y_cls_val, p_val, target=args.thr_target
        )
        np.save(os.path.join(out_dir, "thresholds.npy"), thr)

        # 验证集操作点
        y_pred_val = (p_val >= thr).astype(int)
        f1_micro = float(
            f1_score(y_cls_val, y_pred_val, average="micro")
        )
        prec_micro = float(
            precision_score(
                y_cls_val, y_pred_val, average="micro", zero_division=0
            )
        )
        rec_micro = float(
            recall_score(
                y_cls_val, y_pred_val, average="micro", zero_division=0
            )
        )
        acc_micro = float(
            accuracy_score(
                y_cls_val.reshape(-1), y_pred_val.reshape(-1)
            )
        )

        reg_val_pred = reg.predict(X_val_s)
        reg_summ = eval_regression(y_reg_val, reg_val_pred)

        val_summary = {
            "thresholds": thr.tolist(),
            "AUC_macro_val": auc,
            "AUPRC_macro_val": aupr,
            "Brier": brier,
            "ECE": ece,
            "Acc_micro_val": acc_micro,
            "Precision_micro_val": prec_micro,
            "Recall_micro_val": rec_micro,
            "F1_micro_val": f1_micro,
            "REG_val": reg_summ,
        }
        json.dump(
            val_summary,
            open(os.path.join(out_dir, "val_metrics.json"), "w"),
            indent=2,
        )

        logger.info(
            f"[VAL] {name}: "
            f"AUC_macro={auc:.3f} AUPRC_macro={aupr:.3f} "
            f"Brier={brier:.4f} ECE={ece:.3f} | "
            f"Acc_micro={acc_micro:.3f} "
            f"P/R/F1={prec_micro:.3f}/{rec_micro:.3f}/{f1_micro:.3f} | "
            f"REG {reg_summ}"
        )

    # ---- Test per model ----
    for name in args.models:
        out_dir = os.path.join(args.out, name)
        logger.info(f"===== Test {name} =====")
        clf = load(os.path.join(out_dir, "classifier.joblib"))
        reg = load(os.path.join(out_dir, "regressor.joblib"))
        thr = np.load(os.path.join(out_dir, "thresholds.npy"))

        scaler_path = os.path.join(out_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            scaler = load(scaler_path)
            X_test_s = scaler.transform(X_test)
        else:
            X_test_s = X_test

        # 分类
        p_test = get_proba(clf, X_test_s)
        y_pred = (p_test >= thr).astype(int)

        try:
            auc = float(
                roc_auc_score(y_cls_test, p_test, average="macro")
            )
        except Exception:
            auc = float("nan")
        try:
            aupr = float(
                average_precision_score(
                    y_cls_test, p_test, average="macro"
                )
            )
        except Exception:
            aupr = float("nan")

        f1_micro = float(
            f1_score(y_cls_test, y_pred, average="micro")
        )
        prec_micro = float(
            precision_score(
                y_cls_test, y_pred, average="micro", zero_division=0
            )
        )
        rec_micro = float(
            recall_score(
                y_cls_test, y_pred, average="micro", zero_division=0
            )
        )
        acc_micro = float(
            accuracy_score(
                y_cls_test.reshape(-1), y_pred.reshape(-1)
            )
        )
        brier = float(
            brier_score_loss(
                y_cls_test.reshape(-1), p_test.reshape(-1)
            )
        )
        ece = float(ece_score(y_cls_test, p_test))

        # 回归
        reg_pred = reg.predict(X_test_s)
        reg_summ = eval_regression(y_reg_test, reg_pred)

        report = dict(
            AUC_macro=auc,
            AUPRC_macro=aupr,
            Accuracy_micro=acc_micro,
            Precision_micro=prec_micro,
            Recall_micro=rec_micro,
            F1_micro=f1_micro,
            Brier=brier,
            ECE=ece,
            **reg_summ,
        )
        json.dump(
            report,
            open(os.path.join(out_dir, "test_metrics.json"), "w"),
            indent=2,
        )
        logger.info(f"[TEST] {name}: {report}")

        # ===== ROC 曲线保存（Test，按类，多曲线在一张图上，默认开启）=====
        try:
            import matplotlib.pyplot as plt

            # 科研风格：适当设置图像大小、字体等
            plt.figure(figsize=(5, 5), dpi=300)

            # 设置全局样式（可根据需要调整）
            plt.rcParams.update(
                {
                    "font.size": 10,
                    "axes.linewidth": 1.0,
                    "xtick.direction": "in",
                    "ytick.direction": "in",
                    "xtick.major.size": 4,
                    "ytick.major.size": 4,
                    "xtick.minor.size": 2,
                    "ytick.minor.size": 2,
                    "axes.unicode_minus": False,
                }
            )

            # 红绿蓝配色循环
            color_cycle = ["#DE3826", "#05BF78", "#4083BA"]
            curve_idx = 0

            n_classes = y_cls_test.shape[1]
            # 默认 class 名称
            class_names = [f"Class {i}" for i in range(n_classes)]
            # 如果你想沿用 water/crystal/colloid，可以在这里手动改名
            # 例如：class_names = ["water", "crystal", "colloid"]

            idx_map = list(enumerate(class_names))

            for i, name_c in idx_map:
                # 只画真正的二分类 ROC
                if np.unique(y_cls_test[:, i]).size == 2:
                    try:
                        fpr, tpr, _ = roc_curve(
                            y_cls_test[:, i], p_test[:, i]
                        )

                        # 名称映射（兼容你之前片段里的命名）
                        if name_c.lower() == "water":
                            name_plot = "Glucose"
                        elif name_c.lower() == "crystal":
                            name_plot = "Crystalloid"
                        elif name_c.lower() == "colloid":
                            name_plot = "Colloid"
                        else:
                            name_plot = name_c

                        color = color_cycle[curve_idx % len(color_cycle)]
                        curve_idx += 1

                        plt.plot(
                            fpr,
                            tpr,
                            # 如果想展示 AUC，可以在上面先算 per-class AUC，再把 AUC 塞进 label
                            # label=f"{name_plot} (AUC = {auc_val:.3f})",
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

            # 不加 title，保持科研风格
            # 图例放右下角并去掉边框
            plt.legend(
                loc="lower right",
                frameon=False,
                fontsize=8,
                handlelength=1.8,
            )

            # 细网格
            plt.grid(
                True,
                linestyle=":",
                linewidth=0.6,
                alpha=0.7,
            )

            plt.tight_layout()

            save_path = os.path.join(out_dir, f"roc_test_{name}.png")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            logger.info(f"ROC curve saved to: {save_path}")
        except Exception as e:
            logger.warning(f"Failed to save ROC curve: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
