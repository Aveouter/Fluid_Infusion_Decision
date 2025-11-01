# -*- coding: utf-8 -*-
"""
Refactored ML baselines for burn-fluid tasks
- Fixes data leakage (scaler/thresholds) and probability handling
- Adds robust metrics (AUPRC, Brier, ECE, operating-point PR, P50/P90 MAE)
- Validates thresholds on val set and freezes them for test
- Persists scaler/thresholds/models per baseline

Run:
python ml_baselines_refactor.py \
  --ts_pkl /baksv/CIGIT/GXN_Liuxy/fluid/data/output_data.pkl \
  --base_pkl /baksv/CIGIT/GXN_Liuxy/fluid/data/baseline.pkl \
  --out ml_output_refined
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
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, brier_score_loss,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from utils.data_loader import data_loader

np.random.seed(42)

# ------------------------- utils -------------------------

def setup_logger(output_path):
    os.makedirs(output_path, exist_ok=True)
    logger = logging.getLogger('mlref')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(output_path, 'train_ml.log')); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def get_model(name: str):
    name = name.lower()
    if name == 'randomforest':
        clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=200, max_depth=None, class_weight='balanced', random_state=42))
        reg = MultiOutputRegressor(RandomForestRegressor(n_estimators=400, random_state=42))
        scaler = None
    elif name == 'xgboost':
        clf = OneVsRestClassifier(XGBClassifier(eval_metric='logloss', max_depth=4, subsample=0.9, colsample_bytree=0.9, random_state=42))
        reg = MultiOutputRegressor(XGBRegressor(max_depth=4, subsample=0.9, colsample_bytree=0.9, random_state=42))
        scaler = None
    elif name == 'logistic':
        clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs', random_state=42))
        reg = MultiOutputRegressor(LinearRegression())
        scaler = StandardScaler()
    elif name == 'svm':
        clf = OneVsRestClassifier(SVC(probability=True, class_weight='balanced', random_state=42))
        reg = MultiOutputRegressor(SVR())
        scaler = StandardScaler()
    elif name == 'gbdt':
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        clf = OneVsRestClassifier(GradientBoostingClassifier(random_state=42))
        reg = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
        scaler = None
    else:
        raise ValueError(f"Unsupported model: {name}")
    return clf, reg, scaler


def flatten(loader):
    X, y_cls, y_reg = [], [], []
    for x_ts, x_base, y in loader:
        x = np.concatenate((x_ts[:, -1, :], x_base), axis=1)
        y_cls.append(y[:, -1, :args.num_classes])
        y_reg.append(y[:, -1, args.num_classes:])
        X.append(x)
    if len(X) == 0:
        return None, None, None
    return np.vstack(X), np.vstack(y_cls), np.vstack(y_reg)


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def get_proba(clf, X):
    # Prefer predict_proba from wrapper; fallback to decision_function
    if hasattr(clf, 'predict_proba'):
        try:
            p = clf.predict_proba(X)
            # OneVsRest may return list or array; standardize to ndarray
            if isinstance(p, list):
                p = np.column_stack([pi[:, 1] for pi in p])
            elif p.ndim == 3:  # (n_estimators, n_samples, 2)
                p = np.vstack([e[:,1] for e in p]).T
            return p
        except Exception:
            pass
    if hasattr(clf, 'decision_function'):
        s = clf.decision_function(X)
        if s.ndim == 1:
            s = s[:, None]
        return sigmoid(s)
    raise RuntimeError('Classifier does not expose probability or decision scores.')


def ece_score(y_true, y_prob, n_bins: int = 10) -> float:
    # micro-average ECE over classes
    y_true_f = y_true.reshape(-1)
    y_prob_f = y_prob.reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        m = (y_prob_f >= bins[i]) & (y_prob_f < bins[i+1])
        if m.sum() == 0:
            continue
        conf = y_prob_f[m].mean()
        acc = y_true_f[m].mean()
        ece += (m.mean()) * abs(conf - acc)
    return float(ece)


def select_thresholds(y_true_val: np.ndarray, y_prob_val: np.ndarray, target: str = 'f1', beta: float = 1.0) -> np.ndarray:
    """Per-class threshold selection on validation set.
    target in {'f1', 'precision_at_recall', 'recall_at_precision'}.
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
            if target == 'f1':
                num = (1+beta**2) * precision_score(yt, pred, zero_division=0) * recall_score(yt, pred, zero_division=0)
                den = beta**2 * precision_score(yt, pred, zero_division=0) + recall_score(yt, pred, zero_division=0)
                score = (num/den) if den>0 else 0.0
            elif target == 'precision_at_recall':
                if recall_score(yt, pred, zero_division=0) >= 0.9:
                    score = precision_score(yt, pred, zero_division=0)
                else:
                    score = -1.0
            elif target == 'recall_at_precision':
                if precision_score(yt, pred, zero_division=0) >= 0.8:
                    score = recall_score(yt, pred, zero_division=0)
                else:
                    score = -1.0
            else:
                raise ValueError('Unknown target')
            if score > best:
                best, best_t = score, t
        thr[c] = best_t
    return thr


def eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    err = y_true - y_pred
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    # robust tails
    ae = np.abs(err).reshape(-1)
    p50 = float(np.percentile(ae, 50))
    p90 = float(np.percentile(ae, 90))
    return dict(MAE=mae, RMSE=rmse, P50_MAE=p50, P90_MAE=p90)

# ------------------------- main -------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ts_pkl', required=True)
    ap.add_argument('--base_pkl', required=True)
    ap.add_argument('--out', default='ml_output_refined')
    ap.add_argument('--history_length', type=int, default=1)
    ap.add_argument('--pred_length', type=int, default=1)
    ap.add_argument('--num_classes', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=12)
    ap.add_argument('--models', nargs='+', default=['RandomForest','XGBoost','Logistic','SVM','GBDT'])
    ap.add_argument('--thr_target', default='f1', choices=['f1','precision_at_recall','recall_at_precision'])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    logger = setup_logger(args.out)

    with open(args.ts_pkl, 'rb') as f:
        Ts_data = pickle.load(f)
    with open(args.base_pkl, 'rb') as f:
        Base_data = pickle.load(f)

    train_loader, val_loader, test_loader = data_loader(
        Ts_data, Base_data,
        args.history_length, args.pred_length,
        args.num_classes, args.batch_size
    )

    def flatten(loader):
        X, y_class, y_reg = [], [], []
        for x_ts, x_base, y in loader:
            x = np.concatenate((x_ts[:, -1, :], x_base), axis=1)
            y_class.append(y[:, -1, :args.num_classes])
            y_reg.append(y[:, -1, args.num_classes:])
            X.append(x)
        if len(X) == 0:
            return None, None, None
        return np.vstack(X), np.vstack(y_class), np.vstack(y_reg)

    X_train, y_cls_train, y_reg_train = flatten(train_loader)
    X_val, y_cls_val, y_reg_val = flatten(val_loader)
    X_test, y_cls_test, y_reg_test = flatten(test_loader)

    for name in args.models:
        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f'Training {name}')
        clf, reg, scaler = get_model(name)

        # Fit scaler on TRAIN only, persist
        if scaler is not None:
            scaler.fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_val_s   = scaler.transform(X_val)
            dump(scaler, os.path.join(out_dir, 'scaler.joblib'))
        else:
            X_train_s, X_val_s = X_train, X_val

        # Fit models
        clf.fit(X_train_s, y_cls_train)
        reg.fit(X_train_s, y_reg_train)
        dump(clf, os.path.join(out_dir, 'classifier.joblib'))
        dump(reg, os.path.join(out_dir, 'regressor.joblib'))

        # Validation probabilities + thresholds
        p_val = get_proba(clf, X_val_s)
        # Global summaries
        try:
            auc = roc_auc_score(y_cls_val, p_val, average='macro')
            aupr = average_precision_score(y_cls_val, p_val, average='macro')
        except Exception:
            auc = np.nan; aupr = np.nan
        brier = brier_score_loss(y_cls_val.reshape(-1), p_val.reshape(-1))
        ece = ece_score(y_cls_val, p_val)

        thr = select_thresholds(y_cls_val, p_val, target=args.thr_target)
        json.dump({'thresholds': thr.tolist(), 'AUC_macro_val': float(auc), 'AUPRC_macro_val': float(aupr), 'Brier': float(brier), 'ECE': float(ece)},
                  open(os.path.join(out_dir, 'val_metrics.json'), 'w'))
        np.save(os.path.join(out_dir, 'thresholds.npy'), thr)

        # Validation operating point
        y_pred_val = (p_val >= thr).astype(int)
        f1_micro = f1_score(y_cls_val, y_pred_val, average='micro')
        prec_micro = precision_score(y_cls_val, y_pred_val, average='micro', zero_division=0)
        rec_micro = recall_score(y_cls_val, y_pred_val, average='micro', zero_division=0)
        reg_val_pred = reg.predict(X_val_s)
        reg_summ = eval_regression(y_reg_val, reg_val_pred)
        logger.info(f"[VAL] {name}: AUC_macro={auc:.3f} AUPRC_macro={aupr:.3f} Brier={brier:.4f} ECE={ece:.3f} | micro P/R/F1={prec_micro:.3f}/{rec_micro:.3f}/{f1_micro:.3f} | REG {reg_summ}")

    # ===== Test evaluation =====
    for name in args.models:
        out_dir = os.path.join(args.out, name)
        logger.info(f'Test {name}')
        clf = load(os.path.join(out_dir, 'classifier.joblib'))
        reg = load(os.path.join(out_dir, 'regressor.joblib'))
        thr = np.load(os.path.join(out_dir, 'thresholds.npy'))
        scaler_path = os.path.join(out_dir, 'scaler.joblib')
        if os.path.exists(scaler_path):
            scaler = load(scaler_path)
            X_test_s = scaler.transform(X_test)
        else:
            X_test_s = X_test

        p_test = get_proba(clf, X_test_s)
        y_pred = (p_test >= thr).astype(int)

        # Classification metrics
        auc = roc_auc_score(y_cls_test, p_test, average='macro')
        aupr = average_precision_score(y_cls_test, p_test, average='macro')
        f1_micro = f1_score(y_cls_test, y_pred, average='micro')
        prec_micro = precision_score(y_cls_test, y_pred, average='micro', zero_division=0)
        rec_micro = recall_score(y_cls_test, y_pred, average='micro', zero_division=0)
        brier = brier_score_loss(y_cls_test.reshape(-1), p_test.reshape(-1))
        ece = ece_score(y_cls_test, p_test)

        # Regression metrics
        reg_pred = reg.predict(X_test_s)
        reg_summ = eval_regression(y_reg_test, reg_pred)

        report = dict(AUC_macro=auc, AUPRC_macro=aupr, Precision_micro=prec_micro, Recall_micro=rec_micro, F1_micro=f1_micro, Brier=brier, ECE=ece, **reg_summ)
        json.dump(report, open(os.path.join(out_dir, 'test_metrics.json'), 'w'), indent=2)
        logger.info(f"[TEST] {name}: {report}")

    print('Done.')
