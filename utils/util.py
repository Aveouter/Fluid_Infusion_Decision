
import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, roc_auc_score,
    mean_absolute_error, r2_score,
)
# =============================================================================
# Zero-Inflated LogNormal NLL (stable)
# =============================================================================
def ziln_nll_stable(p0_raw, mu, log_sigma, y, *, p0_is_logit=True, eps=1e-12, reduction="mean"):
    """Zero-Inflated LogNormal negative log-likelihood (NLL), numerically stable."""
    if p0_is_logit:
        p0 = torch.sigmoid(p0_raw)  # (0,1)
    else:
        p0 = p0_raw.clamp(min=eps, max=1.0 - eps)

    sigma = F.softplus(log_sigma).clamp_min(1e-6)  # >0
    y_pos = y.clamp_min(0.0)

    is_zero = (y_pos <= 0.0)
    is_pos = ~is_zero

    ll = torch.zeros_like(y_pos)

    if is_pos.any():
        y_sel = y_pos[is_pos].clamp_min(eps)
        log_y = torch.log(y_sel)
        log_pdf = - (torch.log(y_sel) + torch.log(sigma[is_pos]) + 0.5 * torch.log(torch.tensor(2.0 * np.pi, device=y.device))) \
                  - 0.5 * ((log_y - mu[is_pos]) ** 2) / (sigma[is_pos] ** 2)
        ll_pos = torch.log1p(-p0[is_pos] + eps) + log_pdf
        ll[is_pos] = ll_pos

    if is_zero.any():
        ll_zero = torch.log(p0[is_zero].clamp_min(eps))
        ll[is_zero] = ll_zero

    nll = -ll

    if reduction == "mean":
        return torch.nanmean(nll)
    elif reduction == "sum":
        return torch.nansum(nll)
    else:
        return nll

# =============================================================================
# Activation saturation monitors (optional)
# =============================================================================
def register_activation_saturation_hooks(model, sat_eps=1e-3, pct_warn=0.8):
    handles = []
    def hook_sigmoid(name):
        def fn(mod, inp, out):
            y = out.detach()
            sat = ((y < sat_eps) | (y > 1 - sat_eps)).float().mean().item()
            if sat > pct_warn:
                logging.warning(f"[ActSat] {name}: sigmoid saturated {sat*100:.1f}%")
        return fn
    def hook_tanh(name):
        def fn(mod, inp, out):
            y = out.detach()
            sat = ((y < -1 + sat_eps) | (y > 1 - sat_eps)).float().mean().item()
            if sat > pct_warn:
                logging.warning(f"[ActSat] {name}: tanh saturated {sat*100:.1f}%")
        return fn
    def hook_relu(name):
        def fn(mod, inp, out):
            dead = (out.detach() <= 0).float().mean().item()
            if dead > pct_warn:
                logging.warning(f"[ActSat] {name}: ReLU dead {dead*100:.1f}% (many zeros)")
        return fn
    for name, m in model.named_modules():
        if isinstance(m, nn.Sigmoid):
            handles.append(m.register_forward_hook(hook_sigmoid(name)))
        elif isinstance(m, nn.Tanh):
            handles.append(m.register_forward_hook(hook_tanh(name)))
        elif isinstance(m, nn.ReLU):
            handles.append(m.register_forward_hook(hook_relu(name)))
    return handles

# =============================================================================
# Metrics & Utils
# =============================================================================
def _safe_div(a, b, default=0.0):
    return float(a) / float(b) if b and b > 0 else default

def fbeta_opt_thresh_per_class(y_true_bin, y_prob, betas=None, grid=None):
    """Return per-class best thresholds by maximizing F-beta on validation set."""
    K = y_true_bin.shape[1]
    if betas is None: betas = [1.0]*K
    if grid is None:  grid = np.linspace(0.05, 0.95, 19)
    from sklearn.metrics import fbeta_score
    best = []
    for k in range(K):
        t_best, f_best = 0.5, -1.0
        for t in grid:
            pred = (y_prob[:, k] >= t).astype(int)
            f = fbeta_score(y_true_bin[:, k], pred, beta=betas[k], zero_division=0)
            if f > f_best:
                t_best, f_best = t, f
        best.append(t_best)
    return best

def smooth_probs_over_time(prob, win=5, only_idx=None):
    """Temporal moving-average smoothing on P dimension. prob:(B,P,C)."""
    if win <= 1: return prob
    B,P,C = prob.shape
    out = prob.clone()
    pad = win//2
    for c in range(C):
        if (only_idx is not None) and (c not in set(only_idx)):
            continue
        x = prob[:,:,c:c+1].transpose(1,2)            # (B,1,P)
        x = torch.nn.functional.pad(x, (pad,pad), mode='replicate')
        kernel = torch.ones(1,1,win, device=prob.device) / win
        y = torch.nn.functional.conv1d(x, kernel)     # (B,1,P)
        out[:,:,c] = y.transpose(1,2).squeeze(-1)
    return out

def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_pred - y_true)**2)))

def tolerance_accuracy_abs(y_true, y_pred, tol_ml=250.0):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_pred - y_true) <= tol_ml))

def tolerance_accuracy_pct(y_true, y_pred, pct=0.10, eps=1e-8):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_pred - y_true) <= (pct*np.maximum(np.abs(y_true), eps))))

# ---- Robust scaler per-class (median/IQR on log1p positive targets) ----
def compute_per_class_robust_scaler(loader, device, C=3, P_key_slice=None, log1p=True, sample_limit=200000):
    """Scan train loader to compute per-class median & IQR for regression robust scaling."""
    vals = [[] for _ in range(C)]
    with torch.no_grad():
        for _, _, label_data in loader:
            label_data = label_data.to(device)
            labels_flow = label_data[..., C:]  # (B,T,>=C)
            if P_key_slice is not None:
                labels_flow = labels_flow[:, P_key_slice, :]
            if labels_flow.size(-1) < C: continue
            x = labels_flow
            x = torch.clamp(x, min=0.0)  # only positive
            if log1p:
                x = torch.log1p(x + 1e-8)
            x = x.detach().cpu().numpy()
            for c in range(C):
                vc = x[..., c].reshape(-1)
                vc = vc[np.isfinite(vc)]
                if vc.size:
                    if len(vals[c]) < sample_limit:
                        vals[c].append(vc)
    scaler = []
    for c in range(C):
        if len(vals[c]) == 0:
            scaler.append(dict(median=0.0, iqr=1.0))
        else:
            v = np.concatenate(vals[c], axis=0)
            med = np.median(v)
            q1, q3 = np.percentile(v, [25, 75])
            iqr = max(q3 - q1, 1e-6)
            scaler.append(dict(median=float(med), iqr=float(iqr)))
    return scaler

def robust_scale_torch(x, med, iqr):
    return (x - med) / iqr

