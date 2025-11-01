# utils/ts_augmentation_toolkit.py
"""
Time-series data augmentation toolkit (label-aware) for your pipeline.

- Per-sample transforms: jitter, scaling, time mask, time warp, magnitude warp, permutation.
- Label-aware: DO NOT flip flow semantics; optional recompute of cumulative-volume columns.
- Batch-level mixup for multilabel + regression.

Compatible with BOTH old-style kwargs (e.g. time_warping_prob, magnitude_warping_prob, ...)
and the new dataclass-style kwargs (p_timewarp, p_magwarp, ...).
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F  # 保留这个没问题


# -----------------------------
# Utilities
# -----------------------------
def _rand_bool(p: float) -> bool:
    return torch.rand(()) < p

def _clamp_pos(x: torch.Tensor, min_val: float = 0.0, max_val: Optional[float] = None) -> torch.Tensor:
    if max_val is None:
        return torch.clamp(x, min=min_val)
    return torch.clamp(x, min=min_val, max=max_val)


# -----------------------------
# Pointwise / local transforms
# -----------------------------
def jitter(x: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
    """Additive white noise. x: (T, F)."""
    if sigma <= 0:
        return x
    return x + torch.randn_like(x) * sigma

def scaling(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Feature-wise multiplicative scaling by 1+N(0,sigma)."""
    if sigma <= 0:
        return x
    s = 1.0 + torch.randn((1, x.size(1)), device=x.device, dtype=x.dtype) * sigma
    return x * s

def time_mask(x: torch.Tensor, p: float = 0.1, max_width_ratio: float = 0.1) -> torch.Tensor:
    """Mask a random temporal block by linear interpolation to keep smoothness."""
    T = x.size(0)
    if not _rand_bool(p) or T < 2:
        return x
    w = max(1, int(T * max_width_ratio))
    t0 = torch.randint(0, T - w + 1, (1,), device=x.device).item()
    t1 = t0 + w
    x0 = x[t0 - 1] if t0 > 0 else x[t0]
    x1 = x[t1] if t1 < T else x[t1 - 1]
    ramp = torch.linspace(0, 1, steps=w, device=x.device, dtype=x.dtype).unsqueeze(1)
    x_mask = x.clone()
    x_mask[t0:t1] = x0 * (1 - ramp) + x1 * ramp
    return x_mask

def magnitude_warp(x: torch.Tensor, sigma: float = 0.2, knots: int = 4) -> torch.Tensor:
    """Smooth magnitude modulation via piecewise linear interpolation of random knots."""
    if sigma <= 0 or x.size(0) < knots + 1:
        return x
    T = x.size(0)
    # random knot values around 1.0
    knot_vals = 1.0 + torch.randn((knots + 2, 1), device=x.device, dtype=x.dtype) * sigma
    knot_pos = torch.linspace(0, T - 1, steps=knots + 2, device=x.device, dtype=x.dtype)
    t = torch.arange(T, device=x.device, dtype=x.dtype)

    # torch.interp may be missing on older PyTorch; fall back to numpy
    try:
        scale = torch.interp(t, knot_pos, knot_vals.squeeze(1)).unsqueeze(1)  # type: ignore[attr-defined]
    except Exception:
        scale_np = np.interp(
            t.detach().cpu().numpy(),
            knot_pos.detach().cpu().numpy(),
            knot_vals.squeeze(1).detach().cpu().numpy(),
        )
        scale = torch.from_numpy(scale_np).to(device=x.device, dtype=x.dtype).unsqueeze(1)

    return x * scale

def time_warp_uniform(x: torch.Tensor, max_stretch: float = 0.2) -> torch.Tensor:
    """Uniform time stretch/compress (resample to length T±k then back to T)."""
    if max_stretch <= 0 or x.size(0) < 3:
        return x
    T, feat_dim = x.size()  # 不要用 F 作为变量名

    rate = 1.0 + (torch.rand(()) * 2 - 1) * max_stretch  # in [1-max, 1+max]
    new_T = max(2, int(T * rate))

    # 1D interpolate expects (N, C, L)
    x_new = torch.nn.functional.interpolate(
        x.T.unsqueeze(0), size=new_T, mode="linear", align_corners=False
    ).squeeze(0).T
    x_back = torch.nn.functional.interpolate(
        x_new.T.unsqueeze(0), size=T, mode="linear", align_corners=False
    ).squeeze(0).T
    return x_back

def permute_blocks(x: torch.Tensor, n_segs: int = 3, p: float = 0.5) -> torch.Tensor:
    """Randomly permute temporal segments; keeps intra-segment dynamics."""
    if n_segs <= 1 or not _rand_bool(p):
        return x
    T = x.size(0)
    seg_len = max(1, T // n_segs)
    chunks = [x[i * seg_len : (i + 1) * seg_len] for i in range(n_segs - 1)] + [x[(n_segs - 1) * seg_len :]]
    order = torch.randperm(len(chunks), device=x.device)
    return torch.cat([chunks[i] for i in order], dim=0)


# -----------------------------
# Label-aware helpers
# -----------------------------
def recompute_cumvol(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    rate_cols_in_y=(3, 4, 5),
    type_cols_in_y=(0, 1, 2),
    cumvol_cols_in_x=(6, 7, 8, 9),
) -> torch.Tensor:
    """
    Recompute cumulative volumes in `x` (T,F) using label rates/types from `y` (T,>=6).
    Adjust `cumvol_cols_in_x` according to your actual feature order.
    cumulative dims: [water, crystal, colloid, total].
    """
    T = x.size(0)
    y_np = y.detach().cpu().numpy()
    rates = y_np[:, list(rate_cols_in_y)]  # (T,3)
    types = y_np[:, list(type_cols_in_y)]  # (T,3) 0/1
    eff = rates * types  # (T,3) per-hour volume
    cum = eff.cumsum(axis=0)
    total = cum.sum(axis=1, keepdims=True)
    all_cum = torch.tensor(np.concatenate([cum, total], axis=1), device=x.device, dtype=x.dtype)
    x2 = x.clone()
    for k, col in enumerate(cumvol_cols_in_x):
        x2[:, col] = all_cum[:, k]
    return x2


# -----------------------------
# Batch-level augmentation (mixup)
# -----------------------------
def apply_mixup_batch(
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    alpha: float = 0.4,
    multilabel_soft: bool = True,
    flow_mix_mode: str = "linear",
):
    """
    Mixup for (time_series, baseline, labels) with multilabel + regression.

    - multilabel_soft=True => y_type = λ*y1 + (1-λ)*y2 (BCE 支持软标签)
      False => y_type = OR(y1, y2)
    - flow_mix_mode: 'linear' or 'masked_linear' (masked keeps 0 if both are 0)
    """
    x, b, y = batch  # x:(B,T,F)  b:(B,...)  y:(B,T,>=6)
    B = x.size(0)
    if B < 2:
        return batch
    beta = torch.distributions.Beta(alpha, alpha)
    lam = beta.sample((B,)).to(x.device)
    idx = torch.randperm(B, device=x.device)

    lam_x = lam.view(B, 1, 1)
    lam_b = lam.view(B, 1)
    lam_y = lam.view(B, 1, 1)

    x_mix = lam_x * x + (1 - lam_x) * x[idx]
    b_mix = lam_b * b + (1 - lam_b) * b[idx]

    # labels: [type(3), flow(>=3)]
    y1, y2 = y, y[idx]
    y_type1, y_flow1 = y1[..., :3], y1[..., 3:6]
    y_type2, y_flow2 = y2[..., :3], y2[..., 3:6]

    if multilabel_soft:
        y_type_mix = lam_y * y_type1 + (1 - lam_y) * y_type2
    else:
        y_type_mix = torch.clamp(y_type1 + y_type2, 0, 1)

    if flow_mix_mode == "linear":
        y_flow_mix = lam_y * y_flow1 + (1 - lam_y) * y_flow2
    else:  # masked_linear: keep zero if both are zero
        both_zero = (y_flow1 <= 0) & (y_flow2 <= 0)
        y_flow_mix = lam_y * y_flow1 + (1 - lam_y) * y_flow2
        y_flow_mix[both_zero] = 0.0

    if y.size(-1) > 6:
        y_mix = torch.cat([y_type_mix, y_flow_mix, y[..., 6:]], dim=-1)
    else:
        y_mix = torch.cat([y_type_mix, y_flow_mix], dim=-1)

    return x_mix, b_mix, y_mix


# -----------------------------
# Augmenter class (per-sample)
# -----------------------------
@dataclass(init=False)
class TSAugmenter:
    # 新风格参数（建议使用）
    p_jitter: float = 0.5
    jitter_sigma: float = 0.01
    p_scale: float = 0.3
    scale_sigma: float = 0.05
    p_mask: float = 0.2
    mask_max_width: float = 0.15
    p_timewarp: float = 0.2
    max_stretch: float = 0.15
    p_perm: float = 0.2
    n_perm_segs: int = 3
    p_magwarp: float = 0.2
    magwarp_sigma: float = 0.15
    clamp_positive_cols: Optional[Tuple[int, ...]] = None  # columns (in x) that must stay ≥0
    x_max_clip: Optional[float] = None
    recompute_cumvol_after: bool = False

    def __init__(
        self,
        p_jitter: float = 0.5,
        jitter_sigma: float = 0.01,
        p_scale: float = 0.3,
        scale_sigma: float = 0.05,
        p_mask: float = 0.2,
        mask_max_width: float = 0.15,
        p_timewarp: float = 0.2,
        max_stretch: float = 0.15,
        p_perm: float = 0.2,
        n_perm_segs: int = 3,
        p_magwarp: float = 0.2,
        magwarp_sigma: float = 0.15,
        clamp_positive_cols: Optional[Tuple[int, ...]] = None,
        x_max_clip: Optional[float] = None,
        recompute_cumvol_after: bool = False,
        **legacy_kwargs,
    ):
        """
        兼容旧风格关键字（会被映射或忽略）：
        - time_warping_prob -> p_timewarp
        - scaling_prob      -> p_scale
        - jittering_prob    -> p_jitter
        - permutation_prob  -> p_perm
        - magnitude_warping_prob -> p_magwarp
        - window_warping_prob / window_slicing_prob / cutout_prob / cutmix_prob / mixup_prob / alpha -> 忽略或你可自行扩展
        """
        # 先用新风格赋值
        self.p_jitter = p_jitter
        self.jitter_sigma = jitter_sigma
        self.p_scale = p_scale
        self.scale_sigma = scale_sigma
        self.p_mask = p_mask
        self.mask_max_width = mask_max_width
        self.p_timewarp = p_timewarp
        self.max_stretch = max_stretch
        self.p_perm = p_perm
        self.n_perm_segs = n_perm_segs
        self.p_magwarp = p_magwarp
        self.magwarp_sigma = magwarp_sigma
        self.clamp_positive_cols = clamp_positive_cols
        self.x_max_clip = x_max_clip
        self.recompute_cumvol_after = recompute_cumvol_after

        # 旧→新 名称映射
        mapping = {
            "time_warping_prob": "p_timewarp",
            "scaling_prob": "p_scale",
            "jittering_prob": "p_jitter",
            "permutation_prob": "p_perm",
            "magnitude_warping_prob": "p_magwarp",
            # 下面这些旧参数暂未使用；如果你将来实现，可在此读取
            # "window_warping_prob": None,
            # "window_slicing_prob": None,
            # "cutout_prob": None,
            # "cutmix_prob": None,
            # "mixup_prob": None,
            # "alpha": None,
        }
        for old, new in mapping.items():
            if old in legacy_kwargs and new is not None:
                setattr(self, new, float(legacy_kwargs[old]))

        # 忽略其余未知旧参数
        # 如果你想对未知键给出提示，可解开下面注释：
        # for k in legacy_kwargs:
        #     if k not in mapping:
        #         print(f"[TSAugmenter] Ignore unknown legacy kwarg: {k}")

    def __call__(self, x: torch.Tensor, b: torch.Tensor, y: torch.Tensor):
        """x:(T,F), b:(...), y:(T,>=6)  -> possibly augmented triplet."""
        x_aug = x
        if _rand_bool(self.p_jitter):
            x_aug = jitter(x_aug, self.jitter_sigma)
        if _rand_bool(self.p_scale):
            x_aug = scaling(x_aug, self.scale_sigma)
        if _rand_bool(self.p_mask):
            x_aug = time_mask(x_aug, p=1.0, max_width_ratio=self.mask_max_width)
        if _rand_bool(self.p_timewarp):
            x_aug = time_warp_uniform(x_aug, self.max_stretch)
        if _rand_bool(self.p_perm):
            x_aug = permute_blocks(x_aug, self.n_perm_segs, p=1.0)
        if _rand_bool(self.p_magwarp):
            x_aug = magnitude_warp(x_aug, self.magwarp_sigma)

        # clamp required positive-only channels (e.g., flow-derived engineered features)
        if self.clamp_positive_cols is not None:
            for c in self.clamp_positive_cols:
                x_aug[:, c] = _clamp_pos(x_aug[:, c])
        if self.x_max_clip is not None:
            x_aug = torch.clamp(x_aug, max=self.x_max_clip)

        # keep consistency for engineered cumulative columns (optional)
        if self.recompute_cumvol_after:
            x_aug = recompute_cumvol(x_aug, y)

        return x_aug, b, y


__all__ = [
    "TSAugmenter",
    "apply_mixup_batch",
    "recompute_cumvol",
    "jitter",
    "scaling",
    "time_mask",
    "magnitude_warp",
    "time_warp_uniform",
    "permute_blocks",
]
