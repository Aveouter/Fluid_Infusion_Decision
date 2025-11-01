# -*- coding: utf-8 -*-
"""
Multi-Scale Time-FiLM with ZILN (merged & improved)
- Stable ZILN: p0_logit + BCEWithLogits, weighted composition
- Multi-Scale FiLM: hour/day paths with gamma tanh-bounds and dual gating
- Time encoding in float32 for numerical stability, then cast back
- Optional causal attention in TemporalEncoder
"""

from typing import Tuple, Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# Core blocks
# ================================================================
class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PreNorm residual
        return x + self.net(self.norm(x))


class MHABlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_n = self.norm(x)
        out, _ = self.attn(x_n, x_n, x_n, attn_mask=attn_mask)
        return x + self.drop(out)


class PositionalEncoding(nn.Module):
    """Sinusoidal PE (batch_first): (B,T,E)."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (T,1)
        div = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim, dtype=torch.float)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1,T,E)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :]


class BaselineEncoder(nn.Module):
    """Encodes (B, D_base) -> (B, E) with pre-norm residual blocks."""
    def __init__(self, input_dim: int, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, embed_dim)
        self.block1 = MHABlock(embed_dim, num_heads, dropout)
        self.ff1 = FeedForward(embed_dim, hidden_dim, dropout)
        self.block2 = MHABlock(embed_dim, num_heads, dropout)
        self.ff2 = FeedForward(embed_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D_base)
        x = self.in_proj(x).unsqueeze(1)  # (B,1,E)
        x = self.block1(x)
        x = self.ff1(x)
        x = self.block2(x)
        x = self.ff2(x)
        return x.squeeze(1)  # (B,E)


class TemporalEncoder(nn.Module):
    """
    Simple temporal encoder: Linear -> PE -> [MHA+FFN]*L => (B,T,E).
    Optionally causal (no attention to future positions).
    """
    def __init__(self, input_dim: int, embed_dim: int, num_heads: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, embed_dim)
        self.pe = PositionalEncoding(embed_dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([MHABlock(embed_dim, num_heads, dropout), FeedForward(embed_dim, hidden_dim, dropout)])
            for _ in range(num_layers)
        ])
        self.causal = causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pe(self.in_proj(x))
        if self.causal:
            T = x.size(1)
            # True positions are masked (not allowed to attend)
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            for attn, ffn in self.layers:
                x = attn(x, attn_mask=mask)
                x = ffn(x)
        else:
            for attn, ffn in self.layers:
                x = attn(x)
                x = ffn(x)
        return x


# ================================================================
# Multi-Scale Time Encoding (Hour-of-day + Day-index)
# ================================================================
class HourDayTimeEncoder(nn.Module):
    """
    Build hour-of-day and day-index encodings.
    - hour-of-day: sin/cos -> small MLP -> project to E
    - day-index  : embedding(0..max_days-1) -> project to E
    Optionally add a tiny sinusoidal PE for stabilization.

    Args:
        embed_dim: target E
        hour_proj_dim: internal dim for hour features
        day_embed_dim: embedding dim for day index
        max_days: typically 3 for 0–72h
        add_rel_pe: add small sinusoidal PE
    """
    def __init__(self,
                 embed_dim: int,
                 hour_proj_dim: int = 32,
                 day_embed_dim: int = 16,
                 max_days: int = 3,
                 add_rel_pe: bool = True):
        super().__init__()
        self.max_days = max_days
        self.add_rel_pe = add_rel_pe
        self.day_emb = nn.Embedding(max_days, day_embed_dim)
        self.hour_proj = nn.Sequential(
            nn.Linear(2, hour_proj_dim), nn.GELU(), nn.Linear(hour_proj_dim, embed_dim)
        )
        self.day_proj = nn.Sequential(
            nn.Linear(day_embed_dim, embed_dim)
        )
        if add_rel_pe:
            self.rel_pe = PositionalEncoding(embed_dim)
        else:
            self.rel_pe = None
        self.out_norm = nn.LayerNorm(embed_dim)

    @staticmethod
    def _hour_sin_cos(abs_hour_f32: torch.Tensor) -> torch.Tensor:
        """abs_hour_f32: (B,T) float32; return (B,T,2) of sin/cos(hour_of_day)."""
        hour_frac = torch.remainder(abs_hour_f32, 24.0) / 24.0  # (B,T)
        sin_t = torch.sin(2 * math.pi * hour_frac)
        cos_t = torch.cos(2 * math.pi * hour_frac)
        return torch.stack([sin_t, cos_t], dim=-1)

    @staticmethod
    def _day_index(abs_hour_f32: torch.Tensor) -> torch.Tensor:
        """(B,T) -> (B,T) long day indices."""
        return torch.floor_divide(abs_hour_f32.long(), 24)

    def forward(self, *, abs_hour: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
            abs_hour: (B,T) absolute hours since some reference (can be float)
        Returns:
            H_hour: (B,T,E)
            H_day:  (B,T,E)
        """
        dtype_in = abs_hour.dtype
        ah = abs_hour.to(torch.float32)

        h2 = self._hour_sin_cos(ah)         # (B,T,2) float32
        H_hour = self.hour_proj(h2)         # (B,T,E) float32

        day_idx = self._day_index(ah).clamp_min(0).clamp_max(self.max_days - 1)
        d_emb = self.day_emb(day_idx)       # (B,T,De)
        H_day = self.day_proj(d_emb)        # (B,T,E)

        if self.rel_pe is not None:
            zeros = torch.zeros_like(H_hour)
            H_hour = self.rel_pe(zeros) + H_hour
            H_day  = self.rel_pe(zeros) + H_day

        H_hour = self.out_norm(H_hour).to(dtype_in)
        H_day  = self.out_norm(H_day).to(dtype_in)
        return H_hour, H_day


# ================================================================
# Multi-Scale FiLM (Hour & Day) with Dual Gating
# ================================================================
class MultiScaleTimeFiLM(nn.Module):
    """
    Two FiLM paths (hour-level and day-level) + dual gating.
    gamma/beta depend on: (time code at step) + pooled baseline vector.

    Forward:
        x_seq: (B,T,E)
        b_vec: (B,E)
        H_hour: (B,T,E)
        H_day:  (B,T,E)
    Output:
        (B,T,E)
    """
    def __init__(self, embed_dim: int, hidden: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden or (2 * embed_dim)
        self.norm_x = nn.LayerNorm(embed_dim)
        self.norm_b = nn.LayerNorm(embed_dim)

        def film_mlp():
            return nn.Sequential(
                nn.Linear(2 * embed_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, 2 * embed_dim)
            )

        self.mlp_hour = film_mlp()
        self.mlp_day  = film_mlp()

        # Dual gate for mixing x and x_hd; include pooled day as context
        self.cross_gate = nn.Sequential(
            nn.Linear(4 * embed_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2 * embed_dim)
        )
        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()   # bound gamma in (-1,1)

    def forward(self, x_seq: torch.Tensor, b_vec: torch.Tensor, H_hour: torch.Tensor, H_day: torch.Tensor) -> torch.Tensor:
        B, T, E = x_seq.shape
        x = self.norm_x(x_seq)
        b = self.norm_b(b_vec)

        b_expand = b.unsqueeze(1).expand(-1, T, -1)
        ghb = torch.cat([H_hour, b_expand], dim=-1)
        gdb = torch.cat([H_day,  b_expand], dim=-1)

        gh, bh = self.mlp_hour(ghb).chunk(2, dim=-1)  # (B,T,E)
        gd, bd = self.mlp_day(gdb).chunk(2, dim=-1)

        # gamma via tanh to avoid over-amplification; FiLM sequentially
        x_h  = (1.0 + self.tanh(gh)) * x + bh
        x_hd = (1.0 + self.tanh(gd)) * x_h + bd

        # Dual gating: produce gate for x and x_hd separately
        pool_x    = x.mean(dim=1)
        pool_hour = H_hour.mean(dim=1)
        pool_day  = H_day.mean(dim=1)
        gate_pair = self.cross_gate(torch.cat([pool_x, pool_hour, pool_day, b], dim=-1))  # (B,2E)
        g_x, g_hd = gate_pair.chunk(2, dim=-1)
        g_x  = self.sigmoid(g_x).unsqueeze(1)
        g_hd = self.sigmoid(g_hd).unsqueeze(1)

        out = g_hd * x_hd + g_x * x
        return out


# ================================================================
# Heads
# ================================================================
class TypeHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits (B,T,C)


class ZeroInflatedLogNormalHead(nn.Module):
    """
    Outputs:
      p0_logit: (B,T,D)  zero-probability logits
      mu:       (B,T,D)  mean of log1p(y)
      log_sigma:(B,T,D)
    """
    def __init__(self, in_dim: int, out_dim: int = 3, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.p0_head   = nn.Linear(hidden, out_dim)  # logits
        self.mu_head   = nn.Linear(hidden, out_dim)
        self.lsg_head  = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        p0_logit  = self.p0_head(h)                       # (B,T,D) logits
        mu        = self.mu_head(h)                       # (B,T,D)
        log_sigma = self.lsg_head(h).clamp(-6.0, 6.0)     # (B,T,D)
        return p0_logit, mu, log_sigma


# ================================================================
# ZILN helpers
# ================================================================
def ziln_predict_mean_from_logits(p0_logit: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    # E[y] = (1 - p0) * E[y | y>0]; p0 = sigmoid(p0_logit)
    p0 = torch.sigmoid(p0_logit)
    sigma2 = torch.exp(2.0 * log_sigma)
    pos_mean = torch.exp(mu + 0.5 * sigma2) - 1.0
    return (1.0 - p0) * pos_mean


def ziln_loss_from_logits(
    p0_logit: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor,
    flow: torch.Tensor, lambda_zero: float = 1.0, lambda_pos: float = 1.0
) -> torch.Tensor:
    """
    ZILN loss = λ0 * BCEWithLogits( 1{flow==0} ; p0_logit ) + λpos * LogNormal NLL on positive samples
    flow: (B,T,D) (mL/h)
    """
    eps = 1e-8
    is_zero = (flow <= 0).float()
    is_pos  = 1.0 - is_zero

    # Zero-part: BCE-with-logits (stable)
    bce = F.binary_cross_entropy_with_logits(p0_logit, is_zero, reduction='mean')

    # Positive-part: log1p(flow) ~ N(mu, sigma^2), ignore constant terms (ok for optimization)
    y = torch.log1p(flow.clamp_min(0.0) + eps)
    sigma = torch.exp(log_sigma)
    nll_all = 0.5 * ((y - mu) / sigma).pow(2) + log_sigma
    pos_count = is_pos.sum().clamp_min(1.0)
    nll = (nll_all * is_pos).sum() / pos_count

    return lambda_zero * bce + lambda_pos * nll


# ================================================================
# Baseline with Multi-Scale Time-FiLM (Hour & Day)
# ================================================================
class BaselineNetwork_MultiTimeFiLM_ZILN(nn.Module):
    def __init__(self,
                 input_dim_base: int,
                 input_dim_temporal: int,
                 embed_dim: int,
                 num_heads: int,
                 hidden_dim: int,
                 num_classes: int,
                 pred_length: int,
                 num_temporal_layers: int = 2,
                 out_dim_regression: int = 3,
                 dropout: float = 0.1,
                 # time enc params
                 max_days: int = 3,
                 add_rel_pe: bool = True,
                 # dynamic refiner depth
                 refine_layers: int = 1,
                 # sampling stride in hours for your data (0.1h if every 6 min)
                 step_hours: float = 0.1,
                 # temporal causal attention
                 causal_temporal: bool = False):
        super().__init__()
        self.pred_length = pred_length
        self.step_hours = float(step_hours)

        # Encoders
        self.base_enc = BaselineEncoder(input_dim_base, embed_dim, num_heads, hidden_dim, dropout)
        self.temp_enc = TemporalEncoder(input_dim_temporal, embed_dim, num_heads, hidden_dim,
                                        num_layers=num_temporal_layers, dropout=dropout, causal=causal_temporal)

        # Time enc + Multi-Scale FiLM
        self.time_enc = HourDayTimeEncoder(embed_dim=embed_dim, max_days=max_days, add_rel_pe=add_rel_pe)
        self.ms_film = MultiScaleTimeFiLM(embed_dim, hidden=2 * embed_dim, dropout=dropout)

        # Optional dynamic inter-layer modulation
        self.refiner = nn.ModuleList([
            nn.ModuleList([MHABlock(embed_dim, num_heads, dropout), FeedForward(embed_dim, hidden_dim, dropout)])
        ]) if refine_layers > 0 else None
        self.refine_layers = refine_layers

        # Heads
        self.type_head = TypeHead(in_dim=embed_dim, num_classes=num_classes, hidden=hidden_dim, dropout=dropout)
        self.ziln_head = ZeroInflatedLogNormalHead(in_dim=embed_dim, out_dim=out_dim_regression,
                                                   hidden=hidden_dim, dropout=dropout)

    def _build_abs_hours(self, B: int, T: int, start_hour: Optional[torch.Tensor], device, dtype) -> torch.Tensor:
        """
        Build absolute hour grid for each batch item.
        start_hour: (B,) or None. If None, assume 0 for all.
        Returns abs_hour (B,T) where abs_hour[b,t] = start_hour[b] + t*step_hours
        """
        if start_hour is None:
            s = torch.zeros(B, device=device, dtype=dtype)
        else:
            s = start_hour.to(device=device, dtype=dtype).view(-1)
            if s.numel() == 1 and B > 1:
                s = s.expand(B)
        t_idx = torch.arange(T, device=device, dtype=dtype).unsqueeze(0).expand(B, -1)
        return s.unsqueeze(1) + t_idx * self.step_hours

    def forward(self,
                y_seq: torch.Tensor,
                x_base: torch.Tensor,
                *,
                start_hour: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            y_seq: (B,T,D_temporal) input history window
            x_base: (B,D_base) static/context features
            start_hour: optional (B,) tensor; absolute hour of the first step in y_seq
        Returns:
            flow_hat: (B,P,D_reg)
            type_logits: (B,P,C)
            extras: dict with 'p0_logit','mu','log_sigma'
        """
        B, T, _ = y_seq.shape

        # Encoders
        b_feat = self.base_enc(x_base)                # (B,E)
        t_feat = self.temp_enc(y_seq)                 # (B,T,E)

        # Keep only last P steps to predict
        t_feat = t_feat[:, -self.pred_length:, :]     # (B,P,E)
        P = t_feat.size(1)

        # Build absolute hours for those P steps
        abs_hours_full = self._build_abs_hours(B, T, start_hour, device=y_seq.device, dtype=y_seq.dtype)
        abs_hours_tail = abs_hours_full[:, -P:]       # (B,P)

        # Time encodings
        H_hour, H_day = self.time_enc(abs_hour=abs_hours_tail)  # (B,P,E), (B,P,E)

        # Multi-Scale FiLM fusion
        fused = self.ms_film(t_feat, b_feat, H_hour, H_day)     # (B,P,E)

        # Optional small refinement
        if self.refiner is not None:
            for attn, ffn in self.refiner:
                fused = attn(fused)
                fused = ffn(fused)

        # Heads
        type_logits = self.type_head(fused)                     # (B,P,C)
        p0_logit, mu, log_sigma = self.ziln_head(fused)         # (B,P,D)
        flow_hat = ziln_predict_mean_from_logits(p0_logit, mu, log_sigma)
        extras = { 'p0_logit': p0_logit, 'mu': mu, 'log_sigma': log_sigma }
        return flow_hat, type_logits, extras


# ================================================================
# Builder (keeps your original signature; extra params have defaults)
# ================================================================
def build_model_film_ziln(input_dim_base: int,
                          input_dim_temporal: int,
                          embed_dim: int,
                          num_heads: int,
                          hidden_dim: int,
                          num_classes: int,
                          pred_length: int,
                          out_dim_regression: int = 3,
                          num_temporal_layers: int = 2,
                          dropout: float = 0.1,
                          # === new but with safe defaults ===
                          max_days: int = 3,
                          add_rel_pe: bool = True,
                          refine_layers: int = 1,
                          step_hours: float = 0.1,
                          causal_temporal: bool = False) -> nn.Module:
    """
    Notes:
      - step_hours=0.1 对应 6分钟/步；如果你的步长是 1 小时，设 1.0。
      - causal_temporal=True 时，MHA 不看未来，训练更稳妥（尤其是有 H+P 的设置）。
    """
    return BaselineNetwork_MultiTimeFiLM_ZILN(
        input_dim_base=input_dim_base,
        input_dim_temporal=input_dim_temporal,
        embed_dim=embed_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        pred_length=pred_length,
        num_temporal_layers=num_temporal_layers,
        out_dim_regression=out_dim_regression,
        dropout=dropout,
        max_days=max_days,
        add_rel_pe=add_rel_pe,
        refine_layers=refine_layers,
        step_hours=step_hours,
        causal_temporal=causal_temporal,
    )
