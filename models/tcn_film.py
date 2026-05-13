# -*- coding: utf-8 -*-
# File: tcn_film_ablation_models.py
"""
Hybrid backbone: Dual-Path Dilated TCN × FiLM (baseline fusion) × Causal Transformer
Outputs (runner-aligned):
  forward(input_T, input_B) -> {
      "class_logits": (B,P,3),
      "flow_pred":    (B,P,3),                 # E[rate] from ZILN
      "extras": {"p0_logit":..., "mu":..., "log_sigma":...}
  }
Notes
- input_T: (B, H, D_temporal)  # history concatenated with historical labels
- input_B: (B, D_base)
- Model keeps the last P steps for prediction, controlled by pred_length.
- Causal Transformer uses an upper-triangular mask to prevent future leakage.
- Scale via: embed_dim, n_transformer_layers, n_heads, TCN depth.
- Builders at bottom map 1:1 to your ablation table; names use tcnfilm_* prefix.
"""
from typing import Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# Utils: Causal mask & Positional encoding
# --------------------------
def build_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular mask (True blocks attention) for nn.MultiheadAttention."""
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
    return mask


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:T, :].unsqueeze(0)


# --------------------------
# Base blocks
# --------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[..., :-self.chomp_size] if self.chomp_size > 0 else x


class CausalConv1d(nn.Module):
    def __init__(self, c_in, c_out, k, d, groups=None, causal=True):
        super().__init__()
        pad = (k - 1) * d
        self.causal = causal
        self.pad = pad if causal else (k // 2) * d
        g = groups if groups is not None else c_in
        self.dw = nn.Conv1d(c_in, c_in, k, padding=self.pad, dilation=d, groups=g)
        self.chomp = Chomp1d(self.pad if causal else 0)
        self.pw = nn.Conv1d(c_in, c_out, 1)

    def forward(self, x):  # x: (B,T,C_in)
        x = x.transpose(1, 2)
        y = self.dw(x)
        y = self.chomp(y).transpose(1, 2)
        y = self.pw(y.transpose(1, 2)).transpose(1, 2)
        return y


class SEFiLM(nn.Module):
    def __init__(self, c, hidden=128, film_hidden=256, dropout=0.1):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(c, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, c), nn.Sigmoid()
        )
        self.film = nn.Sequential(
            nn.Linear(2 * c, film_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(film_hidden, 2 * c)
        )
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, b: torch.Tensor):
        # x: (B,T,C), b: (B,C)
        g = self.se(b)                      # (B,C)
        x_ = x * g.unsqueeze(1)             # SE gate
        hb = self.film(torch.cat([x_.mean(1), b], dim=-1))  # (B,2C)
        gamma, beta = hb.chunk(2, dim=-1)
        gamma = 1.0 + self.tanh(gamma)
        return gamma.unsqueeze(1) * x_ + beta.unsqueeze(1)


class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k, d, causal=True, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(c_in, c_out, k, d, causal=causal)
        self.norm1 = nn.LayerNorm(c_out)
        self.conv2 = CausalConv1d(c_out, c_out, k, d, causal=causal)
        self.norm2 = nn.LayerNorm(c_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(c_in, c_out) if c_in != c_out else nn.Identity()

        def forward(self, x):
            y = self.conv1(x)
            y = self.norm1(y)
            y = self.act(y)
            y = self.drop(y)
            y = self.conv2(y)
            y = self.norm2(y)
            return self.act(y + self.proj(x))

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.norm2(y)
        return self.act(y + self.proj(x))


class DualPathTCN(nn.Module):
    def __init__(self, c_in, c, layers_short=3, layers_long=5, k=3, causal=True, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(c_in, c)
        self.short = nn.ModuleList(
            [TCNBlock(c if i > 0 else c, c, k=k, d=2 ** i, causal=causal, dropout=dropout) for i in range(layers_short)]
        )
        self.long = nn.ModuleList(
            [TCNBlock(c if i > 0 else c, c, k=k, d=2 ** (i + 1), causal=causal, dropout=dropout) for i in range(layers_long)]
        )
        self.fuse = nn.Linear(2 * c, c)
        self.norm = nn.LayerNorm(c)

    def forward(self, x):
        h = self.inp(x)
        hs = h
        for blk in self.short:
            hs = blk(hs)
        hl = h
        for blk in self.long:
            hl = blk(hl)
        z = torch.cat([hs, hl], dim=-1)
        z = self.norm(self.fuse(z))
        return z


# --------------------------
# Transformer blocks (causal, pre-norm)
# --------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.ln1(x)
        y, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + self.drop(y)
        h = self.ln2(x)
        z = self.ff(h)
        return x + self.drop(z)


class FFOnlyBlock(nn.Module):
    """Transformer-like block without self-attention (for w/o self-attention ablation)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.ln(x)
        z = self.ff(h)
        return x + self.drop(z)


# --------------------------
# Heads
# --------------------------
class TypeHead(nn.Module):
    def __init__(self, c, num_classes, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(c),
            nn.Linear(c, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class ZILNHead(nn.Module):
    def __init__(self, c, out_dim=3, hidden=256, dropout=0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(c),
            nn.Linear(c, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.p0 = nn.Linear(hidden, out_dim)   # logits for zero inflation
        self.mu = nn.Linear(hidden, out_dim)   # mean in log-space
        self.ls = nn.Linear(hidden, out_dim)   # log sigma

    def forward(self, x):
        h = self.trunk(x)
        p0_logit = self.p0(h)
        mu = self.mu(h)
        log_sigma = self.ls(h).clamp(-6.0, 6.0)
        return p0_logit, mu, log_sigma


def ziln_predict_mean(p0_logit, mu, log_sigma):
    p0 = torch.sigmoid(p0_logit)
    sigma2 = torch.exp(2.0 * log_sigma)
    pos_mean = torch.exp(mu + 0.5 * sigma2) - 1.0
    return (1.0 - p0) * pos_mean


# --------------------------
# Full model with feature flags
# --------------------------
class Baseline_TCN_FiLM_ZILN_Transformer(nn.Module):
    def __init__(
        self,
        input_dim_base: int,
        input_dim_temporal: int,
        embed_dim: int = 256,
        num_classes: int = 3,
        pred_length: int = 1,
        tcn_k: int = 3,
        tcn_layers_short: int = 3,
        tcn_layers_long: int = 5,
        n_transformer_layers: int = 4,
        n_heads: int = 8,
        ff_mult: int = 4,
        causal: bool = True,
        dropout: float = 0.1,
        # ablation flags
        use_tcn_backbone: bool = True,
        use_temporal_transformer: bool = True,
        use_self_attention: bool = True,
        use_baseline_encoder: bool = True,
        use_film: bool = True,
        use_refine_tcn: bool = True,
    ):
        super().__init__()
        self.pred_length = int(pred_length)
        self.embed_dim = int(embed_dim)
        self.causal = causal
        self.use_tcn_backbone = use_tcn_backbone
        self.use_temporal_transformer = use_temporal_transformer
        self.use_self_attention = use_self_attention
        self.use_baseline_encoder = use_baseline_encoder
        self.use_film = use_film
        self.use_refine_tcn = use_refine_tcn

        # baseline encoder
        if use_baseline_encoder:
            self.base_proj = nn.Sequential(
                nn.Linear(input_dim_base, embed_dim), nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim)
            )
        else:
            self.base_proj = None

        # temporal backbone
        if use_tcn_backbone:
            self.backbone = DualPathTCN(
                c_in=input_dim_temporal, c=embed_dim,
                layers_short=tcn_layers_short, layers_long=tcn_layers_long,
                k=tcn_k, causal=causal, dropout=dropout
            )
        else:
            self.backbone = nn.Linear(input_dim_temporal, embed_dim)  # light per-step projection

        # baseline fusion
        if use_film:
            self.sefilm = SEFiLM(embed_dim, hidden=embed_dim // 2, film_hidden=embed_dim, dropout=dropout)
        else:
            self.sefilm = None

        # positional encoding + transformer stack (or FF-only blocks)
        if use_temporal_transformer and n_transformer_layers > 0:
            self.posenc = SinusoidalPositionalEncoding(embed_dim)
            if use_self_attention:
                # TransformerBlock(d_model, n_heads, d_ff, dropout)
                self.tr_blocks = nn.ModuleList([
                    TransformerBlock(embed_dim, n_heads, ff_mult * embed_dim, dropout)
                    for _ in range(n_transformer_layers)
                ])
            else:
                # FFOnlyBlock(d_model, d_ff, dropout)
                self.tr_blocks = nn.ModuleList([
                    FFOnlyBlock(embed_dim, ff_mult * embed_dim, dropout)
                    for _ in range(n_transformer_layers)
                ])
        else:
            self.posenc = None
            self.tr_blocks = nn.ModuleList([])

        # lightweight refine
        self.refine = (
            TCNBlock(embed_dim, embed_dim, k=3, d=1, causal=causal, dropout=dropout)
            if use_refine_tcn else nn.Identity()
        )

        # heads
        self.cls_head = TypeHead(embed_dim, num_classes, hidden=embed_dim, dropout=dropout)
        self.ziln_head = ZILNHead(embed_dim, out_dim=num_classes, hidden=embed_dim, dropout=dropout)

        # gated fusion of pre vs post transformer (only when transformer exists)
        if use_temporal_transformer and n_transformer_layers > 0:
            self.fuse_gate = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim), nn.GELU(),
                nn.Linear(embed_dim, 2), nn.Softmax(dim=-1)
            )
        else:
            self.fuse_gate = None

    def forward(self, input_T: torch.Tensor, input_B: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        input_T: (B, H, D_temporal)
        input_B: (B, D_base)
        """
        B, H, _ = input_T.shape
        device = input_T.device

        # temporal encoding over full history
        h = self.backbone(input_T)               # (B,H,E)

        # keep last P steps for prediction
        P = self.pred_length
        h_last = h[:, -P:, :]                    # (B,P,E)

        # baseline features
        if self.base_proj is not None and input_B is not None:
            b = self.base_proj(input_B)          # (B,E)
        else:
            b = torch.zeros(B, self.embed_dim, device=device)

        # fusion
        z = self.sefilm(h_last, b) if self.sefilm is not None else h_last

        # transformer/temporal encoder
        if len(self.tr_blocks) > 0:
            pre_tr = self.posenc(z) if self.posenc is not None else z
            x = pre_tr
            attn_mask = build_causal_mask(P, device) if (self.causal and self.use_self_attention) else None
            for blk in self.tr_blocks:
                x = blk(x, attn_mask=attn_mask)
            post_tr = x
            if self.fuse_gate is not None:
                w = self.fuse_gate(pre_tr.mean(dim=1))   # (B,2)
                w1, w2 = w[:, 0].unsqueeze(1).unsqueeze(2), w[:, 1].unsqueeze(1).unsqueeze(2)
                z = w1 * pre_tr + w2 * post_tr          # (B,P,E)
            else:
                z = post_tr

        # optional refine
        z = self.refine(z)

        # heads
        class_logits = self.cls_head(z)          # (B,P,C)
        p0_logit, mu, log_sigma = self.ziln_head(z)
        flow_pred = ziln_predict_mean(p0_logit, mu, log_sigma)  # (B,P,C)

        return {
            "class_logits": class_logits,
            "flow_pred": flow_pred,
            "extras": {"p0_logit": p0_logit, "mu": mu, "log_sigma": log_sigma}
        }


# --------------------------
# Builders (tcnfilm_* naming)
# --------------------------
def _common_kwargs(**kw):
    return dict(
        input_dim_base=kw.get('input_dim_base'),
        input_dim_temporal=kw.get('input_dim_temporal'),
        embed_dim=kw.get('embed_dim', 256),
        num_classes=kw.get('num_classes', 3),
        pred_length=kw.get('pred_length', 1),
        tcn_k=kw.get('tcn_k', 3),
        tcn_layers_short=kw.get('tcn_layers_short', 3),
        tcn_layers_long=kw.get('tcn_layers_long', 5),
        n_transformer_layers=kw.get('n_transformer_layers', 4),
        n_heads=kw.get('n_heads', 8),
        ff_mult=kw.get('ff_mult', 4),
        causal=kw.get('causal', True),
        dropout=kw.get('dropout', 0.1),
    )


def build_tcnfilm_full(**kw) -> nn.Module:
    return Baseline_TCN_FiLM_ZILN_Transformer(
        **_common_kwargs(**kw),
        use_tcn_backbone=True,
        use_temporal_transformer=True,
        use_self_attention=True,
        use_baseline_encoder=True,
        use_film=True,
        use_refine_tcn=True,
    )


def build_tcnfilm_wo_temporal(**kw) -> nn.Module:
    common = _common_kwargs(**kw)
    # 这里覆盖为 0，且不要再在调用 Baseline... 时重复传入相同参数名
    common["n_transformer_layers"] = 0

    return Baseline_TCN_FiLM_ZILN_Transformer(
        **common,
        use_tcn_backbone=True,
        use_temporal_transformer=False,
        use_self_attention=False,
        use_baseline_encoder=True,
        use_film=True,
        use_refine_tcn=True,
    )

def build_tcnfilm_wo_baseline(**kw) -> nn.Module:
    return Baseline_TCN_FiLM_ZILN_Transformer(
        **_common_kwargs(**kw),
        use_tcn_backbone=True,
        use_temporal_transformer=True,
        use_self_attention=True,
        use_baseline_encoder=False,
        use_film=False,  # no FiLM without baseline
        use_refine_tcn=True,
    )


def build_tcnfilm_wo_selfattn(**kw) -> nn.Module:
    return Baseline_TCN_FiLM_ZILN_Transformer(
        **_common_kwargs(**kw),
        use_tcn_backbone=True,
        use_temporal_transformer=True,
        use_self_attention=False,
        use_baseline_encoder=True,
        use_film=True,
        use_refine_tcn=True,
    )


def build_tcnfilm_wo_tcn(**kw) -> nn.Module:
    return Baseline_TCN_FiLM_ZILN_Transformer(
        **_common_kwargs(**kw),
        use_tcn_backbone=False,
        use_temporal_transformer=True,
        use_self_attention=True,
        use_baseline_encoder=True,
        use_film=True,
        use_refine_tcn=True,
    )


def build_tcnfilm_wo_film(**kw) -> nn.Module:
    return Baseline_TCN_FiLM_ZILN_Transformer(
        **_common_kwargs(**kw),
        use_tcn_backbone=True,
        use_temporal_transformer=True,
        use_self_attention=True,
        use_baseline_encoder=True,
        use_film=False,
        use_refine_tcn=True,
    )


# Compatibility shim: old entry kept for your existing runner
def build_model_tcn_film_ziln(
    input_dim_base: int,
    input_dim_temporal: int,
    embed_dim: int,
    num_heads: int,             # placeholder for compatibility
    hidden_dim: int,            # placeholder for compatibility
    num_classes: int,
    pred_length: int,
    out_dim_regression: int = 3,   # placeholder
    num_temporal_layers: int = 2,  # placeholder
    dropout: float = 0.1,
    tcn_k: int = 3,
    tcn_layers_short: int = 3,
    tcn_layers_long: int = 5,
    n_transformer_layers: int = 4,
    ff_mult: int = 4,
    causal: bool = True,
) -> nn.Module:
    return build_tcnfilm_full(
        input_dim_base=input_dim_base,
        input_dim_temporal=input_dim_temporal,
        embed_dim=embed_dim,
        num_classes=num_classes,
        pred_length=pred_length,
        tcn_k=tcn_k,
        tcn_layers_short=tcn_layers_short,
        tcn_layers_long=tcn_layers_long,
        n_transformer_layers=n_transformer_layers,
        n_heads=num_heads,
        ff_mult=ff_mult,
        causal=causal,
        dropout=dropout,
    )


# --------------------------
# Utilities
# --------------------------
def count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


if __name__ == "__main__":
    # Simple self-check
    B, H, D_T, D_B, P = 2, 16, 32, 18, 1
    xT = torch.randn(B, H, D_T)
    xB = torch.randn(B, D_B)
    builders = [
        ("tcnfilm_full", build_tcnfilm_full),
        ("tcnfilm_wo_temporal", build_tcnfilm_wo_temporal),
        ("tcnfilm_wo_baseline", build_tcnfilm_wo_baseline),
        ("tcnfilm_wo_selfattn", build_tcnfilm_wo_selfattn),
        ("tcnfilm_wo_tcn", build_tcnfilm_wo_tcn),
        ("tcnfilm_wo_film", build_tcnfilm_wo_film),
    ]
    for name, fn in builders:
        m = fn(
            input_dim_base=D_B, input_dim_temporal=D_T, embed_dim=128, num_classes=3, pred_length=P,
            n_transformer_layers=2, n_heads=4, tcn_layers_short=2, tcn_layers_long=2
        )
        with torch.no_grad():
            out = m(xT, xB)
        print(f"{name:>20s}  params={count_params_m(m):.2f}M  logits={out['class_logits'].shape}  flow={out['flow_pred'].shape}")
