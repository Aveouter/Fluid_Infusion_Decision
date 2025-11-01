import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return x + self.fn(x)

# ------------------------------------------------------------
# Baseline encoder: x -> (B, E)
# ------------------------------------------------------------
class BaselineEncoderMLP(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        # x: (B, D)
        x = self.net(x)
        x = self.norm(x)
        return x  # (B, E)

# ------------------------------------------------------------
# Temporal CNN encoder: y (B, T, Dy) -> (B, T, E)
#  - Conv1d over time (channels = features)
#  - Dilated residual stacks for multi-scale receptive field
# ------------------------------------------------------------
class TemporalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.GELU(),
        )
        self.norm = nn.BatchNorm1d(channels)
    def forward(self, x):
        # x: (B, C, T)
        y = self.block(x)
        y = self.norm(y)
        return x + y

class TemporalCNNEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            TemporalConvBlock(hidden_dim, kernel_size=3, dilation=2**i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
    def forward(self, y):
        # y: (B, T, Dy)
        B, T, Dy = y.shape
        h = self.in_proj(y)                  # (B, T, H)
        h = h.transpose(1, 2)                # (B, H, T)
        for blk in self.blocks:
            h = blk(h)                       # (B, H, T)
        h = h.transpose(1, 2)                # (B, T, H)
        h = self.out_proj(h)                 # (B, T, E)
        h = self.ln(h)
        return h

# ------------------------------------------------------------
# Decoder heads on (B, P, E):
#   - Regression: flow (>=0) via Softplus
#   - Classification: num_classes logits
# 可按需增减任务头
# ------------------------------------------------------------
class CNNDecoderHeads(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.flow_head = nn.Linear(hidden_dim, num_classes)
        self.cls_head  = nn.Linear(hidden_dim, num_classes)
    def forward(self, z):
        # z: (B, P, E)
        h = self.proj(z)
        flow = F.softplus(self.flow_head(h))      # (B, P, 1)
        logits = self.cls_head(h)                 # (B, P, C)
        return flow, logits

# ------------------------------------------------------------
# FiLM-style baseline conditioning:
#   enc_y_p: (B, P, E)
#   enc_x:   (B, E)
# 输出 (B, P, E)
# ------------------------------------------------------------
class FiLM(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gamma = nn.Linear(embed_dim, embed_dim)
        self.beta  = nn.Linear(embed_dim, embed_dim)
    def forward(self, enc_y_p, enc_x):
        # enc_x -> (B, 1, E)
        x = enc_x.unsqueeze(1)
        g = self.gamma(x)
        b = self.beta(x)
        return enc_y_p * (1 + g) + b

# ------------------------------------------------------------
# CNN Baseline Network (替代你给的 Transformer 版本)
# 输入：
#   y: (B, T_y, Dy)
#   x: (B, Dx)
# 流程：
#   enc_x = BaselineEncoderMLP(x)           -> (B,E)
#   enc_y = TemporalCNNEncoder(y)           -> (B,T_y,E)
#   enc_yP = enc_y[:, -P:, :]               -> (B,P,E)
#   z = FiLM(enc_yP, enc_x)  或  cat+Linear  -> (B,P,E)
#   heads 输出 flow_pred, class_logits
# ------------------------------------------------------------
class CNNBaselineNetwork(nn.Module):
    def __init__(self,
                 input_dim_base: int,
                 input_dim_temporal: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 history_length: int,
                 pred_length: int,
                 num_cnn_layers: int = 4,
                 dropout: float = 0.1,
                 use_film: bool = True):
        super().__init__()
        self.pred_length   = pred_length
        self.history_length = history_length
        self.use_film = use_film

        self.baseline_enc = BaselineEncoderMLP(
            input_dim=input_dim_base,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.temporal_enc = TemporalCNNEncoder(
            input_dim=input_dim_temporal,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_cnn_layers,
            dropout=dropout,
        )
        if use_film:
            self.fuser = FiLM(embed_dim)
        else:
            self.fuser = nn.Sequential(
                nn.LazyLinear(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim),
            )
        self.heads = CNNDecoderHeads(embed_dim=embed_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)

    def forward(self, y: torch.Tensor, x: torch.Tensor):
        # y: (B, T_y, Dy), x: (B, Dx)
        B, Ty, Dy = y.shape
        enc_x = self.baseline_enc(x)                  # (B, E)
        enc_y = self.temporal_enc(y)                  # (B, T_y, E)
        enc_yP = enc_y[:, -self.pred_length:, :]      # (B, P, E)
        if self.use_film:
            z = self.fuser(enc_yP, enc_x)             # (B, P, E)
        else:
            enc_x_rep = enc_x.unsqueeze(1).expand(-1, self.pred_length, -1)
            z = torch.cat([enc_yP, enc_x_rep], dim=-1)  # (B, P, 2E)
            z = self.fuser(z)                         # -> (B, P, E)
        flow_pred, class_logits = self.heads(z)       # (B,P,1), (B,P,C)
        return flow_pred,class_logits   # 分类 logits
            # 'features': z                     # 按需返回中间特征


# ------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------
