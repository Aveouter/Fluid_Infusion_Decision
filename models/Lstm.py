import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Baseline encoder: x -> (B, E)
# ------------------------------------------------------------
class BaselineEncoderMLP(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        # x: (B, Dx)
        z = self.net(x)
        return self.norm(z)  # (B, E)

# ------------------------------------------------------------
# Temporal LSTM encoder: y (B, T, Dy) -> (B, T, E)
# ------------------------------------------------------------
class TemporalLSTMEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 embed_dim: int,
                 rnn_hidden: int,
                 num_layers: int = 2,
                 bidirectional: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, rnn_hidden)
        self.lstm = nn.LSTM(
            input_size=rnn_hidden,
            hidden_size=rnn_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = rnn_hidden * (2 if bidirectional else 1)
        self.out_proj = nn.Linear(out_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, y: torch.Tensor):
        # y: (B, T, Dy)
        h = self.in_proj(y)          # (B, T, H)
        h, _ = self.lstm(h)          # (B, T, H*(1|2))
        h = self.out_proj(h)         # (B, T, E)
        return self.norm(h)

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
        x = enc_x.unsqueeze(1)  # (B,1,E)
        g = self.gamma(x)
        b = self.beta(x)
        return enc_y_p * (1 + g) + b

# ------------------------------------------------------------
# Decoder heads on (B, P, E):
#   - Regression: flow (>=0) via Softplus, shape (B,P,C)
#   - Classification: logits (B,P,C)
# ------------------------------------------------------------
class LSTMDecoderHeads(nn.Module):
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
        flow   = F.softplus(self.flow_head(h))  # (B, P, C)
        logits = self.cls_head(h)               # (B, P, C)
        return flow, logits

# ------------------------------------------------------------
# LSTM Baseline Network（输入/输出与 CNN/Transformer 版本一致）
# 输入：
#   y: (B, T_y, Dy)
#   x: (B, Dx)
# 输出：
#   flow_pred:   (B, P, C)
#   class_logits:(B, P, C)
# ------------------------------------------------------------
class LSTMBaselineNetwork(nn.Module):
    def __init__(self,
                 input_dim_base: int,
                 input_dim_temporal: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 history_length: int,
                 pred_length: int,
                 rnn_hidden: int | None = None,
                 num_layers: int = 2,
                 bidirectional: bool = False,
                 dropout: float = 0.1,
                 use_film: bool = True):
        super().__init__()
        self.pred_length    = pred_length
        self.history_length = history_length
        self.use_film       = use_film
        rnn_hidden = hidden_dim if rnn_hidden is None else rnn_hidden

        self.baseline_enc = BaselineEncoderMLP(
            input_dim=input_dim_base,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.temporal_enc = TemporalLSTMEncoder(
            input_dim=input_dim_temporal,
            embed_dim=embed_dim,
            rnn_hidden=rnn_hidden,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        if use_film:
            self.fuser = FiLM(embed_dim)
        else:
            self.fuser = nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
            )
        self.heads = LSTMDecoderHeads(embed_dim=embed_dim, hidden_dim=hidden_dim,
                                      num_classes=num_classes, dropout=dropout)

    def forward(self, y: torch.Tensor, x: torch.Tensor):
        # y: (B, T_y, Dy), x: (B, Dx)
        enc_x = self.baseline_enc(x)                 # (B, E)
        enc_y = self.temporal_enc(y)                 # (B, T_y, E)
        enc_yP = enc_y[:, -self.pred_length:, :]     # (B, P, E)
        if self.use_film:
            z = self.fuser(enc_yP, enc_x)            # (B, P, E)
        else:
            enc_x_rep = enc_x.unsqueeze(1).expand(-1, self.pred_length, -1)
            z = torch.cat([enc_yP, enc_x_rep], dim=-1)
            z = self.fuser(z)                        # (B, P, E)
        flow_pred, class_logits = self.heads(z)      # (B,P,C), (B,P,C)
        return flow_pred,class_logits


# ------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------
if __name__ == '__main__':
    B, Ty, Dy = 8, 32, 50
    Dx = 16
    P = 8
    E = 128
    H = 256
    C = 5

    model = LSTMBaselineNetwork(
        input_dim_base=Dx,
        input_dim_temporal=Dy,
        embed_dim=E,
        hidden_dim=H,
        num_classes=C,
        history_length=Ty,
        pred_length=P,
        rnn_hidden=None,         # 默认用 hidden_dim
        num_layers=2,
        bidirectional=True,      # 可切换单/双向
        dropout=0.1,
        use_film=True,
    )
    y = torch.randn(B, Ty, Dy)
    x = torch.randn(B, Dx)
    out = model(y, x)
    print(out['flow_pred'].shape, out['class_logits'].shape)
    # torch.Size([8, 8, 5]) torch.Size([8, 8, 5])
