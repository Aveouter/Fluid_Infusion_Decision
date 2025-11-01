import math
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
# Sinusoidal positional encoding (batch_first)
# ------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe, persistent=False)
    def forward(self, x):
        # x: (B, T, E)
        T = x.size(1)
        return x + self.pe[:, :T, :]

# ------------------------------------------------------------
# Temporal Transformer encoder: y (B, T, Dy) -> (B, T, E)
#   - Optional causal mask to avoid leakage
# ------------------------------------------------------------
class TemporalTransformerEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 dim_ff: int,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 causal: bool = True):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                               nhead=num_heads,
                                               dim_feedforward=dim_ff,
                                               dropout=dropout,
                                               batch_first=True,
                                               activation='gelu')
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = SinusoidalPositionalEncoding(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.causal = causal
    @staticmethod
    def _generate_subsequent_mask(T: int, device: torch.device):
        # mask: (T, T), True/inf on positions to mask (upper triangular)
        # nn.Transformer expects float mask with -inf for masked, 0 for keep
        mask = torch.full((T, T), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask
    def forward(self, y: torch.Tensor):
        # y: (B, T, Dy)
        B, T, _ = y.shape
        z = self.in_proj(y)          # (B, T, E)
        z = self.pos(z)              # add sinusoidal PE
        src_mask = None
        if self.causal:
            src_mask = self._generate_subsequent_mask(T, y.device)
        z = self.encoder(z, mask=src_mask)  # (B, T, E)
        return self.norm(z)

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
class TransformerDecoderHeads(nn.Module):
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
# Transformer Baseline Network（输入/输出与 CNN 版本一致）
# 输入：
#   y: (B, T_y, Dy)
#   x: (B, Dx)
# 输出：
#   flow_pred:   (B, P, C)
#   class_logits:(B, P, C)
# ------------------------------------------------------------
class TransformerBaselineNetwork(nn.Module):
    def __init__(self,
                 input_dim_base: int,
                 input_dim_temporal: int,
                 embed_dim: int,
                 num_heads: int,
                 dim_ff: int,
                 hidden_dim: int,
                 num_classes: int,
                 history_length: int,
                 pred_length: int,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 causal: bool = True,
                 use_film: bool = True):
        super().__init__()
        self.pred_length    = pred_length
        self.history_length = history_length
        self.use_film       = use_film

        self.baseline_enc = BaselineEncoderMLP(
            input_dim=input_dim_base,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.temporal_enc = TemporalTransformerEncoder(
            input_dim=input_dim_temporal,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dim_ff=dim_ff,
            num_layers=num_layers,
            dropout=dropout,
            causal=causal,
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
        self.heads = TransformerDecoderHeads(embed_dim=embed_dim, hidden_dim=hidden_dim,
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

    model = TransformerBaselineNetwork(
        input_dim_base=Dx,
        input_dim_temporal=Dy,
        embed_dim=E,
        num_heads=4,
        dim_ff=512,
        hidden_dim=H,
        num_classes=C,
        history_length=Ty,
        pred_length=P,
        num_layers=4,
        dropout=0.1,
        causal=True,
        use_film=True,
    )
    y = torch.randn(B, Ty, Dy)
    x = torch.randn(B, Dx)
    out = model(y, x)
    print(out['flow_pred'].shape, out['class_logits'].shape)
    # torch.Size([8, 8, 5]) torch.Size([8, 8, 5])
