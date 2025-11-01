import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPositionalEncoding(nn.Module):
    """
    正弦位置编码（支持 batch_first 开关）
    pe 形状：(max_len, E)
    """
    def __init__(self, embed_dim: int, max_len: int = 5000, batch_first: bool = False):
        super().__init__()
        self.batch_first = batch_first

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)          # (max_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float()
                             * (-math.log(10000.0) / embed_dim))                    # (E/2,)

        pe = torch.zeros(max_len, embed_dim)                                        # (max_len, E)
        pe[:, 0::2] = torch.sin(position * div_term)                                # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)                                # 奇数维
        self.register_buffer("pe", pe)  # 不参与训练 & 自动随设备移动

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (T, B, E) 若 batch_first=False；或 (B, T, E) 若 batch_first=True
        """
        if self.batch_first:
            T = x.size(1)
            return x + self.pe[:T, :].unsqueeze(0)   # (1, T, E) 广播到 (B, T, E)
        else:
            T = x.size(0)
            return x + self.pe[:T, :].unsqueeze(1)   # (T, 1, E) 广播到 (T, B, E)


class MultivariateAttentionBlock(nn.Module):
    """
    多头自注意力的 pre-norm 残差块：
    y = x + Dropout(MHA(LN(x)))
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, batch_first: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # 自注意力
        return x + self.drop(attn_out)


class TemporalEncoderBlock(nn.Module):
    """
    时间编码器（pre-norm Transformer encoder 风格）：
    x = Linear(in->E)
    x = PositionalEncoding(x)
    x = x + MHA(LN(x))            # pre-norm attention
    x = x + FFN(LN(x))            # pre-norm FFN
    """
    def __init__(self,
                 input_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 hidden_dim: int,
                 dropout: float = 0.1,
                 max_len: int = 5000,
                 batch_first: bool = False):
        super().__init__()
        self.batch_first = batch_first

        # 输入投影到 E
        self.embedding = nn.Linear(input_dim, embed_dim)

        # 位置编码（与输入布局保持一致）
        self.positional_encoding = TemporalPositionalEncoding(embed_dim, max_len=max_len, batch_first=batch_first)

        # 注意力块（内部已 pre-norm）
        self.multi_attention = MultivariateAttentionBlock(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)

        # FFN（pre-norm + 残差）
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (T, B, input_dim) 若 batch_first=False；或 (B, T, input_dim) 若 batch_first=True
        返回与输入布局一致的 (.., E)
        """
        # 投影到 E
        x = self.embedding(x)                         # (..., E)

        # 加位置编码
        x = self.positional_encoding(x)               # (..., E)

        # 注意力（pre-norm 在 MultivariateAttentionBlock 内部做）
        x = self.multi_attention(x)                   # (..., E)

        # FFN（pre-norm）
        x = x + self.feed_forward(self.ffn_norm(x))   # (..., E)
        return x
