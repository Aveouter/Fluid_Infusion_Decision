import torch
import torch.nn as nn
import math


class TemporalPositionalEncoding(nn.Module):
    """
    正弦位置编码（batch_first=True），
    pe 形状：(1, max_len, E)；输入输出均为 (B, T, E)
    """
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)      # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim)
        )  # (E/2,)

        pe = torch.zeros(max_len, embed_dim, dtype=torch.float)                  # (max_len, E)
        pe[:, 0::2] = torch.sin(position * div_term)                             # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)                             # 奇数维
        pe = pe.unsqueeze(0)                                                     # (1, max_len, E)
        self.register_buffer("pe", pe)  # 非参数，随设备移动

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, E)
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TemporalEncoderBlockWithCrossAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, hidden_dim, baseline_dim, dropout=0.1,
                 kv_token_count: int = None):  # 新增：K/V 的token数；默认跟随序列长度
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.baseline_proj = nn.Linear(baseline_dim, embed_dim)

        self.positional_encoding = TemporalPositionalEncoding(embed_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # pre-norm
        self.norm_q_cross  = nn.LayerNorm(embed_dim)
        self.norm_kv_cross = nn.LayerNorm(embed_dim)
        self.norm_self     = nn.LayerNorm(embed_dim)
        self.norm_ffn      = nn.LayerNorm(embed_dim)

        self.drop_cross = nn.Dropout(dropout)
        self.drop_self  = nn.Dropout(dropout)
        self.drop_ffn   = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # === 关键：为 baseline 生成多 token 的可学习“时间嵌入” ===
        self.kv_token_count = kv_token_count  # 若为 None，则运行时用当前序列长度 T
        # 这里设上限 512，够用了；也可以设成 2048 或改成动态 ParameterList
        max_tokens = 512
        self.baseline_token_pe = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        nn.init.trunc_normal_(self.baseline_token_pe, std=0.02)

    def forward(self, x, baseline_feat):
        """
        x: (B, T, input_dim)
        baseline_feat: (B, baseline_dim)  —— 原始 baseline
        """
        x = self.input_proj(x)               # (B, T, E)
        x = self.positional_encoding(x)      # (B, T, E)

        B, T, E = x.shape
        b = self.baseline_proj(baseline_feat)  # (B, E)

        # ---- 生成多 token 的 K/V ----
        M = self.kv_token_count or T         # token 数：默认与 T 相同
        if M > self.baseline_token_pe.size(1):
            raise ValueError(f"M={M} exceeds max_tokens={self.baseline_token_pe.size(1)}; enlarge max_tokens.")
        # 广播 baseline 到 M 个 token，并加入可学习位置偏置（确保不同 token 可区分）
        b = b.unsqueeze(1).expand(B, M, E) + self.baseline_token_pe[:, :M, :]  # (B, M, E)

        # Cross-Attn: Q=x, K/V=b （此时 softmax 维度=M>1，依赖 Q）
        x_q = self.norm_q_cross(x)
        b_kv = self.norm_kv_cross(b)
        cross_out, _ = self.cross_attn(x_q, b_kv, b_kv)   # (B, T, E)
        x = x + self.drop_cross(cross_out)

        # Self-Attn
        x_sa = self.norm_self(x)
        self_out, _ = self.self_attn(x_sa, x_sa, x_sa)    # (B, T, E)
        x = x + self.drop_self(self_out)

        # FFN
        x_ffn = self.norm_ffn(x)
        x = x + self.drop_ffn(self.ff(x_ffn))
        return x
