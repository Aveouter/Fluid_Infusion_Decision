import torch.nn as nn

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)  # 添加 dropout
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),             # 添加 dropout
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)              # 添加 dropout
        )

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))  # 对注意力输出应用 dropout
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))    # 对前馈输出应用 dropout
        return x

class BaselineEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(BaselineEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)

        # SelfAttentionBlock 本身是 pre-norm（内部含 LN）
        self.attn_block1 = SelfAttentionBlock(embed_dim, num_heads, dropout)
        self.attn_block2 = SelfAttentionBlock(embed_dim, num_heads, dropout)
        # self.attn_block3 = SelfAttentionBlock(embed_dim, num_heads, dropout)
        # self.attn_block4 = SelfAttentionBlock(embed_dim, num_heads, dropout)

        # FFN：公用一套 MLP；前面各自加 LN 实现 pre-norm
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        # 对每个 FFN 段各自的 pre-norm
        self.ffn1_norm = nn.LayerNorm(embed_dim)
        self.ffn2_norm = nn.LayerNorm(embed_dim)
        # self.ffn3_norm = nn.LayerNorm(embed_dim)
        # self.ffn4_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.input_projection(x)

        # Block 1: Attn (pre-norm inside) -> FFN (pre-norm here) with residual
        x = self.attn_block1(x)
        x = x + self.feed_forward(self.ffn1_norm(x))

        # Block 2
        x = self.attn_block2(x)
        x = x + self.feed_forward(self.ffn2_norm(x))

        # # 如果需要更多层，按同样模式打开即可
        # x = self.attn_block3(x)
        # x = x + self.feed_forward(self.ffn3_norm(x))
        #
        # x = self.attn_block4(x)
        # x = x + self.feed_forward(self.ffn4_norm(x))

        return x
