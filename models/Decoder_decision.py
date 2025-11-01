import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super(FeedForwardBlock, self).__init__()
        self.ff = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.ff(x)  # 残差连接

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x, _ = self.attn(x, x, x)
        x = self.dropout(x)
        return residual + x  # 残差连接

class SpeedPredictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super(SpeedPredictor, self).__init__()
        self.ff = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x):
        return self.ff(x)

class TypePredictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super(TypePredictor, self).__init__()
        self.ff = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x):
        return self.ff(x)

class Decoder_dec(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads=4, dropout=0.1):
        super(Decoder_dec, self).__init__()

        # 两个完全分离的路径
        self.speed_branch = nn.Sequential(
            FeedForwardBlock(embed_dim, hidden_dim, dropout),
            SelfAttentionBlock(embed_dim, num_heads, dropout),
            SpeedPredictor(embed_dim, hidden_dim, dropout)
        )

        self.type_branch = nn.Sequential(
            FeedForwardBlock(embed_dim, hidden_dim, dropout),
            SelfAttentionBlock(embed_dim, num_heads, dropout),
            TypePredictor(embed_dim, hidden_dim, dropout)
        )

    def forward(self, x):
        """
        输入: x 形状为 [batch, seq_len, embed_dim]
        输出: 两个任务的输出 [batch, seq_len, 3]
        """
        speed_pred = self.speed_branch(x)
        type_pred = self.type_branch(x)
        return speed_pred, type_pred
