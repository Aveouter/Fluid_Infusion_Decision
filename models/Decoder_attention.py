import torch.nn as nn

class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.ff(self.norm(x))

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        attn_output, _ = self.attn(x, x, x)
        return residual + self.dropout(attn_output)

class WaterClassifier(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_speed=1500.0):
        super().__init__()
        self.max_speed = max_speed
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout)
        )
        # 回归头：连续输出 -> 用 sigmoid 映射到 [0, max_speed]
        self.fspeed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(embed_dim, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 3)  # 速度是 1 维；若你要 3 维，就改成 3
        )
        # 分类头（保留原始）
        self.ftype = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(embed_dim, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        h = self.norm(x)
        h = self.classifier(h) + x
        speed = self.fspeed(h)
        # speed = torch.sigmoid(speed_raw) * self.max_speed   # ∈ [0,1500]
        return speed, self.ftype(h)

class TransformerDecoderWithClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                FeedForwardBlock(embed_dim, hidden_dim, dropout),
                SelfAttentionBlock(embed_dim, num_heads, dropout)
            ) for _ in range(num_layers)
        ])
        self.classifier = WaterClassifier(embed_dim, dropout)

    def forward(self, x):
        # print(f'x.begin:{x.shape}')
        for layer in self.layers:
            x = layer[0](x)
            x = layer[1](x)
        # print(x.shape)
        # x = x.mean(dim=1)  # Global average pooling on time axis
        # print(x.shape)
        # exit()
        return self.classifier(x)
