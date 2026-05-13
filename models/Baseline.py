import torch.nn as nn
from models.BaseEncoder import BaselineEncoder
from models.TemporalEncoder import TemporalEncoderBlock
from models.TemporalEncoderBlockWithCrossAttention import TemporalEncoderBlockWithCrossAttention
from models.Decoder_decision import Decoder_dec
from models.Decoder_attention import TransformerDecoderWithClassifier  # Using existing decoder
import torch

class BaselineNetwork(nn.Module):
    def __init__(self, input_dim_base,input_dim_temporal, embed_dim, num_heads, hidden_dim, num_classes,history_length,pred_length):
        super(BaselineNetwork, self).__init__()
        self.pred_length = pred_length
        self.history_length = history_length
        self.BaselineEncoder = BaselineEncoder(input_dim_base, embed_dim, num_heads, hidden_dim)
        self.TemporalEncoder = TemporalEncoderBlock(input_dim_temporal, embed_dim, num_heads, hidden_dim)
        self.decoder = Decoder_dec(hidden_dim, hidden_dim, 4)

    def forward(self,y,x):
        encoded_x = self.BaselineEncoder(x)
        encoded_y = self.TemporalEncoder(y)
        encoded_y = encoded_y[:,-self.pred_length:,:].view(encoded_x.size(0), -1)  # Flatten the output  
        # 在特征维度相加
        encoded_x = torch.cat((encoded_x, encoded_y), dim=1)

        a,b = self.decoder(encoded_x)

        return a.unsqueeze(1), b.unsqueeze(1)
    

class BaselineNetwork_1(nn.Module):
    """
    y: (B, T_y, input_dim_temporal)  —— 时序输入
    x: (B, input_dim_base)          —— 基线/静态输入

    流程：
      enc_x = BaselineEncoder(x)                  # (B, E)
      enc_y = TemporalEncoder(y, x)               # (B, T_y, E)
      enc_yP = enc_y[:, -pred_length:, :]         # (B, P, E)
      enc_x_rep = repeat enc_x to (B, P, E)       # (B, P, E)
      dec_in = cat(enc_yP, enc_x_rep, dim=-1)     # (B, P, 2E)
      dec_in = LazyLinear(2E -> hidden_dim)       # (B, P, hidden_dim)
      outputs = decoder(dec_in)                   # e.g. (fspeed, ftype)

    注意：
      - 不再展平时间维；这样 decoder 的注意力层才有意义。
      - 使用 LazyLinear 自动适配拼接后的维度，无需手算 2E。
    """

    def __init__(self,
                 input_dim_base: int,
                 input_dim_temporal: int,
                 embed_dim: int,
                 num_heads: int,
                 hidden_dim: int,
                 num_classes: int,
                 history_length: int,
                 pred_length: int):
        super().__init__()
        self.pred_length = pred_length
        self.history_length = history_length

        # 编码器（你已有的实现）
        self.BaselineEncoder = BaselineEncoder(
            input_dim=input_dim_base,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim
        )

        self.TemporalEncoder = TemporalEncoderBlockWithCrossAttention(
            input_dim=input_dim_temporal,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            baseline_dim=input_dim_base,
            dropout=0.1
        )

        # 将 (B, P, 2E) 自适配到 decoder 期望的 embed_dim（此处沿用 hidden_dim）
        # LazyLinear 会在第一次前向时根据输入最后一维自动建立权重
        self.to_decoder = nn.LazyLinear(hidden_dim)

        # 解码器：保持你之前的用法（embed_dim=hidden_dim）
        self.decoder = TransformerDecoderWithClassifier(
            embed_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_heads=num_heads,
            dropout=0.1
        )

    def forward(self, y, x):
        """
        y: (B, T_y, input_dim_temporal)
        x: (B, input_dim_base)
        return: 与 decoder 对齐的输出（例如 (fspeed, ftype)）
        """
        encoded_x = self.BaselineEncoder(x)  # -> (B, E)

        encoded_y = self.TemporalEncoder(y, x)  # -> (B, T_y, E)

        encoded_y = encoded_y[:, -self.pred_length:, :]

        encoded_x_rep = encoded_x.unsqueeze(1).expand(-1, self.pred_length, -1)

        dec_in = torch.cat([encoded_y, encoded_x_rep], dim=-1)

        dec_in = self.to_decoder(dec_in)

        return self.decoder(dec_in)
