

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, input_dim_base,input_dim_temporal,history_length,pred_length,hidden_dim):
        super(SimpleMLP, self).__init__()
        self.input_shapes = input_dim_base + input_dim_temporal * history_length
        self.history_length = history_length
        self.pred_length = pred_length
        self.input_dim_base = input_dim_base
        self.input_dim_temporal = input_dim_temporal
        self.hidden_dim = hidden_dim
        # 构建更深的 MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.input_shapes, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 6),
            nn.Sigmoid()
        )  
    def forward(self, x, y):
        # 将所有输入 flatten 并合并
        x = x.view(x.size(0), -1)  # [B, N1 + N2 + ...]
        y = y.view(y.size(0), -1)
        flat_inputs = torch.cat([x, y], dim=1)# [B, N1 + N2 + ...]
        en_x = self.mlp(flat_inputs)
        m ,n = en_x[:,:3], en_x[:,3:]
        return m,n