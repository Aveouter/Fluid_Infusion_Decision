# -*- coding: utf-8 -*-
"""
Differentiable 'traditional ML' style models for your trainer:
- TorchLRGBRApprox:  logistic regression (classification) + shallow MLP regressor (Huber)  → 代替 LR+GBR
- TorchRidge:        L2 线性分类 + 线性回归（带权衰减实现 Ridge）
- TorchSVRApprox:    线性分类 + 线性回归（回归头建议配合 epsilon/Huber 损失，在 run_epoch 侧已有鲁棒项即可）

统一前向:
    forward(input_T, input_B) -> {
        "class_logits": (B,P,3),
        "flow_pred":    (B,P,3),     # 非负，单位与标签一致
        "extras": {}
    }

特征构造（轻量+稳定）:
    - 取最后 P 个时间步：Z_t = input_T[:, -P:, :]  ->  (B,P,D_t)
    - baseline 平铺到每个时间步：Z_b = repeat(input_B, P)  ->  (B,P,D_b)
    - 拼接: F = [Z_t | Z_b]  ->  (B,P,D_t + D_b)
    - 可选：简单统计增强（mean/std/last），默认关闭以保持快速与稳健

超参:
    - pred_length:   P
    - in_temporal:   D_t
    - in_baseline:   D_b
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------- 通用小块 ---------
class _TwoLayerMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _build_concat_features(x_T: torch.Tensor, x_B: torch.Tensor, pred_length: int) -> torch.Tensor:
    """
    x_T: (B, H, D_t)
    x_B: (B, D_b)
    return F: (B, P, D_t + D_b)
    """
    B, H, Dt = x_T.shape
    P = int(pred_length)
    assert H >= P, f"H={H} must be ≥ pred_length={P}"

    Z_t = x_T[:, -P:, :]                             # (B,P,Dt)
    Z_b = x_B.unsqueeze(1).expand(B, P, x_B.size(-1))# (B,P,Db)
    F = torch.cat([Z_t, Z_b], dim=-1)                # (B,P,Dt+Db)
    return F


# ===============================================================
# 1) LR + 'GBR' 的可微近似（分类：逻辑回归；回归：浅层 MLP + Softplus）
# ===============================================================
class TorchLRGBRApprox(nn.Module):
    def __init__(self,
                 input_dim_temporal: int,
                 input_dim_base: int,
                 pred_length: int,
                 hidden_reg: int = 128,
                 dropout: float = 0.0,
                 num_classes: int = 3):
        super().__init__()
        self.pred_length = int(pred_length)
        in_dim = int(input_dim_temporal + input_dim_base)

        # 分类：线性层实现逻辑回归（BCEWithLogits 由外部 loss 处理）
        self.cls_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, num_classes)   # logits
        )

        # 回归：用浅层 MLP 近似 GBR 的非线性映射
        self.reg_head = _TwoLayerMLP(in_dim, hidden_reg, num_classes, dropout=dropout)

        # 输出非负（速度≥0）
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, input_T: torch.Tensor, input_B: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        input_T: (B, H, D_t)
        input_B: (B, D_b)
        """
        F = _build_concat_features(input_T, input_B, self.pred_length)    # (B,P,Dt+Db)

        logits = self.cls_head(F)                                         # (B,P,3)
        flow   = self.softplus(self.reg_head(F))                          # (B,P,3)

        return {
            "class_logits": logits,
            "flow_pred": flow,
            "extras": {}
        }


# ===============================================================
# 2) Ridge：线性分类 + 线性回归（训练时用 weight_decay≈L2）
# ===============================================================
class TorchRidge(nn.Module):
    """
    训练时设置优化器的 weight_decay>0，即可得到 Ridge 效果（L2 正则）。
    """
    def __init__(self,
                 input_dim_temporal: int,
                 input_dim_base: int,
                 pred_length: int,
                 num_classes: int = 3):
        super().__init__()
        self.pred_length = int(pred_length)
        in_dim = int(input_dim_temporal + input_dim_base)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, num_classes)
        )
        self.reg_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, num_classes)
        )
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, input_T: torch.Tensor, input_B: torch.Tensor) -> Dict[str, torch.Tensor]:
        F = _build_concat_features(input_T, input_B, self.pred_length)
        logits = self.cls_head(F)
        flow   = self.softplus(self.reg_head(F))
        return {"class_logits": logits, "flow_pred": flow, "extras": {}}


# ===============================================================
# 3) SVR 的可微近似：线性回归 + ε-不敏感/Huber 由外部 loss 近似
#    （你当前 run_epoch 里已有鲁棒项和 log1p 选项，直接用 MSE/Huber 即可）
# ===============================================================
class TorchSVRApprox(nn.Module):
    def __init__(self,
                 input_dim_temporal: int,
                 input_dim_base: int,
                 pred_length: int,
                 num_classes: int = 3):
        super().__init__()
        self.pred_length = int(pred_length)
        in_dim = int(input_dim_temporal + input_dim_base)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, num_classes)
        )
        # 线性回归头；ε-不敏感/Huber 在 run_epoch 的损失里实现（或直接用 MSE/Huber）
        self.reg_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, num_classes)
        )
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, input_T: torch.Tensor, input_B: torch.Tensor) -> Dict[str, torch.Tensor]:
        F = _build_concat_features(input_T, input_B, self.pred_length)
        logits = self.cls_head(F)
        flow   = self.softplus(self.reg_head(F))
        return {"class_logits": logits, "flow_pred": flow, "extras": {}}


# ===============================================================
# 工厂函数（与你 main/build_model 的调用保持一致）
# ===============================================================
def build_model_ml_lr_gbr(input_dim_base: int,
                          input_dim_temporal: int,
                          embed_dim: int,          # 兼容占位，不用
                          num_heads: int,          # 兼容占位，不用
                          hidden_dim: int,         # 用作 reg_head 的 hidden
                          num_classes: int,
                          pred_length: int,
                          **kwargs) -> nn.Module:
    dropout = float(kwargs.get("dropout", 0.0))
    return TorchLRGBRApprox(
        input_dim_temporal=input_dim_temporal,
        input_dim_base=input_dim_base,
        pred_length=pred_length,
        hidden_reg=max(32, int(hidden_dim)),
        dropout=dropout,
        num_classes=num_classes
    )


def build_model_ml_ridge(input_dim_base: int,
                         input_dim_temporal: int,
                         embed_dim: int,
                         num_heads: int,
                         hidden_dim: int,
                         num_classes: int,
                         pred_length: int,
                         **kwargs) -> nn.Module:
    return TorchRidge(
        input_dim_temporal=input_dim_temporal,
        input_dim_base=input_dim_base,
        pred_length=pred_length,
        num_classes=num_classes
    )


def build_model_ml_svr(input_dim_base: int,
                       input_dim_temporal: int,
                       embed_dim: int,
                       num_heads: int,
                       hidden_dim: int,
                       num_classes: int,
                       pred_length: int,
                       **kwargs) -> nn.Module:
    return TorchSVRApprox(
        input_dim_temporal=input_dim_temporal,
        input_dim_base=input_dim_base,
        pred_length=pred_length,
        num_classes=num_classes
    )
