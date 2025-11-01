# -*- coding: utf-8 -*-
"""
多任务模型集合 + 传统机器学习分支：
- BaselineNetwork / BaselineNetwork_1（线性+FiLM 轻量版）
- TCN + FiLM（build_model_tcn_film_ziln）
- TransformerBaselineNetwork（可选因果掩码、可选FiLM）
- LSTMBaselineNetwork
- CNNBaselineNetwork
- SimpleMLP
- NEW: LinearTraditionalModel（Logistic + Ridge 的浅层头，端到端可微）
- NEW: SklearnKNNMultiTask / SklearnRFMultiTask（两阶段：先 fit_sklearn 再 forward）

统一前向：forward(base_x, temporal_x) -> dict
  返回：
    {
      'cls_logits': (B, pred_len, num_classes),
      'regression': (B, pred_len, out_dim_regression),
      'feat': (B, pred_len, D)
    }

输入：
  base_x:      (B, input_dim_base)
  temporal_x:  (B, history_len, input_dim_temporal)
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 通用小组件
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class FiLM(nn.Module):
    def __init__(self, c_feat: int, c_cond: int):
        super().__init__()
        self.proj = nn.Linear(c_cond, 2 * c_feat)
    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, T, C = feat.shape
        gamma_beta = self.proj(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1).expand(B, T, C)
        beta  = beta .unsqueeze(1).expand(B, T, C)
        return feat * (1.0 + torch.tanh(gamma)) + beta

class TemporalConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, d=1, dropout=0.1, causal=True):
        super().__init__()
        pad = (k - 1) * d if causal else (k - 1) // 2 * d
        self.causal = causal
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=k, dilation=d, padding=pad)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=k, dilation=d, padding=pad)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv1(x)
        if self.causal:
            cut = self.conv1.padding[0]
            out = out[..., :-cut] if cut > 0 else out
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        if self.causal:
            cut = self.conv2.padding[0]
            out = out[..., :-cut] if cut > 0 else out
        out = self.relu(out)
        out = self.dropout(out)
        return out + self.proj(x)

def _causal_mask(Tq: int, Tk: int, device: torch.device):
    m = torch.full((Tq, Tk), fill_value=float('-inf'), device=device)
    m = torch.triu(m, diagonal=1)
    return m

class MultiTaskHead(nn.Module):
    def __init__(self, d_in: int, num_classes: int, pred_len: int,
                 out_dim_regression: int = 1, dropout: float = 0.1):
        super().__init__()
        self.pred_len = pred_len
        self.cls = nn.Sequential(
            nn.LayerNorm(d_in), nn.Dropout(dropout), nn.Linear(d_in, num_classes)
        )
        self.reg = nn.Sequential(
            nn.LayerNorm(d_in), nn.Dropout(dropout), nn.Linear(d_in, out_dim_regression)
        )
    def forward(self, feat_seq: torch.Tensor):
        B, T, D = feat_seq.shape
        if T != self.pred_len:
            x = feat_seq.transpose(1, 2)
            x = F.interpolate(x, size=self.pred_len, mode='linear', align_corners=False)
            feat_seq = x.transpose(1, 2)
        return self.cls(feat_seq), self.reg(feat_seq)

# -----------------------------
# 1) Baseline / Baseline_1
# -----------------------------
class BaselineNetwork(nn.Module):
    def __init__(self, input_dim_base, input_dim_temporal, embed_dim,
                 num_heads, hidden_dim, num_classes, history_length, pred_length,
                 dropout: float = 0.1):
        super().__init__()
        self.pred_length = pred_length
        self.base_proj = nn.Linear(input_dim_base, embed_dim)
        self.temp_proj = nn.Linear(input_dim_temporal, embed_dim)
        self.film = FiLM(embed_dim, embed_dim)
        self.encoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.head = MultiTaskHead(embed_dim, num_classes, pred_length, out_dim_regression=1, dropout=dropout)
    def forward(self, base_x, temporal_x):
        b = self.base_proj(base_x)
        x = self.temp_proj(temporal_x)
        x = self.film(x, b)
        x = self.encoder(x)
        cls, reg = self.head(x)
        return {'cls_logits': cls, 'regression': reg, 'feat': x}

class BaselineNetwork_1(nn.Module):
    def __init__(self, input_dim_base, input_dim_temporal, embed_dim,
                 num_heads, hidden_dim, num_classes, history_length, pred_length,
                 dropout: float = 0.1):
        super().__init__()
        self.base_proj = nn.Sequential(
            nn.Linear(input_dim_base, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.temp_proj = nn.Linear(input_dim_temporal, embed_dim)
        self.film = FiLM(embed_dim, embed_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, embed_dim)
            ) for _ in range(3)
        ])
        self.head = MultiTaskHead(embed_dim, num_classes, pred_length, out_dim_regression=1, dropout=dropout)
    def forward(self, base_x, temporal_x):
        b = self.base_proj(base_x)
        x = self.temp_proj(temporal_x)
        x = self.film(x, b)
        for blk in self.blocks:
            x = x + blk(x)
        cls, reg = self.head(x)
        return {'cls_logits': cls, 'regression': reg, 'feat': x}

# -----------------------------
# 2) TCN + FiLM
# -----------------------------
class TCNFiLMModel(nn.Module):
    def __init__(self, input_dim_base, input_dim_temporal, embed_dim,
                 num_classes, pred_length,
                 tcn_k=3, tcn_layers_short=3, tcn_layers_long=5, causal=True, dropout=0.1,
                 out_dim_regression: int = 3):
        super().__init__()
        self.pred_length = pred_length
        self.base_proj = nn.Linear(input_dim_base, embed_dim)
        self.temp_proj = nn.Linear(input_dim_temporal, embed_dim)
        self.film = FiLM(embed_dim, embed_dim)
        ch = embed_dim
        self.short = nn.ModuleList([
            TemporalConvBlock(ch, ch, k=tcn_k, d=2**i, dropout=dropout, causal=causal)
            for i in range(tcn_layers_short)
        ])
        self.long = nn.ModuleList([
            TemporalConvBlock(ch, ch, k=tcn_k, d=2**i, dropout=dropout, causal=causal)
            for i in range(tcn_layers_long)
        ])
        self.fuse = nn.Sequential(nn.Conv1d(2*ch, ch, 1), nn.ReLU(inplace=True))
        self.head = MultiTaskHead(embed_dim, num_classes, pred_length, out_dim_regression=out_dim_regression, dropout=dropout)
    def forward(self, base_x, temporal_x):
        b = self.base_proj(base_x)
        x = self.temp_proj(temporal_x)
        x = self.film(x, b)
        x1 = x.transpose(1, 2)
        for blk in self.short: x1 = blk(x1)
        x2 = x.transpose(1, 2)
        for blk in self.long: x2 = blk(x2)
        x_f = self.fuse(torch.cat([x1, x2], dim=1)).transpose(1, 2)
        cls, reg = self.head(x_f)
        return {'cls_logits': cls, 'regression': reg, 'feat': x_f}

def build_model_tcn_film_ziln(input_dim_base, input_dim_temporal, embed_dim, num_heads, hidden_dim,
                               num_classes, pred_length, dropout=0.1, tcn_k=3, tcn_layers_short=3,
                               tcn_layers_long=5, causal=True):
    return TCNFiLMModel(input_dim_base, input_dim_temporal, embed_dim, num_classes, pred_length,
                        tcn_k=tcn_k, tcn_layers_short=tcn_layers_short, tcn_layers_long=tcn_layers_long,
                        causal=causal, dropout=dropout, out_dim_regression=3)

# -----------------------------
# 3) Transformer Baseline
# -----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim_ff, d_model)
        )
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        y, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = x + self.drop(y)
        y = self.ff(self.ln2(x))
        x = x + self.drop(y)
        return x

class TransformerBaselineNetwork(nn.Module):
    def __init__(self, input_dim_base: int, input_dim_temporal: int, embed_dim: int,
                 num_heads: int, dim_ff: int, hidden_dim: int, num_classes: int,
                 history_length: int, pred_length: int, num_layers: int = 4, dropout: float = 0.1,
                 causal: bool = True, use_film: bool = True):
        super().__init__()
        self.pred_length = pred_length
        self.base_proj = nn.Linear(input_dim_base, embed_dim)
        self.temp_proj = nn.Linear(input_dim_temporal, embed_dim)
        self.pos = nn.Parameter(torch.randn(1, 1024, embed_dim) * 0.01)
        self.use_film = use_film
        if use_film:
            self.film = FiLM(embed_dim, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dim_ff, dropout=dropout) for _ in range(num_layers)
        ])
        self.head = MultiTaskHead(embed_dim, num_classes, pred_length, out_dim_regression=1, dropout=dropout)
        self.causal = causal
    def forward(self, base_x, temporal_x):
        B, T, _ = temporal_x.shape
        b = self.base_proj(base_x)
        x = self.temp_proj(temporal_x)
        x = x + self.pos[:, :T, :]
        if self.use_film:
            x = self.film(x, b)
        attn_mask = _causal_mask(T, T, x.device) if self.causal else None
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        cls, reg = self.head(x)
        return {'cls_logits': cls, 'regression': reg, 'feat': x}

# -----------------------------
# 4) LSTM Baseline
# -----------------------------
class LSTMBaselineNetwork(nn.Module):
    def __init__(self, input_dim_base: int, input_dim_temporal: int, embed_dim: int,
                 hidden_dim: int, num_classes: int, history_length: int, pred_length: int,
                 rnn_hidden: Optional[int] = None, num_layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.1, use_film: bool = True,
                 out_dim_regression: int = 1):
        super().__init__()
        self.pred_length = pred_length
        self.base_proj = nn.Linear(input_dim_base, embed_dim)
        self.temp_proj = nn.Linear(input_dim_temporal, embed_dim)
        self.use_film = use_film
        if use_film:
            self.film = FiLM(embed_dim, embed_dim)
        H = rnn_hidden or hidden_dim
        self.lstm = nn.LSTM(embed_dim, H, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional)
        out_dim = H * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_dim, embed_dim)
        self.head = MultiTaskHead(embed_dim, num_classes, pred_length, out_dim_regression=out_dim_regression, dropout=dropout)
    def forward(self, base_x, temporal_x):
        b = self.base_proj(base_x)
        x = self.temp_proj(temporal_x)
        if self.use_film:
            x = self.film(x, b)
        x, _ = self.lstm(x)
        x = self.proj(x)
        cls, reg = self.head(x)
        return {'cls_logits': cls, 'regression': reg, 'feat': x}

# -----------------------------
# 5) CNN Baseline
# -----------------------------
class CNNBaselineNetwork(nn.Module):
    def __init__(self, input_dim_base, input_dim_temporal, embed_dim, hidden_dim,
                 num_classes, history_length, pred_length, dropout: float = 0.1,
                 out_dim_regression: int = 1):
        super().__init__()
        self.pred_length = pred_length
        self.base_proj = nn.Linear(input_dim_base, embed_dim)
        self.temp_proj = nn.Linear(input_dim_temporal, embed_dim)
        self.film = FiLM(embed_dim, embed_dim)
        ch = embed_dim
        self.conv = nn.Sequential(
            nn.Conv1d(ch, hidden_dim, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, ch, kernel_size=1)
        )
        self.head = MultiTaskHead(embed_dim, num_classes, pred_length, out_dim_regression=out_dim_regression, dropout=dropout)
    def forward(self, base_x, temporal_x):
        b = self.base_proj(base_x)
        x = self.temp_proj(temporal_x)
        x = self.film(x, b)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        cls, reg = self.head(x)
        return {'cls_logits': cls, 'regression': reg, 'feat': x}

# -----------------------------
# 6) Simple MLP
# -----------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim_base, input_dim_temporal, history_length, pred_length, hidden_dim,
                 num_classes: int = 2, out_dim_regression: int = 1, dropout: float = 0.1):
        super().__init__()
        self.pred_length = pred_length
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim_base + input_dim_temporal, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)
        )
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.reg_head = nn.Linear(hidden_dim, out_dim_regression)
    def forward(self, base_x, temporal_x):
        x_pool = self.pool(temporal_x.transpose(1, 2)).squeeze(-1)
        x = torch.cat([base_x, x_pool], dim=-1)
        h = self.fc(x)
        cls = self.cls_head(h).unsqueeze(1).repeat(1, self.pred_length, 1)
        reg = self.reg_head(h).unsqueeze(1).repeat(1, self.pred_length, 1)
        return {'cls_logits': cls, 'regression': reg, 'feat': h.unsqueeze(1).repeat(1, self.pred_length, 1)}

# -----------------------------
# 7) 仅 FiLM + 线性细化（build_model_film_ziln）
# -----------------------------
class FiLMOnlyModel(nn.Module):
    def __init__(self, input_dim_base: int, input_dim_temporal: int, embed_dim: int,
                 num_heads: int, hidden_dim: int, num_classes: int, pred_length: int,
                 out_dim_regression: int = 3, num_temporal_layers: int = 2, dropout: float = 0.1,
                 add_rel_pe: bool = True, max_days: int = 3, refine_layers: int = 1, step_hours: float = 0.1):
        super().__init__()
        self.pred_length = pred_length
        self.base_proj = nn.Linear(input_dim_base, embed_dim)
        self.temp_proj = nn.Linear(input_dim_temporal, embed_dim)
        self.film = FiLM(embed_dim, embed_dim)
        self.add_rel_pe = add_rel_pe
        if add_rel_pe:
            Tmax = int(max(1, round(max_days * 24 / max(step_hours, 1e-6))))
            self.rel_time = nn.Parameter(torch.randn(1, Tmax, embed_dim) * 0.01)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, embed_dim)
            ) for _ in range(num_temporal_layers)
        ])
        self.refine = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))
            for _ in range(refine_layers)
        ])
        self.head = MultiTaskHead(embed_dim, num_classes, pred_length, out_dim_regression=out_dim_regression, dropout=dropout)
    def forward(self, base_x, temporal_x):
        B, T, _ = temporal_x.shape
        b = self.base_proj(base_x)
        x = self.temp_proj(temporal_x)
        x = self.film(x, b)
        if self.add_rel_pe:
            Tmax = self.rel_time.shape[1]
            if T > Tmax:
                pe = F.interpolate(self.rel_time.transpose(1, 2), size=T, mode='linear', align_corners=False).transpose(1, 2)
            else:
                pe = self.rel_time[:, :T, :]
            x = x + pe
        for layer in self.layers: x = x + layer(x)
        for layer in self.refine: x = x + layer(x)
        cls, reg = self.head(x)
        return {'cls_logits': cls, 'regression': reg, 'feat': x}

def build_model_film_ziln(input_dim_base: int, input_dim_temporal: int, embed_dim: int,
                          num_heads: int, hidden_dim: int, num_classes: int, pred_length: int,
                          out_dim_regression: int = 3, num_temporal_layers: int = 2, dropout: float = 0.1,
                          max_days: int = 3, add_rel_pe: bool = True, refine_layers: int = 1, step_hours: float = 0.1):
    return FiLMOnlyModel(input_dim_base, input_dim_temporal, embed_dim, num_heads, hidden_dim, num_classes, pred_length,
                         out_dim_regression=out_dim_regression, num_temporal_layers=num_temporal_layers, dropout=dropout,
                         max_days=max_days, add_rel_pe=add_rel_pe, refine_layers=refine_layers, step_hours=step_hours)

# -----------------------------
# 8) 传统机器学习风格：浅层特征 + 线性/Sklearn 头
# -----------------------------
class ClassicFeatureExtractor(nn.Module):
    def __init__(self, input_dim_base:int, input_dim_temporal:int, embed_dim:int=64):
        super().__init__()
        self.base_proj = nn.Linear(input_dim_base, embed_dim)
        self.temp_proj = nn.Linear(input_dim_temporal*5, embed_dim)
        self.out_dim = embed_dim * 2
    def forward(self, base_x:torch.Tensor, temporal_x:torch.Tensor):
        b = self.base_proj(base_x)
        t_mean = temporal_x.mean(dim=1)
        t_std  = temporal_x.std(dim=1)
        t_min  = temporal_x.min(dim=1).values
        t_max  = temporal_x.max(dim=1).values
        t_last = temporal_x[:, -1, :]
        t_feat = torch.cat([t_mean, t_std, t_min, t_max, t_last], dim=-1)
        t = self.temp_proj(t_feat)
        return torch.cat([b, t], dim=-1)

class LinearTraditionalModel(nn.Module):
    def __init__(self, input_dim_base, input_dim_temporal, num_classes:int, pred_length:int,
                 embed_dim:int=64, out_dim_regression:int=1):
        super().__init__()
        self.pred_length = pred_length
        self.feat = ClassicFeatureExtractor(input_dim_base, input_dim_temporal, embed_dim=embed_dim)
        Fdim = self.feat.out_dim
        self.cls_head = nn.Linear(Fdim, num_classes)
        self.reg_head = nn.Linear(Fdim, out_dim_regression)
    def forward(self, base_x, temporal_x):
        z = self.feat(base_x, temporal_x)
        cls = self.cls_head(z).unsqueeze(1).repeat(1, self.pred_length, 1)
        reg = self.reg_head(z).unsqueeze(1).repeat(1, self.pred_length, 1)
        return { 'cls_logits': cls, 'regression': reg, 'feat': z.unsqueeze(1).repeat(1, self.pred_length, 1) }

# sklearn 包装（可选依赖）
try:
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

class _SklearnMultiTaskBase(nn.Module):
    def __init__(self, input_dim_base, input_dim_temporal, pred_length, embed_dim=64,
                 num_classes=2, out_dim_regression=1, kind='knn', **kwargs):
        super().__init__()
        if not _HAS_SKLEARN:
            raise ImportError('scikit-learn 未安装，无法使用经典模型：pip install scikit-learn')
        self.pred_length = pred_length
        self.feat = ClassicFeatureExtractor(input_dim_base, input_dim_temporal, embed_dim=embed_dim)
        if kind == 'knn':
            n_neighbors = kwargs.get('n_neighbors', 15)
            self.clf = KNeighborsClassifier(n_neighbors=n_neighbors)
            self.reg = KNeighborsRegressor(n_neighbors=n_neighbors)
        elif kind == 'rf':
            self.clf = RandomForestClassifier(n_estimators=kwargs.get('n_estimators', 200),
                                              max_depth=kwargs.get('max_depth', None), n_jobs=-1)
            self.reg = RandomForestRegressor(n_estimators=kwargs.get('n_estimators', 200),
                                             max_depth=kwargs.get('max_depth', None), n_jobs=-1)
        else:
            raise ValueError(f'Unknown sklearn kind: {kind}')
        self._fitted = False
        self._num_classes = num_classes
        self._out_dim_regression = out_dim_regression
    @torch.no_grad()
    def fit_sklearn(self, base_x:torch.Tensor, temporal_x:torch.Tensor, y_cls:torch.Tensor, y_reg:torch.Tensor):
        self.eval()
        z = self.feat(base_x.cpu(), temporal_x.cpu()).numpy()
        y_cls_1 = y_cls[:, 0].cpu().numpy()
        y_reg_1 = y_reg[:, 0].cpu().numpy()
        self.clf.fit(z, y_cls_1)
        self.reg.fit(z, y_reg_1)
        self._fitted = True
        return self
    @torch.no_grad()
    def forward(self, base_x, temporal_x):
        import numpy as np
        z = self.feat(base_x, temporal_x).cpu().numpy()
        if not self._fitted:
            raise RuntimeError('Sklearn 模型尚未 fit_sklearn(...)，无法 forward。')
        if hasattr(self.clf, 'predict_proba'):
            proba = self.clf.predict_proba(z)
            proba = np.clip(proba, 1e-6, 1.0)
            logits = torch.from_numpy(np.log(proba)).float()
        else:
            pred = self.clf.predict(z)
            logits = torch.zeros((z.shape[0], self._num_classes), dtype=torch.float32)
            logits[torch.arange(z.shape[0]), torch.from_numpy(pred).long()] = 1.0
        yreg = torch.from_numpy(self.reg.predict(z)).float()
        cls = logits.unsqueeze(1).repeat(1, self.pred_length, 1)
        reg = yreg.unsqueeze(1).repeat(1, self.pred_length, 1)
        return { 'cls_logits': cls.to(base_x.device), 'regression': reg.to(base_x.device),
                 'feat': torch.from_numpy(z).to(base_x.device).unsqueeze(1).repeat(1, self.pred_length, 1) }

class SklearnKNNMultiTask(_SklearnMultiTaskBase):
    def __init__(self, input_dim_base, input_dim_temporal, num_classes, pred_length,
                 embed_dim=64, out_dim_regression=1, **kwargs):
        super().__init__(input_dim_base, input_dim_temporal, pred_length, embed_dim,
                         num_classes, out_dim_regression, kind='knn', **kwargs)

class SklearnRFMultiTask(_SklearnMultiTaskBase):
    def __init__(self, input_dim_base, input_dim_temporal, num_classes, pred_length,
                 embed_dim=64, out_dim_regression=1, **kwargs):
        super().__init__(input_dim_base, input_dim_temporal, pred_length, embed_dim,
                         num_classes, out_dim_regression, kind='rf', **kwargs)

# -----------------------------
# 9) 工厂函数（含传统模型分支）
# -----------------------------

def build_model(args, device):
    if args.model == 'base':
        model = BaselineNetwork(args.input_dim_base, args.input_dim_temporal, args.embed_dim,
                                args.num_heads, args.hidden_dim, args.num_classes,
                                args.history_length, args.pred_length).to(device)

    elif args.model == 'tcn_film':
        model = build_model_tcn_film_ziln(
            input_dim_base=args.input_dim_base,
            input_dim_temporal=args.input_dim_temporal,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
            pred_length=args.pred_length,
            dropout=getattr(args, "dropout", 0.1),
            tcn_k=getattr(args, "tcn_k", 3),
            tcn_layers_short=getattr(args, "tcn_layers_short", 3),
            tcn_layers_long=getattr(args, "tcn_layers_long", 5),
            causal=getattr(args, "causal_temporal", True),
        ).to(device)

    elif args.model == 'transformer':
        model = TransformerBaselineNetwork(
            input_dim_base=args.input_dim_base,
            input_dim_temporal=args.input_dim_temporal,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            dim_ff=args.hidden_dim,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
            history_length=args.history_length,
            pred_length=args.pred_length,
            num_layers=getattr(args, "num_layers", 4),
            dropout=getattr(args, "dropout", 0.1),
            causal=getattr(args, "causal", True),
            use_film=getattr(args, "use_film", True)
        ).to(device)

    elif args.model == 'base_1':
        model = BaselineNetwork_1(args.input_dim_base, args.input_dim_temporal, args.embed_dim,
                                  args.num_heads, args.hidden_dim, args.num_classes,
                                  args.history_length, args.pred_length).to(device)

    elif args.model == "flim":
        model = build_model_film_ziln(
            input_dim_base=args.input_dim_base,
            input_dim_temporal=args.input_dim_temporal,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
            pred_length=args.pred_length,
            out_dim_regression=getattr(args, "out_dim_regression", 3),
            num_temporal_layers=getattr(args, "num_temporal_layers", 2),
            dropout=getattr(args, "dropout", 0.1),
            max_days=getattr(args, "max_days", 3),
            add_rel_pe=getattr(args, "add_rel_pe", True),
            refine_layers=getattr(args, "refine_layers", 1),
            step_hours=getattr(args, "step_hours", 0.1),
        ).to(device)

    elif args.model == 'lstm_new' or args.model == 'lstm':
        model = LSTMBaselineNetwork(
            input_dim_base=args.input_dim_base,
            input_dim_temporal=args.input_dim_temporal,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
            history_length=args.history_length,
            pred_length=args.pred_length,
            rnn_hidden=getattr(args, "rnn_hidden", None),
            num_layers=getattr(args, "num_layers", 2),
            bidirectional=getattr(args, "bidirectional", True),
            dropout=getattr(args, "dropout", 0.1),
            use_film=getattr(args, "use_film", True)
        ).to(device)

    elif args.model == 'mlp':
        model = SimpleMLP(args.input_dim_base, args.input_dim_temporal,
                          args.history_length, args.pred_length, args.hidden_dim,
                          num_classes=getattr(args, 'num_classes', 2),
                          out_dim_regression=getattr(args, 'out_dim_regression', 1),
                          dropout=getattr(args, 'dropout', 0.1)).to(device)

    elif args.model == 'cnn':
        model = CNNBaselineNetwork(args.input_dim_base, args.input_dim_temporal,
                                   args.embed_dim, args.hidden_dim, args.num_classes,
                                   args.history_length, args.pred_length,
                                   dropout=getattr(args, 'dropout', 0.1),
                                   out_dim_regression=getattr(args, 'out_dim_regression', 1)).to(device)

    # -------- 新增：传统机器学习分支 --------
    elif args.model == 'linear_trad':
        model = LinearTraditionalModel(
            input_dim_base=args.input_dim_base,
            input_dim_temporal=args.input_dim_temporal,
            num_classes=args.num_classes,
            pred_length=args.pred_length,
            embed_dim=getattr(args, 'embed_dim', 64),
            out_dim_regression=getattr(args, 'out_dim_regression', 1),
        ).to(device)

    elif args.model == 'knn_trad':
        model = SklearnKNNMultiTask(
            input_dim_base=args.input_dim_base,
            input_dim_temporal=args.input_dim_temporal,
            num_classes=args.num_classes,
            pred_length=args.pred_length,
            embed_dim=getattr(args, 'embed_dim', 64),
            out_dim_regression=getattr(args, 'out_dim_regression', 1),
            n_neighbors=getattr(args, 'n_neighbors', 15),
        ).to(device)

    elif args.model == 'rf_trad':
        model = SklearnRFMultiTask(
            input_dim_base=args.input_dim_base,
            input_dim_temporal=args.input_dim_temporal,
            num_classes=args.num_classes,
            pred_length=args.pred_length,
            embed_dim=getattr(args, 'embed_dim', 64),
            out_dim_regression=getattr(args, 'out_dim_regression', 1),
            n_estimators=getattr(args, 'n_estimators', 200),
            max_depth=getattr(args, 'max_depth', None),
        ).to(device)

    elif args.model == 'gru':
        raise NotImplementedError("GRU is not implemented here.")
    else:
        raise ValueError(f"Invalid model type: {args.model}")

    return model
