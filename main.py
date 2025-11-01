# main.py
# -*- coding: utf-8 -*-
import os
import time
import json
import glob
import pickle
import logging
import argparse
import random
import numpy as np
import torch

from utils.data_loader import data_loader
from train import train, test  # 需在 train.py 中实现 train() 和 test()
from models.Baseline import BaselineNetwork, BaselineNetwork_1
from models.cnn import CNNBaselineNetwork
from models.transformer import TransformerBaselineNetwork
from models.Lstm import LSTMBaselineNetwork
from models.Flim import build_model_film_ziln
from models.mlp import SimpleMLP
from models.tcn_film import build_model_tcn_film_ziln


# ----------------------------
# Args
# ----------------------------
def arg_parser():
    parser = argparse.ArgumentParser(description="Train/Eval for Fluid Prediction (multilabel + regression)")

    # ===== 数据与模型 =====
    parser.add_argument('--pkl_ts_path', type=str, default='data/output_data_final.pkl', help='时间序列 pkl 路径')
    parser.add_argument('--pkl_base_path', type=str, default='data/baseline.pkl', help='基线特征 pkl 路径')
    parser.add_argument('-history_length', type=int, default=1)
    parser.add_argument('-pred_length', type=int, default=1)
    parser.add_argument('--model', type=str, default='base_1',
                        choices=['base', 'transformer', 'lstm', 'mlp', 'cnn', 'gru', 'base_1', 'flim', 'lstm_new', 'tcn_film'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--head_lr_mult', type=float, default=1.0, help='分类/回归 head 的 LR 倍率（train.py 会用）')
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--input_dim_base', type=int, default=10)
    parser.add_argument('--input_dim_temporal', type=int, default=19)
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train')
    parser.add_argument('--output_path', type=str, default='./ckpts')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--args_name', type=str, default='default')

    # ===== 复现实验 =====
    parser.add_argument('--seed', type=int, default=42)
    parser.addargument = parser.add_argument  # 小别名，方便你临时添加开关
    parser.add_argument('--deterministic', action='store_true', help='启用确定性 cuDNN（可能牺牲速度）')

    # ===== 调试门控（逐类回归门）=====
    parser.add_argument('--debug_gate', action='store_true',
                        help='开启逐类回归门控的调试日志（logging.DEBUG 级别）')

    # ===== 学习率 & 训练策略 =====
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['step', 'cosine'])
    parser.add_argument('--lr_step', type=int, default=100)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--t0', type=int, default=50, help='CosineWarmRestarts 的 T0')
    parser.add_argument('--eta_min_ratio', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=None, help='在 run_epoch 中生效')
    parser.add_argument('--lambda_reg_warmup_epochs', type=int, default=0)
    parser.add_argument('--lambda_reg_warmup_power', type=float, default=1.0)

    # ===== 混合精度 =====
    parser.add_argument('--amp', action='store_true', help='使用 torch.cuda.amp 推理/训练（需模型支持）')

    # ===== 分类 =====
    parser.add_argument('--use_focal', dest='use_focal', action='store_true', help="使用 Focal Loss（默认开）")
    parser.add_argument('--no_focal', dest='use_focal', action='store_false', help="关闭 Focal Loss 用 BCE")
    parser.set_defaults(use_focal=True)
    parser.add_argument('--focal_alpha', type=float, nargs=3, default=[0.75, 0.75, 0.50])
    parser.add_argument('--focal_gamma', type=float, nargs=3, default=[2.0, 2.0, 5.0])
    parser.add_argument('--cls_pos_weight', type=float, nargs=3, default=None)

    # ===== 阈值 & 平滑（重要）=====
    parser.add_argument('--tune_thresholds', dest='tune_thresholds', action='store_true',
                        help="验证时按 F-beta 自动调阈值（默认开）")
    parser.add_argument('--no_tune_thresholds', dest='tune_thresholds', action='store_false',
                        help="关闭自动调阈，改用 fixed_thresholds")
    parser.set_defaults(tune_thresholds=True)

    parser.add_argument('--fbeta_per_class', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="每类 F-beta 系数；<1 偏 Precision，>1 偏 Recall")
    parser.add_argument('--smooth_win', type=int, default=1, help="概率时间平滑窗口（1=不平滑）")
    parser.add_argument('--smooth_classes', type=int, nargs='*', default=[2], help="需要平滑的类别索引（默认仅 water=2）")
    parser.add_argument('--fixed_thresholds', type=float, nargs=3, default=[0.5, 0.5, 0.25],
                        help="关闭调阈时使用")

    # ★★★ 保守化相关参数（与 run_epoch 协同）★★★
    parser.add_argument('--prec_floor', type=float, nargs=3, default=[0.70, 0.70, 0.70],
                        help="每类精度下限（用于 eval 时的受约束调阈）")
    parser.add_argument('--min_thresholds', type=float, nargs=3, default=[0.40, 0.40, 0.40],
                        help="每类阈值下界，避免过低阈值导致几乎全阳性")
    parser.add_argument('--temp_scale', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="eval-only 温度缩放 logits/T_c（>1 使概率更保守）")

    # ===== 回归 & 一致性 =====
    parser.add_argument('--lambda_reg', type=float, default=1.0)
    parser.add_argument('--use_log1p_reg', dest='use_log1p_reg', action='store_true')
    parser.add_argument('--no_use_log1p_reg', dest='use_log1p_reg', action='store_false')
    parser.set_defaults(use_log1p_reg=True)

    parser.add_argument('--use_robust_reg', dest='use_robust_reg', action='store_true')
    parser.add_argument('--no_use_robust_reg', dest='use_robust_reg', action='store_false')
    parser.set_defaults(use_robust_reg=True)

    parser.add_argument('--pos_weight', type=float, default=4.0)
    parser.add_argument('--zero_weight', type=float, default=1.2)
    parser.add_argument('--pos_weight_vec', type=float, nargs=3, default=None)
    parser.add_argument('--zero_weight_vec', type=float, nargs=3, default=None)

    parser.add_argument('--lambda_consistency', type=float, default=0.1)
    parser.add_argument('--cons_pos_eps', type=float, default=5.0)
    parser.add_argument('--cons_zero_eps', type=float, default=1.0)
    parser.add_argument('--zero_target_eps', type=float, default=5.0)

    parser.add_argument('--tol_abs_ml', type=float, default=50.0)
    parser.add_argument('--tol_pct', type=float, default=0.10)
    parser.add_argument('--patience', type=int, default=10)

    # ===== 评估 & 恢复 =====
    parser.add_argument('--resume_ckpt', type=str, default=None, help='eval 模式：指定 ckpt；为空则自动找最新')
    parser.add_argument('--thresholds_path', type=str, default=None, help='eval 使用的 best_thresholds.json 路径')

    # ===== 主指标（关键：以 F 为主导）=====
    parser.add_argument('--main_metric', type=str, default='f1',
                        choices=['f1', 'fbeta', 'auc', 'acc', 'loss'],
                        help='选择早停与最佳模型保存的主指标（默认 f1）')
    parser.add_argument('--fbeta', type=float, default=1.0,
                        help='当 main_metric=fbeta 时使用的 β 值；>1 偏 recall，<1 偏 precision')

    return parser


# ----------------------------
# Logging setup
# ----------------------------
def setup_run_logging(args, phase: str):
    ts = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    log_dir = os.path.join(args.output_path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_filename = (
        f"{phase}_{args.model}"
        f"_hs{args.history_length}_pd{args.pred_length}"
        f"_lr{args.learning_rate}_bs{args.batch_size}_{ts}.log"
    )
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if getattr(args, "debug_gate", False) else logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)

    logging.info(f"日志初始化完成 -> {log_path}")
    return log_path


# ----------------------------
# Build model
# ----------------------------
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
            num_heads=args.num_heads,            # 兼容参数占位
            hidden_dim=args.hidden_dim,          # 兼容参数占位
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
                          args.history_length, args.pred_length, args.hidden_dim).to(device)

    elif args.model == 'cnn':
        model = CNNBaselineNetwork(args.input_dim_base, args.input_dim_temporal,
                                   args.embed_dim, args.hidden_dim, args.num_classes,
                                   args.history_length, args.pred_length).to(device)

    elif args.model == 'gru':
        raise NotImplementedError("GRU is not implemented here.")
    else:
        raise ValueError(f"Invalid model type: {args.model}")

    return model


# ----------------------------
# Utils
# ----------------------------
def _set_seed(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _pick_device(dev_str: str) -> torch.device:
    if torch.cuda.is_available() and dev_str.startswith("cuda"):
        return torch.device(dev_str)
    # 自动回退
    if torch.cuda.is_available():
        logging.warning(f"请求的 device={dev_str} 不可用，回退到 cuda:0")
        return torch.device("cuda:0")
    logging.warning(f"CUDA 不可用，使用 CPU。原 device={dev_str}")
    return torch.device("cpu")


# ----------------------------
# Main
# ----------------------------
def main():
    # 解析参数
    parser = arg_parser()
    args = parser.parse_args()

    # 设备
    device = _pick_device(str(args.device))
    args.device = device
    print(f"Using device {device}")

    # 随机种子
    _set_seed(args.seed, args.deterministic)

    # 目录
    os.makedirs(args.output_path, exist_ok=True)

    # 读取数据（PKL）
    if not os.path.exists(args.pkl_ts_path):
        raise FileNotFoundError(f"时间序列 pkl 未找到: {args.pkl_ts_path}")
    if not os.path.exists(args.pkl_base_path):
        raise FileNotFoundError(f"基线特征 pkl 未找到: {args.pkl_base_path}")

    with open(args.pkl_ts_path, 'rb') as f:
        Ts_data = pickle.load(f)
    with open(args.pkl_base_path, 'rb') as f:
        Base_data = pickle.load(f)

    # DataLoader（明确按病人划分）
    train_loader, val_loader, test_loader = data_loader(
        Ts_data, Base_data,
        history_length=args.history_length,
        pred_length=args.pred_length,
        classes=args.num_classes,
        batch_size=args.batch_size,
        split_mode="patient",
        splits=(0.7, 0.15, 0.15),
        seed=args.seed
    )

    # 构建模型
    model = build_model(args, device)

    # 训练 / 评估
    if args.mode == 'train':
        # 训练日志
        setup_run_logging(args, phase="training")
        logging.info("Training model...")
        logging.info(args)
        train(model, train_loader, val_loader, args)

        # 训练后立即 TEST（固定使用 VAL 阈值 + 最近 best .pth）
        setup_run_logging(args, phase="test")
        logging.info("Training finished. Running TEST with VAL-fixed thresholds...")
        _ = test(model, test_loader, args, ckpt_path=None, thresholds_path=None)

    elif args.mode == 'eval':
        # 仅测试（从 output_path 自动找最近的 .pth 和 best_thresholds.json，或使用命令行指定）
        setup_run_logging(args, phase="test")
        logging.info("Evaluating on TEST set with VAL-fixed thresholds...")
        _ = test(model, test_loader, args, ckpt_path=args.resume_ckpt, thresholds_path=args.thresholds_path)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
