#!/bin/bash

# 模型列表
models=(
    # "base"
    "transformer"
    # "lstm"
    # "mlp"
    "cnn"
    # "gru"
    "base_1"
    "flim"
    "lstm_new"
    "tcn_film"
    "ml_lr_gbr"
    "ml_rf"
    "ml_ridge"

    "ml_svr"
    # # 新增：消融
    # "ours"
    "ours_wo_temporal"
    "ours_wo_baseline"
    "ours_wo_selfattn"
    "ours_wo_tcn"
    "ours_wo_film"
)

# 循环运行所有模型
for model in "${models[@]}"
do
    echo "正在运行模型: $model"
    python main.py --model="$model"
    
    # 检查上一个命令的退出状态
    if [ $? -eq 0 ]; then
        echo "模型 $model 运行成功"
    else
        echo "模型 $model 运行失败"
        # exit 1  # 可选：如果失败则停止运行
    fi
    
    echo "----------------------------------------"
done

echo "所有模型运行完成"