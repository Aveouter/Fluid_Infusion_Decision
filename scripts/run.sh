#!/usr/bin/env bash
set -euo pipefail

# MODELS=("base" "transformer" "lstm" "mlp" "cnn" "gru" "base_1" "flim" "lstm_new" "tcn_film")
MODELS=("base_1")
PY="${PYTHON:-python}"

cd "$(dirname "$0")"

echo "即将依次运行模型:"
printf "%s\n" "${MODELS[@]}"
echo

FAILED=()

for model in "${MODELS[@]}"; do
    echo "=============================="
    echo "运行模型: $model"
    echo "=============================="

    if ! $PY -u main.py --model="$model" "$@"; then
        echo "⚠️ 模型 $model 运行失败"
        FAILED+=("$model")
    fi

    echo
done

echo "=============================="
echo "全部模型运行完毕"
echo "=============================="

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "❌ 运行失败的模型:"
    printf "%s\n" "${FAILED[@]}"
else
    echo "✅ 所有模型运行成功!"
fi