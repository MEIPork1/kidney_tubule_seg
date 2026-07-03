#!/bin/bash
# 在服务器上运行：上传模型权重到 Hugging Face
# 用法: HF_TOKEN="hf_xxx" bash upload_models.sh
set -e

if [ -z "$HF_TOKEN" ]; then
    echo "错误: 请设置 HF_TOKEN 环境变量"
    echo "用法: HF_TOKEN='hf_xxx' bash upload_models.sh"
    echo "获取 token: https://huggingface.co/settings/tokens"
    exit 1
fi

MODEL_DIR="/root/kidney/models"
HF_REPO="MEIPork/kidney-tubule-cpsam"

echo "===== 1. 安装 huggingface_hub ====="
pip install huggingface_hub -q

echo "===== 2. 登录 Hugging Face ====="
hf auth login --token "$HF_TOKEN"

echo "===== 3. 上传模型权重 ====="
for FOLD in 0 1 2 3 4; do
    SRC="${MODEL_DIR}/cpsam_v2_fold${FOLD}/best_model"
    DST="cpsam_v2_fold${FOLD}/best_model"
    if [ -f "$SRC" ]; then
        SIZE=$(du -h "$SRC" | cut -f1)
        echo "  上传 fold${FOLD} (${SIZE})..."
        hf upload "$HF_REPO" "$SRC" "$DST" --commit-message "Add fold${FOLD} CPSAM model"
    else
        echo "  跳过 fold${FOLD}: 文件不存在"
    fi
done

CLS_SRC="${MODEL_DIR}/classifier/cnn_classifier_fold0.pth"
if [ -f "$CLS_SRC" ]; then
    echo "  上传 classifier..."
    hf upload "$HF_REPO" "$CLS_SRC" "classifier/cnn_classifier_fold0.pth" --commit-message "Add CNN classifier"
fi

echo ""
echo "===== 完成！====="
echo "https://huggingface.co/${HF_REPO}"
