#!/bin/bash
# 在服务器上运行：上传模型权重到 Hugging Face
# 用法:
#   export HF_TOKEN="hf_xxx"
#   bash upload_models.sh
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
cd "$MODEL_DIR"

# 上传全部模型到 HF
hf upload "$HF_REPO" . . \
    --include "cpsam_v2_fold*/best_model" \
    --include "classifier/*.pth" \
    --message "Upload 5-fold CPSAM models + CNN classifier"

echo ""
echo "===== 完成！ ====="
echo "模型地址: https://huggingface.co/${HF_REPO}"
echo ""
echo "他人下载方式:"
echo "  hf download ${HF_REPO} --local-dir models/"
