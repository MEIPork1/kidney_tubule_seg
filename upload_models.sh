#!/bin/bash
# 上传模型权重到 Hugging Face（通过国内镜像 hf-mirror.com）
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

echo "===== 上传模型到 Hugging Face (via hf-mirror.com) ====="

/root/miniconda3/envs/cellpose/bin/python -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import HfApi, login
login(token='${HF_TOKEN}')
api = HfApi(endpoint='https://hf-mirror.com')

# 上传 CPSAM 最佳模型 (fold 3)
api.upload_file(
    path_or_fileobj='${MODEL_DIR}/cpsam_v2_fold3/best_model',
    path_in_repo='cpsam_v2_fold3/best_model',
    repo_id='${HF_REPO}',
    commit_message='Add best CPSAM model (fold 3, IoU=0.858, AP=0.871)'
)

# 上传 CNN 分类器
api.upload_file(
    path_or_fileobj='${MODEL_DIR}/classifier/cnn_classifier_fold0.pth',
    path_in_repo='classifier/cnn_classifier_fold0.pth',
    repo_id='${HF_REPO}',
    commit_message='Add CNN classifier (ResNet18)'
)

print('All done!')
"

echo ""
echo "===== 完成！====="
echo "https://huggingface.co/${HF_REPO}"
