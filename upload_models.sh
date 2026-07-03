#!/bin/bash
# 在服务器上运行：上传模型权重到 GitHub (Git LFS)
# 用法: bash upload_models.sh
set -e

# 先在终端执行: export GITHUB_TOKEN="你的token"
# 或者直接运行: GITHUB_TOKEN="ghp_xxx" bash upload_models.sh
if [ -z "$GITHUB_TOKEN" ]; then
    echo "错误: 请设置 GITHUB_TOKEN 环境变量"
    echo "用法: GITHUB_TOKEN='ghp_xxx' bash upload_models.sh"
    exit 1
fi
REPO_URL="https://MEIPork1:${GITHUB_TOKEN}@github.com/MEIPork1/kidney_tubule_seg.git"
MODEL_DIR="/root/kidney/models"
TMP_DIR="/tmp/kidney_models_upload"

echo "===== 1. 安装 git-lfs ====="
which git-lfs || (apt-get update && apt-get install -y git-lfs)
git lfs install

echo "===== 2. 克隆仓库 ====="
rm -rf "$TMP_DIR"
git clone "$REPO_URL" "$TMP_DIR"
cd "$TMP_DIR"

echo "===== 3. 复制模型权重 ====="
mkdir -p models/classifier
for FOLD in 0 1 2 3 4; do
    SRC="${MODEL_DIR}/cpsam_v2_fold${FOLD}/best_model"
    DST="models/cpsam_v2_fold${FOLD}/best_model"
    if [ -f "$SRC" ]; then
        mkdir -p "models/cpsam_v2_fold${FOLD}"
        cp "$SRC" "$DST"
        echo "  Copied fold${FOLD}: $(du -h "$DST" | cut -f1)"
    else
        echo "  WARNING: $SRC not found"
    fi
done

# CNN 分类器
CLS_SRC="${MODEL_DIR}/classifier/cnn_classifier_fold0.pth"
if [ -f "$CLS_SRC" ]; then
    cp "$CLS_SRC" "models/classifier/"
    echo "  Copied classifier: $(du -h "$CLS_SRC" | cut -f1)"
fi

echo "===== 4. Git LFS 追踪 + 提交 ====="
git lfs track "models/**"
git add .gitattributes models/
git commit -m "Add model weights via Git LFS

- 5x CPSAM fine-tuned segmentation models (fold 0-4)
- 1x ResNet18 CNN classifier (proximal/distal)
- 5-fold CV IoU: 0.844, AP@0.5: 0.845"
git push origin main

echo "===== 完成！ ====="
echo "模型已上传到: https://github.com/MEIPork1/kidney_tubule_seg"
