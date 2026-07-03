# Kidney Tubule Segmentation & Classification

基于 CellPose + CPSAM 的肾小管自动分割与近端/远端分类。

## 功能

1. **肾小管分割** — CPSAM (SAM ViT-L) 微调，5-fold CV IoU 0.844
2. **近端/远端分类** — ResNet18 CNN，per-instance 分类
3. **批量推理** — 自动输出分类可视化 PNG + 逐实例 CSV
4. **TTA 推理** — D4 二面体变换 + flow 翻转增强

## 性能

| 指标 | 值 |
|------|------|
| 5-fold CV IoU | 0.844 |
| 5-fold CV Dice | 0.914 |
| 5-fold CV AP@0.5 | 0.845 ± 0.017 |
| 最佳单图 AP@0.5 | 0.964 (Captured 32) |

## 环境要求

```bash
conda create -n cellpose python=3.10
conda activate cellpose
pip install cellpose==4.2.1 torch==2.5.1 opencv-python numpy
```

GPU 推理推荐 NVIDIA GPU (≥8GB VRAM)，CPU 也可运行但较慢。

## 模型权重

模型权重通过 Git LFS 管理，或从 [Releases](../../releases) 下载。

```
models/
├── cpsam_v2_fold0/best_model      # 分割模型 fold 0
├── cpsam_v2_fold1/best_model      # 分割模型 fold 1
├── cpsam_v2_fold2/best_model      # 分割模型 fold 2
├── cpsam_v2_fold3/best_model      # 分割模型 fold 3
├── cpsam_v2_fold4/best_model      # 分割模型 fold 4
└── classifier/
    └── cnn_classifier_fold0.pth   # 近端/远端分类器
```

## 数据准备

标注格式为 GeoJSON (class 0=proximal, 1=distal, 2=other)：

```
data/
└── raw/
    ├── image_001.jpg
    ├── image_001.geojson
    ├── image_002.jpg
    ├── image_002.geojson
    └── ...
```

生成训练掩码和 5-fold 划分：

```bash
python prepare_multiclass_data.py \
    --geojson_dir data/raw \
    --img_dir data/raw \
    --out_dir data/multiclass \
    --n_folds 5
```

## 训练

### 1. 分割模型（5-fold CV）

```bash
for FOLD in 0 1 2 3 4; do
    python -m cellpose \
        --dir data/multiclass/fold_${FOLD}/train \
        --pretrained_model cpsam \
        --chan 0 --chan2 0 \
        --learning_rate 0.0001 \
        --n_epochs 200 \
        --save_path models/cpsam_v2_fold${FOLD}
done
```

### 2. 分类器

```bash
python classify_cnn.py \
    --img_dir data/multiclass/all \
    --mask_dir data/multiclass/all \
    --model_path models/classifier/cnn_classifier_fold0.pth
```

### 3. 动态参数调优

```bash
python tune_dynamics.py \
    --model_path models/cpsam_v2_fold0/best_model \
    --data_dir data/multiclass/fold_0/test
```

推荐参数：`flow_threshold=0.3`, `cellprob_threshold=-0.5`

## 推理

### 全 Pipeline（分割 + 分类）

```bash
python predict_tubule.py \
    --img_dir data/test_images \
    --seg_model models/cpsam_v2_fold0/best_model \
    --cls_model models/classifier/cnn_classifier_fold0.pth \
    --out_dir results/output \
    --flow_threshold 0.3 \
    --cellprob_threshold -0.5
```

### 5-fold 一键评估

```bash
bash run_folds.sh
```

### 输出文件

每张输入图生成：

| 文件 | 内容 |
|------|------|
| `*_classified.png` | 分割+分类叠加图 (红=近端, 蓝=远端) |
| `*_analysis.csv` | 每实例: class, confidence, area_pixels, area_um2, bbox |
| `summary.json` | 该 fold 汇总统计 |

CSV 字段：

```csv
instance_id,class,class_id,confidence,area_pixels,area_um2,bbox_x,bbox_y,bbox_w,bbox_h
```

## 目录结构

```
kidney_tubule_seg/
├── README.md
├── prepare_multiclass_data.py   # GeoJSON → 掩码 + 数据划分
├── train_tubule.py              # 分割模型训练
├── classify_cnn.py              # CNN 分类器训练
├── classify_tubules.py          # 分类器推理接口
├── predict_tubule.py            # 主推理脚本 (分割+分类+可视化)
├── eval_ensemble.py             # 5-fold CV 评估
├── tune_dynamics.py             # flow/cellprob 参数调优
├── tune_dynamics_fast.py        # 快速参数扫描
├── postprocess_masks.py         # 掩码后处理
├── prepare_tubule_data.py       # 备选数据准备
├── run_folds.sh                 # 一键 5-fold Pipeline
└── run_full_pipeline.sh         # 完整 Pipeline (含 ensemble)
```

## 常见问题

**Q: Cellpose v4 `model.eval()` 返回格式变了？**
A: v4 单图返回 `(H,W)` ndarray，不是 `[(H,W)]` 列表。`predict_tubule.py` 已适配。

**Q: `_flows` 文件被误读？**
A: 确保图片 glob 过滤 `_flows` 和 `_masks` 文件。

**Q: CPU 推理太慢？**
A: CPSAM ViT-L 在 CPU 上很慢。建议用 GPU，或将图片 resize 到 512px。
