"""
近端/远端肾小管分类器 —— Stage 2 of the two-stage pipeline.

输入：分割后的实例掩码 + 原图 + class_map (GT)
输出：每个实例的近端/远端预测

方法：组合特征
  1. 形态学特征 (area, perimeter, solidity, eccentricity, intensity stats)
  2. 纹理特征 (GLCM contrast/homogeneity, edge density for brush border detection)
  3. CNN 特征 (ResNet18 on cropped ROI) — 可选

用法：
  python classify_tubules.py --data_dir /root/kidney/data/multiclass/all --model_dir models/cpsam_improved_fold0
"""

import argparse
import csv
import json
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
from scipy import ndimage
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline

import torch
import torch.nn as nn
import torchvision.models as models

# ----------------- 配置 -----------------
N_FOLDS = 5
SEED = 42
# ----------------------------------------


def extract_morphological_features(mask_instance, image_gray):
    """
    提取单个肾小管实例的形态学和纹理特征。

    特征维度 ~30:
    - 面积、周长、等效直径
    - 坚固度 (solidity)、偏心率 (eccentricity)、延展度 (extent)
    - 最小/最大 Feret 直径近似
    - 平均强度、标准差、偏度、峰度
    - 管腔 vs 管壁的强度对比（刷状缘检测）
    - 边缘密度（刷状缘 proxy）
    - 核位置特征（管腔 vs 细胞质 vs 边缘）
    """
    mask = mask_instance.astype(np.uint8)
    if mask.sum() < 20:
        return None

    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return None

    # 基本形状
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    x, y, bw, bh = cv2.boundingRect(cnt)
    aspect_ratio = bw / bh if bh > 0 else 0
    extent = area / (bw * bh) if bw * bh > 0 else 0
    equiv_diameter = np.sqrt(4 * area / np.pi)

    # 中心矩
    moments = cv2.moments(cnt)
    if moments['mu20'] + moments['mu02'] > 0:
        eccentricity = np.sqrt(1 - (4 * moments['mu11']**2 /
                                    ((moments['mu20'] + moments['mu02'])**2 + 1e-8)))
    else:
        eccentricity = 0

    # 强度特征
    masked_pixels = image_gray[mask > 0].astype(np.float32)
    if len(masked_pixels) < 10:
        return None

    mean_intensity = masked_pixels.mean()
    std_intensity = masked_pixels.std()
    min_intensity = masked_pixels.min()
    max_intensity = masked_pixels.max()
    skew_intensity = skew(masked_pixels) if len(masked_pixels) > 10 else 0
    kurt_intensity = kurtosis(masked_pixels) if len(masked_pixels) > 10 else 0

    # 管腔检测（远端小管管腔更宽大）
    # Erode mask to get inner region (lumen), dilate to get outer region
    kernel_size = max(3, int(equiv_diameter * 0.05))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(mask, kernel, iterations=3)
    dilated = cv2.dilate(mask, kernel, iterations=3)
    border_region = cv2.subtract(dilated, eroded)

    lumen_pixels = image_gray[(eroded > 0) & (mask > 0)] if eroded.sum() > 0 else np.array([0])
    wall_pixels = image_gray[border_region > 0] if border_region.sum() > 0 else np.array([0])
    lumen_intensity = lumen_pixels.mean() if len(lumen_pixels) > 0 else 0
    wall_intensity = wall_pixels.mean() if len(wall_pixels) > 0 else 0
    lumen_wall_ratio = lumen_intensity / (wall_intensity + 1e-8)

    # 边缘密度 (brush border proxy — 近端小管刷状缘边缘更多)
    edges = cv2.Canny(mask * 255, 50, 150)
    edge_density = edges.sum() / (area + 1e-8)

    # 内部纹理 (GLCM 近似)
    # 用小 patch 的梯度强度作为纹理复杂度 proxy
    grad_x = cv2.Sobel(image_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    interior_grad = grad_mag[mask > 0].mean() if mask.sum() > 0 else 0
    interior_grad_std = grad_mag[mask > 0].std() if mask.sum() > 0 else 0

    # 邻近区域强度（间质 vs 肾小管对比）
    dilated_big = cv2.dilate(mask, kernel, iterations=8)
    surrounding = cv2.subtract(dilated_big, mask)
    surr_pixels = image_gray[surrounding > 0] if surrounding.sum() > 0 else np.array([0])
    surr_intensity = surr_pixels.mean() if len(surr_pixels) > 0 else 0
    tubule_surr_ratio = mean_intensity / (surr_intensity + 1e-8)

    features = np.array([
        area, perimeter, circularity, solidity, aspect_ratio, extent,
        equiv_diameter, eccentricity,
        mean_intensity, std_intensity, min_intensity, max_intensity,
        skew_intensity, kurt_intensity,
        lumen_intensity, wall_intensity, lumen_wall_ratio,
        edge_density, interior_grad, interior_grad_std,
        surr_intensity, tubule_surr_ratio,
        bw, bh,  # 边界框宽高
    ], dtype=np.float32)

    # 检查 NaN
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def extract_features_for_image(image_path, mask_path, class_csv_path=None):
    """
    从一张图和它的掩码中提取所有实例的特征。
    返回 (features_array, instance_ids, class_labels)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None, None, None

    # 加载类别标签
    class_map = {}
    if class_csv_path and Path(class_csv_path).exists():
        with open(class_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_map[int(row["instance_id"])] = int(row["class_id"])

    features_list = []
    ids_list = []
    labels_list = []

    inst_ids = np.unique(mask)
    inst_ids = inst_ids[inst_ids > 0]

    for inst_id in inst_ids:
        inst_mask = (mask == inst_id).astype(np.uint8)
        feats = extract_morphological_features(inst_mask, gray)
        if feats is None:
            continue
        features_list.append(feats)
        ids_list.append(inst_id)
        if class_map:
            labels_list.append(class_map.get(inst_id, -1))

    if not features_list:
        return None, None, None

    return np.stack(features_list, axis=0), ids_list, labels_list if class_map else None


def train_morphological_classifier(data_dir, model_dir, n_folds=N_FOLDS):
    """
    基于形态学特征训练分类器 + 交叉验证。
    """
    data_path = Path(data_dir)
    img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    all_features = []
    all_labels = []

    img_files = [f for f in sorted(data_path.iterdir())
                 if f.suffix.lower() in img_exts and "_masks" not in f.stem]

    for img_path in img_files:
        stem = img_path.stem
        mask_path = img_path.with_name(f"{stem}_masks.png")
        csv_path = img_path.with_name(f"{stem}_class_map.csv")

        if not mask_path.exists():
            continue

        feats, ids_list, labels = extract_features_for_image(img_path, mask_path, csv_path)
        if feats is not None and labels is not None:
            # 过滤 class 2 (罕见)
            for f, l in zip(feats, labels):
                if l in (0, 1):  # 只保留 proximal & distal
                    all_features.append(f)
                    all_labels.append(l)

    X = np.stack(all_features, axis=0)
    y = np.array(all_labels)
    print(f">>> 提取了 {len(y)} 个实例: proximal={sum(y==0)}, distal={sum(y==1)}")

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    classifiers = {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=10,
                                           class_weight="balanced", random_state=SEED))
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                               learning_rate=0.05, random_state=SEED))
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=SEED))
        ]),
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, pipe in classifiers.items():
        scores = cross_val_score(pipe, X, y, cv=skf, scoring="f1")
        print(f"  {name}: F1 = {scores.mean():.4f} ± {scores.std():.4f}")

        # 完整训练评估
        y_pred_cv = []
        y_prob_cv = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            pipe.fit(X_train, y_train)
            y_pred_cv.extend(pipe.predict(X_test))
            if hasattr(pipe.named_steps["clf"], "predict_proba"):
                y_prob_cv.extend(pipe.named_steps["clf"].predict_proba(
                    pipe.named_steps["scaler"].transform(X_test))[:, 1])

        acc = accuracy_score(y, y_pred_cv)
        f1 = f1_score(y, y_pred_cv)
        roc = roc_auc_score(y, y_prob_cv) if y_prob_cv else 0
        print(f"    CV Accuracy={acc:.4f}, F1={f1:.4f}, ROC-AUC={roc:.4f}")
        print(f"    Confusion:\n{confusion_matrix(y, y_pred_cv)}")

        if f1 > best_score:
            best_score = f1
            best_model = pipe
            best_name = name

    # 在全部数据上训练最佳模型
    best_model.fit(X, y)

    # 保存模型
    model_path = Path(model_dir) / "classifier_morph.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"model": best_model, "name": best_name,
                     "feature_names": [f"feat_{i}" for i in range(X.shape[1])],
                     "n_features": X.shape[1], "classes": ["proximal", "distal"]}, f)
    print(f">>> 最佳模型: {best_name} (F1={best_score:.4f}) 保存到 {model_path}")

    # 特征重要性
    if hasattr(best_model.named_steps.get("clf", None), "feature_importances_"):
        importances = best_model.named_steps["clf"].feature_importances_
        feat_names = [f"feat_{i}" for i in range(X.shape[1])]
        top = np.argsort(importances)[::-1][:10]
        print(">>> Top 10 features:")
        for i in top:
            print(f"    {feat_names[i]}: {importances[i]:.4f}")

    return best_model, best_name, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/root/kidney/data/multiclass/all")
    parser.add_argument("--model_dir", default="models/classifier")
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 60)
    print(">>> 训练近端/远端肾小管分类器")
    print("=" * 60)

    train_morphological_classifier(args.data_dir, args.model_dir)


if __name__ == "__main__":
    main()
