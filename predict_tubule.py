"""
肾小管分割 + 近端/远端分类 —— 完整推理 pipeline。

特性:
  - TTA (测试时增强): 内置 augment (tiled flip) + 可选 D4 flow ensemble
  - 多尺度推理: 0.8x / 1.0x / 1.2x diameter
  - CNN 分类器集成: ResNet18 近端/远端分类
  - 可视化: 分割轮廓 + 分类着色 (红=近端, 蓝=远端)
  - 量化分析: 每实例面积、类别、置信度 CSV

用法:
  python predict_tubule.py --img_dir data/test \
      --seg_model models/cpsam_v2_fold0/best_model \
      --cls_model models/classifier/cnn_classifier_fold0.pth \
      --out_dir results/
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models

from cellpose import models as cp_models, io, metrics

SEED = 42
IMG_SIZE = 128

# 可视化颜色 (BGR)
PROXIMAL_COLOR = (0, 0, 255)    # 红色 = 近端小管
DISTAL_COLOR = (255, 0, 0)      # 蓝色 = 远端小管
UNKNOWN_COLOR = (0, 255, 0)     # 绿色 = 未分类


# ---------------------------------------------------------------------------
# CNN 分类器
# ---------------------------------------------------------------------------

class TubuleClassifier:
    """ResNet18 近端/远端分类器。"""

    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def classify_crop(self, crop_bgr):
        """Classify a single BGR crop. Returns (class_id, confidence)."""
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            cls_id = probs.argmax(dim=1).item()
            conf = probs.max(dim=1).values.item()
        return cls_id, conf

    def classify_instances(self, image_bgr, mask, min_area=100):
        """对掩码中每个实例分类，返回 {instance_id: (class_id, confidence)}."""
        h, w = mask.shape
        results = {}
        for inst_id in np.unique(mask):
            if inst_id == 0:
                continue
            inst_mask = (mask == inst_id).astype(np.uint8)
            ys, xs = np.where(inst_mask)
            if len(ys) < min_area:
                results[inst_id] = (-1, 0.0)
                continue

            margin = 20
            y1, y2 = max(0, ys.min() - margin), min(h, ys.max() + margin)
            x1, x2 = max(0, xs.min() - margin), min(w, xs.max() + margin)
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                results[inst_id] = (-1, 0.0)
                continue

            cls_id, conf = self.classify_crop(crop)
            results[inst_id] = (cls_id, conf)
        return results


# ---------------------------------------------------------------------------
# TTA utilities: D4 dihedral group
# ---------------------------------------------------------------------------

def _apply_d4(img, t_idx):
    """Apply D4 transform to HxWxC image."""
    if t_idx == 0:   return img.copy()
    if t_idx == 1:   return np.rot90(img, k=1, axes=(0, 1)).copy()
    if t_idx == 2:   return np.rot90(img, k=2, axes=(0, 1)).copy()
    if t_idx == 3:   return np.rot90(img, k=3, axes=(0, 1)).copy()
    if t_idx == 4:   return img[:, ::-1].copy()
    if t_idx == 5:   return img[::-1, :].copy()
    if t_idx == 6:   return np.rot90(img[:, ::-1], k=1, axes=(0, 1)).copy()
    if t_idx == 7:   return np.rot90(img[::-1, :], k=1, axes=(0, 1)).copy()
    raise ValueError(f"Invalid t_idx: {t_idx}")


def _invert_d4(img, t_idx):
    """Invert D4 transform on HxW array."""
    if t_idx == 0:   return img.copy()
    if t_idx == 1:   return np.rot90(img, k=3, axes=(0, 1)).copy()
    if t_idx == 2:   return np.rot90(img, k=2, axes=(0, 1)).copy()
    if t_idx == 3:   return np.rot90(img, k=1, axes=(0, 1)).copy()
    if t_idx == 4:   return img[:, ::-1].copy()
    if t_idx == 5:   return img[::-1, :].copy()
    if t_idx == 6:   return np.rot90(img, k=3, axes=(0, 1))[:, ::-1].copy()
    if t_idx == 7:   return np.rot90(img, k=3, axes=(0, 1))[::-1, :].copy()
    raise ValueError(f"Invalid t_idx: {t_idx}")


def _invert_flow(flow_y, flow_x, t_idx):
    """Invert flow fields under D4 transform.

    Given a flow field F(p) = displacement at pixel p, and a geometric transform T,
    we have F_T(T(p)) = T(p + F(p)) - T(p) ≈ dT(F(p)).
    This handles the exact inversion for each D4 element.
    """
    flow = np.stack([flow_y, flow_x], axis=-1).astype(np.float32)

    if t_idx == 0:
        pass
    elif t_idx == 1:  # rot90: flow is also rotated
        flow = np.rot90(flow, k=3, axes=(0, 1)).copy()
        flow = flow[..., ::-1]
        flow[..., 1] = -flow[..., 1]
    elif t_idx == 2:  # rot180: negate both components
        flow = np.rot90(flow, k=2, axes=(0, 1)).copy()
        flow = -flow
    elif t_idx == 3:  # rot270
        flow = np.rot90(flow, k=1, axes=(0, 1)).copy()
        flow = flow[..., ::-1]
        flow[..., 0] = -flow[..., 0]
    elif t_idx == 4:  # flip LR: negate x component
        flow = flow[:, ::-1].copy()
        flow[..., 1] = -flow[..., 1]
    elif t_idx == 5:  # flip UD: negate y component
        flow = flow[::-1, :].copy()
        flow[..., 0] = -flow[..., 0]
    elif t_idx == 6:  # flip LR then rot90
        flow = flow[:, ::-1].copy()
        flow[..., 1] = -flow[..., 1]
        flow = np.rot90(flow, k=3, axes=(0, 1)).copy()
        flow = flow[..., ::-1]
        flow[..., 1] = -flow[..., 1]
    elif t_idx == 7:  # flip UD then rot90
        flow = flow[::-1, :].copy()
        flow[..., 0] = -flow[..., 0]
        flow = np.rot90(flow, k=3, axes=(0, 1)).copy()
        flow = flow[..., ::-1]
        flow[..., 1] = -flow[..., 1]

    return flow[..., 0].copy(), flow[..., 1].copy()


# ---------------------------------------------------------------------------
# 分割
# ---------------------------------------------------------------------------

def segment_tta(model, img, diameter, flow_threshold=0.4, cellprob_threshold=0.0,
                tta_transforms=8, multi_scale=None):
    """
    TTA 分割: D4 变换 → 网络推理 → 逆变换 → 平均 flows → dynamics。

    返回 masks (HxW uint16)
    """
    if multi_scale is None:
        multi_scale = [1.0]

    all_flows_y, all_flows_x, all_cellprobs = [], [], []

    for scale in multi_scale:
        scaled_diam = diameter * scale if diameter else None
        for t_idx in range(tta_transforms):
            aug_img = _apply_d4(img, t_idx)

            masks, flows, _ = model.eval(
                aug_img, diameter=scaled_diam, channels=[0, 0],
                flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
                compute_masks=True)

            if len(flows) == 0:
                continue

            if len(flows) < 3:
                continue

            cellprob = flows[2]
            flow_raw = flows[1]
            n_ch = flow_raw.shape[0]

            if n_ch >= 2:
                fy, fx = flow_raw[0], flow_raw[1]
            else:
                continue

            fy_inv, fx_inv = _invert_flow(fy, fx, t_idx)
            cp_inv = _invert_d4(cellprob, t_idx)

            all_flows_y.append(fy_inv)
            all_flows_x.append(fx_inv)
            all_cellprobs.append(cp_inv)

    if not all_flows_y:
        masks, _, _ = model.eval(img, diameter=diameter, channels=[0, 0],
                                flow_threshold=flow_threshold,
                                cellprob_threshold=cellprob_threshold)
        return masks

    avg_fy = np.mean(all_flows_y, axis=0)
    avg_fx = np.mean(all_flows_x, axis=0)
    avg_cp = np.mean(all_cellprobs, axis=0)

    from cellpose.dynamics import compute_masks
    niter = int(200 / (1.0 if diameter is None else max(1, 30.0 / diameter)))
    dP = np.stack([avg_fy, avg_fx], axis=0)
    masks_out = compute_masks(
        dP, avg_cp,
        niter=niter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=15,
    )[0]

    return masks_out.astype(np.uint16)


def segment_simple(model, img, diameter, flow_threshold=0.4, cellprob_threshold=0.0,
                   augment=True):
    """标准分割（cellpose 内置 augment=TTA）。"""
    masks, _, _ = model.eval(
        img, diameter=diameter, channels=[0, 0],
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        augment=augment)
    return masks


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------

def seg_metrics(gt, pred):
    gt_fg = gt > 0
    pred_fg = pred > 0
    inter = np.logical_and(gt_fg, pred_fg).sum()
    union = np.logical_or(gt_fg, pred_fg).sum()
    iou = inter / union if union else 1.0
    denom = gt_fg.sum() + pred_fg.sum()
    dice = 2 * inter / denom if denom else 1.0
    miou = 0.5 * (iou + np.logical_and(~gt_fg, ~pred_fg).sum() / (np.logical_or(~gt_fg, ~pred_fg).sum() or 1))
    ap = metrics.average_precision(gt.astype(np.int32), pred.astype(np.int32),
                                   threshold=[0.5])[0]
    return iou, dice, miou, float(np.atleast_1d(ap)[0])


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def visualize_results(image_bgr, mask, class_results, out_path):
    """分割轮廓 + 分类着色 (红=近端, 蓝=远端, 绿=未分类)。"""
    vis = image_bgr.copy()
    overlay = np.zeros_like(vis)

    for inst_id in np.unique(mask):
        if inst_id == 0:
            continue
        cls_id, conf = class_results.get(inst_id, (-1, 0))

        if cls_id == 0:
            color = PROXIMAL_COLOR
        elif cls_id == 1:
            color = DISTAL_COLOR
        else:
            color = UNKNOWN_COLOR

        inst_mask = (mask == inst_id).astype(np.uint8)
        contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
        overlay[inst_mask > 0] = color

        if len(contours) > 0:
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                label = f"{'P' if cls_id == 0 else 'D' if cls_id == 1 else '?'}{inst_id}"
                cv2.putText(vis, label, (cx - 8, cy + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    alpha = 0.35
    vis_blend = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)
    cv2.putText(vis_blend, "Red=Proximal  Blue=Distal  Green=Unknown",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.imwrite(str(out_path), vis_blend)
    return vis_blend


def write_results_csv(mask, class_results, pixel_size_um, out_path):
    """写出每实例量化分析 CSV。"""
    pixel_area_um2 = pixel_size_um ** 2
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["instance_id", "class", "class_id", "confidence",
                        "area_pixels", "area_um2",
                        "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
        for inst_id in np.unique(mask):
            if inst_id == 0:
                continue
            cls_id, conf = class_results.get(inst_id, (-1, 0))
            cls_name = {0: "proximal", 1: "distal"}.get(cls_id, "unknown")
            inst_mask = (mask == inst_id).astype(np.uint8)
            area = inst_mask.sum()
            ys, xs = np.where(inst_mask)
            writer.writerow([inst_id, cls_name, cls_id, f"{conf:.4f}",
                            area, area * pixel_area_um2,
                            xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()])


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="肾小管分割 + 分类推理")
    parser.add_argument("--img", default=None, help="单张输入图")
    parser.add_argument("--img_dir", default=None, help="批量处理目录")
    parser.add_argument("--seg_model", required=True, help="CPSAM 模型路径")
    parser.add_argument("--cls_model", default=None, help="CNN 分类器路径")
    parser.add_argument("--out_dir", default="results", help="输出目录")
    parser.add_argument("--diameter", type=float, default=None,
                       help="肾小管直径(像素), None=模型默认")
    parser.add_argument("--flow_threshold", type=float, default=0.4)
    parser.add_argument("--cellprob_threshold", type=float, default=0.0)
    parser.add_argument("--tta", type=int, default=0, choices=[0, 4, 8],
                       help="D4 TTA 数量 (0=内置 augment 模式, 4/8=D4 flow ensemble)")
    parser.add_argument("--multi_scale", type=str, default=None,
                       help="多尺度因子, 逗号分隔, 如 '0.8,1.0,1.2'")
    parser.add_argument("--pixel_size", type=float, default=0.5,
                       help="像素尺寸 (um/pixel)")
    parser.add_argument("--no_classify", action="store_true", help="跳过分类")
    parser.add_argument("--no_augment", action="store_true", help="禁用内置 augment")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f">>> device = {device}")

    # 加载模型
    print(f">>> 加载分割模型: {args.seg_model}")
    model = cp_models.CellposeModel(gpu=use_gpu, pretrained_model=args.seg_model, device=device)

    diameter = args.diameter or float(model.net.diam_labels.item())
    print(f">>> diameter = {diameter:.1f}px")

    # 分类器
    classifier = None
    if args.cls_model and not args.no_classify:
        print(f">>> 加载分类器: {args.cls_model}")
        classifier = TubuleClassifier(args.cls_model, device)

    # 收集图像
    img_paths = []
    if args.img:
        img_paths = [Path(args.img)]
    if args.img_dir:
        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            img_paths.extend(sorted(Path(args.img_dir).glob(f"*{ext}")))
    img_paths = [p for p in img_paths if "_masks" not in p.stem and "_flows" not in p.stem]

    if not img_paths:
        print("未找到图像！")
        return

    multi_scale = None
    if args.multi_scale:
        multi_scale = [float(s) for s in args.multi_scale.split(",")]

    all_metrics = []
    all_summaries = []

    for img_path in img_paths:
        t0 = time.time()
        print(f"\n>>> 处理: {img_path.name}")

        img = io.imread(str(img_path))
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        # 分割
        if args.tta > 0:
            print(f"  TTA={args.tta}, multi_scale={multi_scale}")
            mask = segment_tta(model, img, diameter,
                              flow_threshold=args.flow_threshold,
                              cellprob_threshold=args.cellprob_threshold,
                              tta_transforms=args.tta,
                              multi_scale=multi_scale or [1.0])
        else:
            mask = segment_simple(model, img, diameter,
                                 flow_threshold=args.flow_threshold,
                                 cellprob_threshold=args.cellprob_threshold,
                                 augment=not args.no_augment)

        n_inst = len(np.unique(mask)) - 1
        seg_time = time.time() - t0
        print(f"  检测到 {n_inst} 个肾小管 ({seg_time:.1f}s)")

        # GT 指标
        gt_path = img_path.with_name(img_path.stem + "_masks.png")
        if gt_path.exists():
            gt = io.imread(str(gt_path))
            if gt.ndim == 3:
                gt = gt[..., 0]
            iou, dice, miou, ap50 = seg_metrics(gt, mask)
            all_metrics.append((iou, dice, miou, ap50))
            print(f"  IoU={iou:.4f} Dice={dice:.4f} mIoU={miou:.4f} AP@0.5={ap50:.4f}")

        # 分类
        class_results = {}
        img_bgr = cv2.imread(str(img_path))
        if classifier and n_inst > 0:
            class_results = classifier.classify_instances(img_bgr, mask)
            n_p = sum(1 for c, _ in class_results.values() if c == 0)
            n_d = sum(1 for c, _ in class_results.values() if c == 1)
            print(f"  近端: {n_p}, 远端: {n_d}")

        # 可视化
        vis_path = out_dir / f"{img_path.stem}_classified.png"
        visualize_results(img_bgr, mask, class_results, vis_path)

        # CSV
        csv_path = out_dir / f"{img_path.stem}_analysis.csv"
        write_results_csv(mask, class_results, args.pixel_size, csv_path)

        all_summaries.append({
            "image": img_path.name,
            "n_instances": n_inst,
            "n_proximal": sum(1 for c, _ in class_results.values() if c == 0),
            "n_distal": sum(1 for c, _ in class_results.values() if c == 1),
            "time_s": time.time() - t0,
        })

    # 汇总
    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    if all_metrics:
        arr = np.array(all_metrics)
        m = arr.mean(axis=0)
        s = arr.std(axis=0)
        print(f"\n{'='*60}")
        print(f">>> 测试集 ({len(all_metrics)} 张)")
        print(f"  IoU:     {m[0]:.4f} ± {s[0]:.4f}")
        print(f"  Dice:    {m[1]:.4f} ± {s[1]:.4f}")
        print(f"  mIoU:    {m[2]:.4f} ± {s[2]:.4f}")
        print(f"  AP@0.5:  {m[3]:.4f} ± {s[3]:.4f}")

    print(f"\n>>> 输出到 {out_dir}")


if __name__ == "__main__":
    main()
