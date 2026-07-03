"""
CellPose dynamics 参数调优 —— 在验证集上网格搜索 flow_threshold & cellprob_threshold。

原理:
  flow_threshold (默认 0.4): 控制 flow following 的误差容忍度
    - 更高 → 更少的假阳性，但可能丢失小管
    - 更低 → 更多检测，但可能有噪声
  cellprob_threshold (默认 0.0): 控制 cell probability 的阈值
    - 更高 → 只保留高置信度区域
    - 更低 → 更多候选区域

用法:
  python tune_dynamics.py --data_dir data/multiclass/fold_0/test \
      --seg_model models/cpsam_v2_fold0/best_model \
      --out_dir models/cpsam_v2_fold0/
"""

import argparse
import json
import itertools
from pathlib import Path

import numpy as np
import torch

from cellpose import models, io, metrics

io.logger_setup()


def seg_metrics(gt, pred):
    gt_fg, pred_fg = gt > 0, pred > 0
    inter = np.logical_and(gt_fg, pred_fg).sum()
    union = np.logical_or(gt_fg, pred_fg).sum()
    iou = inter / union if union else 1.0
    denom = gt_fg.sum() + pred_fg.sum()
    dice = 2 * inter / denom if denom else 1.0
    miou = 0.5 * (iou + np.logical_and(~gt_fg, ~pred_fg).sum() / (np.logical_or(~gt_fg, ~pred_fg).sum() or 1))
    ap = metrics.average_precision(gt.astype(np.int32), pred.astype(np.int32),
                                   threshold=[0.5])[0]
    return iou, dice, miou, float(np.atleast_1d(ap)[0])


def evaluate_params(model, eval_imgs, eval_gts, diameter, flow_th, cp_th):
    ious, dices, mious, aps = [], [], [], []
    for img, gt in zip(eval_imgs, eval_gts):
        masks = model.eval(img, diameter=diameter, channels=[0, 0],
                          flow_threshold=flow_th, cellprob_threshold=cp_th,
                          augment=True)[0]
        iou, dice, miou, ap50 = seg_metrics(gt, masks)
        ious.append(iou)
        dices.append(dice)
        mious.append(miou)
        aps.append(ap50)
    return np.mean(ious), np.mean(dices), np.mean(mious), np.mean(aps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--seg_model", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--diameter", type=float, default=None)
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    model = models.CellposeModel(gpu=use_gpu, pretrained_model=args.seg_model, device=device)
    diameter = args.diameter or float(model.net.diam_labels.item())

    # 加载验证集
    data_dir = Path(args.data_dir)
    img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    img_files = sorted([f for f in data_dir.iterdir()
                       if f.suffix.lower() in img_exts and "_masks" not in f.stem])

    eval_imgs, eval_gts = [], []
    for f in img_files:
        gt_path = f.with_name(f.stem + "_masks.png")
        if not gt_path.exists():
            continue
        eval_imgs.append(io.imread(str(f)))
        g = io.imread(str(gt_path))
        if g.ndim == 3:
            g = g[..., 0]
        eval_gts.append(g)

    print(f">>> 验证集: {len(eval_imgs)} 张图, diameter={diameter:.1f}")

    # 网格搜索
    flow_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    cellprob_thresholds = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

    best_ap = -1
    best_params = None
    results = []

    total = len(flow_thresholds) * len(cellprob_thresholds)
    print(f">>> 搜索 {total} 个组合...")

    for i, (ft, ct) in enumerate(itertools.product(flow_thresholds, cellprob_thresholds)):
        iou, dice, miou, ap50 = evaluate_params(model, eval_imgs, eval_gts, diameter, ft, ct)
        results.append({
            "flow_threshold": ft,
            "cellprob_threshold": ct,
            "iou": float(iou),
            "dice": float(dice),
            "miou": float(miou),
            "ap50": float(ap50),
        })
        flag = ""
        if ap50 > best_ap:
            best_ap = ap50
            best_params = (ft, ct)
            flag = " <-- BEST"
        print(f"  [{i+1:2d}/{total}] flow={ft:.1f} cp={ct:.1f}  "
              f"IoU={iou:.4f} AP@0.5={ap50:.4f}{flag}")

    # 输出
    print(f"\n>>> 最佳参数: flow_threshold={best_params[0]:.1f}, "
          f"cellprob_threshold={best_params[1]:.1f}  (AP@0.5={best_ap:.4f})")

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.seg_model).parent
    out_path = out_dir / "best_dynamics_params.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "best_flow_threshold": best_params[0],
            "best_cellprob_threshold": best_params[1],
            "best_ap50": best_ap,
            "all_results": results,
        }, f, indent=2)
    print(f">>> 参数保存到 {out_path}")


if __name__ == "__main__":
    main()
