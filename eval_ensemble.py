"""
5-fold 交叉验证评估 —— 每模型独立评估，不做 mask 合并。

每个 fold 模型在所有 46 张测试图上独立推理，报告 per-model 和均值指标。

用法:
  python eval_ensemble.py --data_dir data/multiclass/all \
      --model_dir models/cpsam_v2_fold \
      --out_dir results/ensemble/
"""

import argparse
import json
import time
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/root/kidney/data/multiclass/all")
    parser.add_argument("--model_dir", default="/root/kidney/models/cpsam_v2_fold")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--out_dir", default="/root/kidney/results/ensemble")
    parser.add_argument("--flow_threshold", type=float, default=0.4)
    parser.add_argument("--cellprob_threshold", type=float, default=0.0)
    parser.add_argument("--diameter", type=float, default=None)
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f">>> device = {device}")

    data_dir = Path(args.data_dir)
    img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    img_files = sorted([f for f in data_dir.iterdir()
                       if f.suffix.lower() in img_exts and "_masks" not in f.stem])

    # 预加载所有 GT
    gts = {}
    for img_path in img_files:
        gt_path = img_path.with_name(img_path.stem + "_masks.png")
        if gt_path.exists():
            gt = io.imread(str(gt_path))
            if gt.ndim == 3:
                gt = gt[..., 0]
            gts[img_path.name] = gt

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_fold_results = []
    t0 = time.time()

    for fold in range(args.n_folds):
        model_path = f"{args.model_dir}{fold}/best_model"
        if not Path(model_path).exists():
            print(f"[警告] 模型不存在: {model_path}")
            continue

        print(f"\n--- Fold {fold} ---")
        model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_path, device=device)
        diameter = args.diameter or float(model.net.diam_labels.item())

        fold_metrics = []
        for img_path in img_files:
            img = io.imread(str(img_path))
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)

            t_img = time.time()
            masks, _, _ = model.eval(
                img, diameter=diameter, channels=[0, 0],
                flow_threshold=args.flow_threshold,
                cellprob_threshold=args.cellprob_threshold,
                augment=True)
            # Cellpose v4: 单张图直接返回 (H,W) 数组

            n_inst = len(np.unique(masks)) - 1
            gt = gts.get(img_path.name)
            if gt is not None:
                iou, dice, miou, ap50 = seg_metrics(gt, masks)
                fold_metrics.append({
                    "image": img_path.name, "n_inst": n_inst,
                    "iou": float(iou), "dice": float(dice),
                    "miou": float(miou), "ap50": float(ap50),
                    "time_s": time.time() - t_img,
                })

        if fold_metrics:
            arr = np.array([[m["iou"], m["dice"], m["miou"], m["ap50"]] for m in fold_metrics])
            m = arr.mean(axis=0)
            print(f"  {len(fold_metrics)} 张图 | IoU={m[0]:.4f} Dice={m[1]:.4f} mIoU={m[2]:.4f} AP@0.5={m[3]:.4f}")
            all_fold_results.append({"fold": fold, "metrics": fold_metrics, "mean": m.tolist()})
        else:
            print(f"  无有效结果")

    # 汇总
    if all_fold_results:
        per_fold_means = np.array([r["mean"] for r in all_fold_results])
        overall_mean = per_fold_means.mean(axis=0)
        overall_std = per_fold_means.std(axis=0)
        labels = ["IoU", "Dice", "mIoU", "AP@0.5"]

        print(f"\n{'='*60}")
        print(f">>> 5-fold CV 汇总 ({len(all_fold_results)} folds x {len(img_files)} 张图)")
        for i, label in enumerate(labels):
            print(f"  {label}:     {overall_mean[i]:.4f} +- {overall_std[i]:.4f}")
        print(f"  Time:    {time.time() - t0:.0f}s")

        with open(out_dir / "ensemble_results.json", "w") as f:
            json.dump({
                "n_folds": len(all_fold_results),
                "n_images": len(img_files),
                "flow_threshold": args.flow_threshold,
                "cellprob_threshold": args.cellprob_threshold,
                "overall_mean": {labels[i]: float(overall_mean[i]) for i in range(4)},
                "overall_std": {labels[i]: float(overall_std[i]) for i in range(4)},
                "per_fold": [{"fold": r["fold"], "mean": {labels[i]: r["mean"][i] for i in range(4)}} for r in all_fold_results],
            }, f, indent=2)

    print(f">>> 输出到 {out_dir}")


if __name__ == "__main__":
    main()
