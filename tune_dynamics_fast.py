"""
快速 dynamics 参数调优 —— 只评估 3 张验证图 + 无 augment，在 5 分钟内完成。

用法:
  python tune_dynamics_fast.py --data_dir data/multiclass/fold_0/test \
      --seg_model models/cpsam_v2_fold0/best_model \
      --out_dir models/cpsam_v2_fold0/
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from cellpose import models, io, metrics


def seg_metrics(gt, pred):
    gt_fg, pred_fg = gt > 0, pred > 0
    inter = np.logical_and(gt_fg, pred_fg).sum()
    union = np.logical_or(gt_fg, pred_fg).sum()
    iou = inter / union if union else 1.0
    denom = gt_fg.sum() + pred_fg.sum()
    dice = 2 * inter / denom if denom else 1.0
    miou = 0.5 * (iou + np.logical_and(~gt_fg, ~pred_fg).sum() / (np.logical_or(~gt_fg, ~pred_fg).sum() or 1))
    ap = metrics.average_precision(gt.astype(np.int32), pred.astype(np.int32), threshold=[0.5])[0]
    return iou, dice, miou, float(np.atleast_1d(ap)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--seg_model", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_imgs", type=int, default=3)
    args = parser.parse_args()

    model = models.CellposeModel(gpu=True, pretrained_model=args.seg_model)
    diameter = float(model.net.diam_labels.item())
    print(f"diameter={diameter:.1f}")

    data_dir = Path(args.data_dir)
    img_exts = {".jpg", ".jpeg", ".png"}
    img_files = sorted([f for f in data_dir.iterdir()
                       if f.suffix.lower() in img_exts and "_masks" not in f.stem])
    img_files = img_files[:args.n_imgs]

    eval_imgs, eval_gts = [], []
    for f in img_files:
        eval_imgs.append(io.imread(str(f)))
        g = io.imread(str(f.with_name(f.stem + "_masks.png")))
        if g.ndim == 3:
            g = g[..., 0]
        eval_gts.append(g)
    print(f"eval on {len(eval_imgs)} images")

    flow_vals = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    cp_vals = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

    best_ap = -1.0
    best_ft = 0.4
    best_ct = 0.0
    results = []

    for ft, ct in itertools.product(flow_vals, cp_vals):
        ious, aps = [], []
        for img, gt in zip(eval_imgs, eval_gts):
            masks = model.eval(
                img, diameter=diameter, channels=[0, 0],
                flow_threshold=ft, cellprob_threshold=ct, augment=False)[0]
            iou, _, _, ap50 = seg_metrics(gt, masks)
            ious.append(iou)
            aps.append(ap50)

        m_iou = float(np.mean(ious))
        m_ap = float(np.mean(aps))
        results.append({"flow_threshold": ft, "cellprob_threshold": ct, "iou": m_iou, "ap50": m_ap})

        if m_ap > best_ap:
            best_ap = m_ap
            best_ft = ft
            best_ct = ct
            print(f"  [BEST] ft={ft:.1f} ct={ct:.1f}  IoU={m_iou:.4f} AP@0.5={m_ap:.4f}")
        else:
            print(f"  ft={ft:.1f} ct={ct:.1f}  IoU={m_iou:.4f} AP@0.5={m_ap:.4f}")

    print(f"\nBest: flow_threshold={best_ft}, cellprob_threshold={best_ct}, AP@0.5={best_ap:.4f}")

    out_path = Path(args.out_dir) / "best_dynamics_params.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "best_flow_threshold": best_ft,
            "best_cellprob_threshold": best_ct,
            "best_ap50": best_ap,
            "all_results": results,
        }, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
