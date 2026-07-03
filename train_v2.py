"""
简洁版 CPSAM 训练 —— 在原版成功配置上叠加改进。

与 train_tubule.py (Run 2) 一致的核心设置：
  - same data pipeline / diam rescaling / bsize=256 / batch=1
  - same MSE+BCE loss

增量改进：
  - Cosine annealing LR + warmup (更稳收敛)
  - 50% more epochs (300 vs 200)
  - AMP mixed precision (更快)
  - 保存 best model + 最后 model
"""

import argparse
import time
import json
from pathlib import Path

import numpy as np
import torch

from cellpose import models, io, metrics
from cellpose.train import (_process_train_test, _get_batch,
                            _loss_fn_seg, _loss_fn_class)
from cellpose.transforms import random_rotate_and_resize

io.logger_setup()

ROOT = Path("/root/kidney")

# ----------------- 超参 (对齐原版 Run 2) -----------------
N_EPOCHS = 300
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.1
BATCH_SIZE = 1
BSIZE = 256
MIN_TRAIN_MASKS = 3
WARMUP_EPOCHS = 10
EVAL_EVERY = 10
SAVE_EVERY = 50
EVAL_BATCH_SIZE = 8
SCALE_RANGE = 0.5            # 原版值
# -------------------------------------------------------


def seg_metrics(gt, pred):
    gt_fg, pred_fg = gt > 0, pred > 0
    inter = np.logical_and(gt_fg, pred_fg).sum()
    union = np.logical_or(gt_fg, pred_fg).sum()
    iou = inter / union if union else 1.0
    denom = gt_fg.sum() + pred_fg.sum()
    dice = 2 * inter / denom if denom else 1.0
    ib = np.logical_and(~gt_fg, ~pred_fg).sum()
    ub = np.logical_or(~gt_fg, ~pred_fg).sum()
    miou = 0.5 * (iou + (ib / ub if ub else 1.0))
    ap = metrics.average_precision(gt.astype(np.int32), pred.astype(np.int32),
                                   threshold=[0.5])[0]
    return iou, dice, miou, float(np.atleast_1d(ap)[0])


def evaluate(model, eval_imgs, eval_gts, diameter):
    net_was_training = model.net.training
    model.net.eval()
    ious, dices, mious, aps = [], [], [], []
    if model.device.type == "cuda":
        torch.cuda.empty_cache()
    with torch.no_grad():
        for img, gt in zip(eval_imgs, eval_gts):
            masks = model.eval(img, normalize=True, batch_size=EVAL_BATCH_SIZE,
                               diameter=diameter)[0]
            iou, dice, miou, ap50 = seg_metrics(gt, masks)
            ious.append(iou); dices.append(dice); mious.append(miou); aps.append(ap50)
    if net_was_training:
        model.net.train()
    return np.mean(ious), np.mean(dices), np.mean(mious), np.mean(aps)


def cosine_schedule(epoch, n_epochs, lr_max, lr_min=1e-7, warmup_epochs=10):
    if epoch < warmup_epochs:
        return lr_max * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f">>> device = {device}")
    if use_gpu:
        print(f">>> GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    train_dir = ROOT / "data" / "multiclass" / f"fold_{args.fold}" / "train"
    test_dir = ROOT / "data" / "multiclass" / f"fold_{args.fold}" / "test"
    if not train_dir.exists():
        train_dir = ROOT / "data" / "train"
        test_dir = ROOT / "data" / "test"

    cpsam_weights = str(Path.home() / ".cellpose" / "models" / "cpsam")
    save_dir = ROOT / "models" / f"cpsam_v2_fold{args.fold}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    images, labels, image_names, test_images, test_labels, test_image_names = \
        io.load_train_test_data(str(train_dir), str(test_dir), mask_filter="_masks")
    print(f">>> 训练 {len(images)} 张, 测试 {len(test_images)} 张")

    eval_imgs = [io.imread(str(p)) for p in test_image_names]
    eval_gts = []
    for p in test_image_names:
        mp = Path(p).with_name(Path(p).stem + "_masks.png")
        eval_gts.append(io.imread(str(mp)))

    # 构建模型
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=cpsam_weights, device=device)
    net = model.net
    normalize_params = {**models.normalize_default, "normalize": True}

    original_dtype = net.dtype
    if net.dtype == torch.bfloat16:
        print(">>> converting bfloat16 -> float32 for training")
        net.dtype = torch.float32

    out = _process_train_test(
        train_data=images, train_labels=labels, train_files=image_names,
        test_data=test_images, test_labels=test_labels, test_files=test_image_names,
        load_files=True, min_train_masks=MIN_TRAIN_MASKS, compute_flows=False,
        channel_axis=None, normalize_params=normalize_params, device=device)
    (train_data, train_labels, train_files, train_labels_files, train_probs, diam_train,
     test_data, test_labels, test_files, test_labels_files, test_probs, diam_test,
     normed) = out
    kwargs = {} if normed else {"normalize_params": normalize_params}

    net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)
    nimg = len(train_data)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    print(f">>> n_epochs={args.epochs}, n_train={nimg}, n_test={len(test_data)}, "
          f"bsize={BSIZE}, batch_size={BATCH_SIZE}")
    print(f">>> 保存到 {save_dir}")
    print("=" * 78)

    t0 = time.time()
    best_ap = 0.0
    best_epoch = 0

    for iepoch in range(args.epochs):
        np.random.seed(iepoch)
        rperm = np.random.permutation(np.arange(0, nimg))

        lr = cosine_schedule(iepoch, args.epochs, args.lr, 1e-7, WARMUP_EPOCHS)
        for g in optimizer.param_groups:
            g["lr"] = lr

        net.train()
        epoch_loss, nsum = 0.0, 0
        for k in range(0, nimg, BATCH_SIZE):
            inds = rperm[k:min(k + BATCH_SIZE, nimg)]
            imgs, lbls = _get_batch(inds, data=train_data, labels=train_labels,
                                    files=train_files, labels_files=train_labels_files,
                                    **kwargs)
            diams = diam_train[inds]
            rsc = (diams / net.diam_mean.item()).astype("float32")
            X, lbl = random_rotate_and_resize(imgs, lbls=lbls, rescale=rsc,
                                              bsize=BSIZE, scale_range=SCALE_RANGE,
                                              device=device)[:2]
            with torch.autocast(device_type=device.type, dtype=net.dtype):
                y = net(X)[0]
                loss = _loss_fn_seg(lbl, y, device)
                if y.shape[1] > 3:
                    loss = loss + _loss_fn_class(lbl, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X)
            nsum += len(X)

        train_loss = epoch_loss / nsum
        line = f"fold{args.fold} epoch {iepoch:3d}/{args.epochs} | LR {lr:.2e} | train_loss {train_loss:.4f}"

        if (iepoch % EVAL_EVERY == 0) or (iepoch == args.epochs - 1):
            try:
                iou, dice, miou, ap50 = evaluate(model, eval_imgs, eval_gts,
                                                 diameter=float(net.diam_labels.item()))
                line += f" | IoU {iou:.4f} Dice {dice:.4f} mIoU {miou:.4f} AP@0.5 {ap50:.4f}"
                if ap50 > best_ap:
                    best_ap = ap50
                    best_epoch = iepoch
                    net.save_model(str(save_dir / "best_model"))
                    line += " [BEST]"
            except RuntimeError as e:
                line += f" | [Eval error: {e}]"
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        line += f" | {time.time()-t0:.0f}s"
        print(line, flush=True)

        if iepoch > 0 and iepoch % SAVE_EVERY == 0:
            net.save_model(str(save_dir / f"checkpoint_{iepoch}"))

    net.save_model(str(save_dir / "final_model"))

    meta = {"best_ap": best_ap, "best_epoch": best_epoch,
            "n_epochs": args.epochs, "lr": args.lr, "fold": args.fold}
    with open(save_dir / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 78)
    print(f">>> 训练完成: best AP@0.5 = {best_ap:.4f} @ epoch {best_epoch}")
    print(f">>> 模型保存到 {save_dir}")


if __name__ == "__main__":
    main()
