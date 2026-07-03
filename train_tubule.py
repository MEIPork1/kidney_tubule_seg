"""
微调 Cellpose-SAM (cpsam) 训练肾小管实例分割模型 —— 自定义训练循环版。

相比直接调用 cellpose 的 train.train_seg，本脚本展开了等价的训练循环，
以便在【每一轮】打印分割指标：train_loss / test_loss / IoU / Dice / mIoU / AP@0.5。

  - IoU   : 前景(肾小管 vs 背景)像素级 交并比
  - Dice  : 前景像素级 Dice 系数
  - mIoU  : (前景IoU + 背景IoU) / 2  —— 标准二类语义分割 mIoU
  - AP@0.5: 实例级 average precision(IoU阈值0.5) —— Cellpose 本职的实例指标
  - loss  : 与 train_seg 完全一致(MSE 流场 + BCE 概率)

前置：先运行 prepare_tubule_data.py 生成 data/train 与 data/test。
运行：conda run -n cellpose --no-capture-output python train_tubule.py

说明：
  - 训练循环(LR调度/增强/损失/保存)与官方 train_seg 逐行对齐，行为一致。
  - 指标在测试集上用 model.eval() 跑完整推理(含动力学)后计算，较慢；
    每轮评估在 ~7 张大图上约需数秒。想加速可调大 EVAL_EVERY。
"""

from pathlib import Path
import time

import numpy as np
import torch

from cellpose import models, io, metrics
from cellpose.train import (_process_train_test, _get_batch,
                            _loss_fn_seg, _loss_fn_class)
from cellpose.transforms import random_rotate_and_resize

io.logger_setup()

ROOT = Path(__file__).parent
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test"
CPSAM_WEIGHTS = ROOT / "weight" / "cellpose_model-v4" / "cpsam"

# ----------------- 训练超参 -----------------
N_EPOCHS = 200            # 数据少(~39张)，200~300 epoch 较稳；先 200
LEARNING_RATE = 1e-5      # 微调常用 1e-5
WEIGHT_DECAY = 0.1
BATCH_SIZE = 1            # 8GB 显存务必保持 1
MIN_TRAIN_MASKS = 5       # 每图实例 <该值会被剔除；本数据每图≥20
MODEL_NAME = "cpsam_tubule"
SAVE_EVERY = 50           # 每多少轮保存一次
EVAL_EVERY = 10            # 每多少轮在测试集上评估指标(1=每轮；调大可加速)
EVAL_BATCH_SIZE = 8       # 推理时的 tile batch；若评估时 OOM 可调小
# --------------------------------------------


def seg_metrics(gt, pred):
    """像素级前景 IoU/Dice、二类 mIoU，以及实例级 AP@0.5。"""
    gt_fg = gt > 0
    pred_fg = pred > 0
    inter = np.logical_and(gt_fg, pred_fg).sum()
    union = np.logical_or(gt_fg, pred_fg).sum()
    iou_fg = inter / union if union else 1.0
    denom = gt_fg.sum() + pred_fg.sum()
    dice_fg = 2 * inter / denom if denom else 1.0
    # 背景类
    gt_bg, pred_bg = ~gt_fg, ~pred_fg
    inter_b = np.logical_and(gt_bg, pred_bg).sum()
    union_b = np.logical_or(gt_bg, pred_bg).sum()
    iou_bg = inter_b / union_b if union_b else 1.0
    miou = 0.5 * (iou_fg + iou_bg)
    # 实例级 AP@0.5
    ap = metrics.average_precision(gt.astype(np.int32), pred.astype(np.int32),
                                   threshold=[0.5])[0]
    ap50 = float(np.atleast_1d(ap)[0])
    return iou_fg, dice_fg, miou, ap50


def evaluate(model, eval_imgs, eval_gts, diameter):
    """在测试集上跑推理并返回各指标的均值。

    diameter: 推理时把物体缩放到 30px 训练尺度，须与训练时的缩放一致，
    否则指标虚低。一般传 net.diam_labels(= 训练集肾小管平均直径)。
    """
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


def main():
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f">>> device = {device}")

    # 加载训练/测试数据
    images, labels, image_names, test_images, test_labels, test_image_names = \
        io.load_train_test_data(str(TRAIN_DIR), str(TEST_DIR), mask_filter="_masks")
    print(f">>> 训练 {len(images)} 张, 测试 {len(test_images)} 张")

    # 评估用：直接从磁盘读 原图 + 干净实例标签(_masks.png)，与训练管线解耦
    eval_imgs = [io.imread(str(p)) for p in test_image_names]
    eval_gts = []
    for p in test_image_names:
        mp = Path(p).with_name(Path(p).stem + "_masks.png")
        eval_gts.append(io.imread(str(mp)))

    # 构建模型(微调 cpsam)
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=str(CPSAM_WEIGHTS),
                                 device=device)
    net = model.net

    # —— 以下复刻 train.train_seg 的准备工作 ——
    normalize_params = {**models.normalize_default, "normalize": True}

    # bfloat16 网络转 float32 训练(与官方一致)
    original_dtype = net.dtype
    if net.dtype == torch.bfloat16:
        print(">>> converting bfloat16 network to float32 for training")
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

    # 学习率调度(与 train_seg 完全一致)
    LR = np.linspace(0, LEARNING_RATE, 10)
    LR = np.append(LR, LEARNING_RATE * np.ones(max(0, N_EPOCHS - 10)))
    if N_EPOCHS > 300:
        LR = LR[:-100]
        for _ in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(10))
    elif N_EPOCHS > 99:
        LR = LR[:-50]
        for _ in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(5))

    optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)

    save_dir = ROOT / "models"
    save_dir.mkdir(exist_ok=True)
    filename = save_dir / MODEL_NAME
    print(f">>> n_epochs={N_EPOCHS}, n_train={nimg}, n_test={len(test_data)}, "
          f"bsize=256, batch_size={BATCH_SIZE}")
    print(f">>> 保存到 {filename}")
    print("=" * 78)

    t0 = time.time()
    for iepoch in range(N_EPOCHS):
        np.random.seed(iepoch)
        rperm = np.random.permutation(np.arange(0, nimg))
        for g in optimizer.param_groups:
            g["lr"] = LR[iepoch]
        net.train()
        epoch_loss, nsum = 0.0, 0
        for k in range(0, nimg, BATCH_SIZE):
            inds = rperm[k:min(k + BATCH_SIZE, nimg)]
            imgs, lbls = _get_batch(inds, data=train_data, labels=train_labels,
                                    files=train_files, labels_files=train_labels_files,
                                    **kwargs)
            # 关键修复：把物体缩放到 cpsam 训练尺度(diam_mean=30px)。
            # rsc = 该图肾小管直径 / 30；random_rotate_and_resize 内部按 1/rsc 缩放，
            # 于是 ~300px 的肾小管被缩到 ~30px，落进 cpsam 的有效感受野，避免碎斑。
            diams = diam_train[inds]
            rsc = (diams / net.diam_mean.item()).astype("float32")
            X, lbl = random_rotate_and_resize(imgs, lbls=lbls, rescale=rsc,
                                              bsize=256, scale_range=0.5,
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

        # —— 每 EVAL_EVERY 轮评估指标 ——
        line = f"epoch {iepoch:3d}/{N_EPOCHS} | LR {LR[iepoch]:.2e} | train_loss {train_loss:.4f}"
        if (iepoch % EVAL_EVERY == 0) or (iepoch == N_EPOCHS - 1):
            try:
                iou, dice, miou, ap50 = evaluate(model, eval_imgs, eval_gts,
                                                 diameter=net.diam_labels.item())
                line += (f" | IoU {iou:.4f} | Dice {dice:.4f} | "
                         f"mIoU {miou:.4f} | AP@0.5 {ap50:.4f}")
            except RuntimeError as e:
                line += f" | [评估失败:{type(e).__name__} {e}]"
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        line += f" | {time.time()-t0:.0f}s"
        print(line, flush=True)

        if iepoch == N_EPOCHS - 1 or (iepoch % SAVE_EVERY == 0 and iepoch != 0):
            net.save_model(str(filename))
            print(f"    >>> 已保存模型到 {filename}", flush=True)

    net.save_model(str(filename))
    if original_dtype != torch.float32:
        net.dtype = original_dtype
    print("=" * 78)
    print(f">>> 训练完成，模型保存到: {filename}")


if __name__ == "__main__":
    main()
