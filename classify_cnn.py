"""
CNN 近端/远端肾小管分类器 —— ResNet18 on cropped tubule ROIs.

比形态学特征更优：CNN 可以学习刷状缘纹理、胞质密度、管腔形状等视觉模式。
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

BATCH_SIZE = 32
N_EPOCHS = 50
LEARNING_RATE = 1e-4
IMG_SIZE = 128  # resize each tubule ROI to 128x128


class TubuleDataset(Dataset):
    """从多类别掩码中提取每个实例的 ROI crop 和 label。"""

    def __init__(self, data_dir, transform=None):
        self.samples = []
        self.transform = transform

        data_path = Path(data_dir)
        img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        img_files = [f for f in sorted(data_path.iterdir())
                     if f.suffix.lower() in img_exts and "_masks" not in f.stem]

        for img_path in img_files:
            stem = img_path.stem
            mask_path = img_path.with_name(f"{stem}_masks.png")
            csv_path = img_path.with_name(f"{stem}_class_map.csv")
            if not mask_path.exists():
                continue

            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if img is None or mask is None:
                continue

            # 加载类别
            class_map = {}
            if csv_path.exists():
                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        class_map[int(row["instance_id"])] = int(row["class_id"])

            h, w = mask.shape
            for inst_id in np.unique(mask):
                if inst_id == 0:
                    continue
                cls_id = class_map.get(inst_id, -1)
                if cls_id not in (0, 1):  # only proximal/distal
                    continue

                inst_mask = (mask == inst_id).astype(np.uint8)
                ys, xs = np.where(inst_mask)
                if len(ys) < 20:
                    continue

                # Crop with margin
                margin = 20
                y1, y2 = max(0, ys.min() - margin), min(h, ys.max() + margin)
                x1, x2 = max(0, xs.min() - margin), min(w, xs.max() + margin)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                self.samples.append((crop, cls_id))

        print(f"Loaded {len(self.samples)} tubule crops: "
              f"proximal={sum(1 for _, l in self.samples if l==0)}, "
              f"distal={sum(1 for _, l in self.samples if l==1)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        crop, label = self.samples[idx]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # Resize to fixed size
        crop_rgb = cv2.resize(crop_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        if self.transform:
            crop_rgb = self.transform(crop_rgb)
        else:
            crop_rgb = torch.from_numpy(crop_rgb).float().permute(2, 0, 1) / 255.0
        return crop_rgb, torch.tensor(label, dtype=torch.long)

    def get_labels(self):
        return [l for _, l in self.samples]


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(x)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_probs, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(x)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(x)
        all_probs.append(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
        all_labels.append(y.cpu().numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    acc = accuracy_score(labels, probs > 0.5)
    f1 = f1_score(labels, probs > 0.5)
    roc = roc_auc_score(labels, probs)
    return total_loss / n, correct / n, acc, f1, roc, labels, probs > 0.5


def build_model(device):
    """ResNet18 with transfer learning."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2),
    )
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/root/kidney/data/multiclass/all")
    parser.add_argument("--model_dir", default="/root/kidney/models/classifier")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> device = {device}")

    # Augmentation for training
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_dataset = TubuleDataset(args.data_dir, transform=None)
    all_labels = full_dataset.get_labels()

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    indices = np.arange(len(full_dataset))

    fold_results = []
    all_test_labels, all_test_preds, all_test_probs = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, all_labels)):
        print(f"\n{'='*50}")
        print(f">>> Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")

        train_ds = torch.utils.data.Subset(full_dataset, train_idx)
        test_ds = torch.utils.data.Subset(full_dataset, test_idx)

        # Override transform for subsets
        train_samples = [(full_dataset.samples[i][0], full_dataset.samples[i][1]) for i in train_idx]
        test_samples = [(full_dataset.samples[i][0], full_dataset.samples[i][1]) for i in test_idx]

        train_ds_custom = TubuleDataset.__new__(TubuleDataset)
        train_ds_custom.samples = train_samples
        train_ds_custom.transform = train_transform

        test_ds_custom = TubuleDataset.__new__(TubuleDataset)
        test_ds_custom.samples = test_samples
        test_ds_custom.transform = test_transform

        train_loader = DataLoader(train_ds_custom, batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_ds_custom, batch_size=args.batch_size, shuffle=False, num_workers=2)

        model = build_model(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_f1 = 0.0
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc, acc, f1, roc, labels, preds = evaluate(
                model, test_loader, criterion, device)
            scheduler.step()

            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(),
                          Path(args.model_dir) / f"cnn_classifier_fold{fold}.pth")

            if epoch % 5 == 0 or epoch == args.epochs - 1:
                print(f"  epoch {epoch:3d}: train_loss={train_loss:.4f} test_acc={acc:.4f} "
                      f"f1={f1:.4f} roc={roc:.4f}")

        print(f"  Best F1: {best_f1:.4f}")
        fold_results.append(best_f1)
        all_test_labels.append(labels)
        all_test_preds.append(preds)

    # Overall results
    all_labels = np.concatenate(all_test_labels)
    all_preds = np.concatenate(all_test_preds)
    print(f"\n{'='*50}")
    print(">>> 5-Fold CV Summary")
    print(f"  Per-fold F1: {[f'{f:.4f}' for f in fold_results]}")
    print(f"  Mean F1: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    print(f"  Overall Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"  Overall F1: {f1_score(all_labels, all_preds):.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(all_labels, all_preds)}")
    print(f"  Report:\n{classification_report(all_labels, all_preds, target_names=['Proximal', 'Distal'])}")

    # Save results
    out_path = Path(args.model_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "cnn_results.json", "w") as f:
        json.dump({
            "fold_f1_scores": [float(x) for x in fold_results],
            "mean_f1": float(np.mean(fold_results)),
            "std_f1": float(np.std(fold_results)),
            "overall_accuracy": float(accuracy_score(all_labels, all_preds)),
            "overall_f1": float(f1_score(all_labels, all_preds)),
        }, f, indent=2)

    print(f">>> Results saved to {out_path / 'cnn_results.json'}")


if __name__ == "__main__":
    main()
