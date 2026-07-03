"""快速评估 CNN 分类器 fold 0 模型在所有数据上的性能。"""
import csv
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device = {device}")

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.3), nn.Linear(512, 128), nn.ReLU(),
    nn.Dropout(0.3), nn.Linear(128, 2),
)
state = torch.load("/root/kidney/models/classifier/cnn_classifier_fold0.pth",
                   map_location=device, weights_only=True)
model.load_state_dict(state)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

data_dir = Path("/root/kidney/data/multiclass/all")
img_exts = {".jpg", ".jpeg", ".png"}
img_files = sorted([f for f in data_dir.iterdir()
                   if f.suffix.lower() in img_exts and "_masks" not in f.stem])

all_preds, all_labels = [], []
for img_path in img_files:
    img = cv2.imread(str(img_path))
    mask_path = img_path.with_name(img_path.stem + "_masks.png")
    csv_path = img_path.with_name(img_path.stem + "_class_map.csv")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if img is None or mask is None:
        continue

    class_map = {}
    if csv_path.exists():
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                class_map[int(row["instance_id"])] = int(row["class_id"])

    h, w = mask.shape
    for inst_id in np.unique(mask):
        if inst_id == 0:
            continue
        cls_id = class_map.get(inst_id, -1)
        if cls_id not in (0, 1):
            continue
        inst_mask = (mask == inst_id).astype(np.uint8)
        ys, xs = np.where(inst_mask)
        if len(ys) < 20:
            continue
        margin = 20
        y1, y2 = max(0, ys.min() - margin), min(h, ys.max() + margin)
        x1, x2 = max(0, xs.min() - margin), min(w, xs.max() + margin)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = transform(crop_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            pred = logits.argmax(dim=1).item()
        all_preds.append(pred)
        all_labels.append(cls_id)

n = len(all_labels)
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
roc = roc_auc_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print(f"Fold 0 model on ALL {n} instances")
print(f"  Accuracy:  {acc:.4f}")
print(f"  F1:        {f1:.4f}")
print(f"  ROC-AUC:   {roc:.4f}")
print(f"  Confusion Matrix (rows=GT, cols=Pred):")
print(f"    Proximal (GT): {cm[0]}")
print(f"    Distal   (GT): {cm[1]}")
print(f"  Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Proximal", "Distal"]))
