"""
多类别数据预处理：保留近端/远端肾小管分类标签 + K-fold 交叉验证划分。

class 0 → 近端小管 (proximal)
class 1 → 远端小管 (distal)
class 2 → 罕见/其他 (保留但量少)

输出:
  data/multiclass/
    ├── all/           # 全部46张图 + _masks.png（多类uint16）+ _class_map.csv
    ├── fold_0/ .. fold_4/
    │   ├── train/     # 训练集
    │   └── test/      # 验证集

掩码格式：uint16, 每个实例有唯一ID。辅助 CSV 记录每个实例ID对应的类别。
"""

import json
import shutil
from pathlib import Path

import numpy as np
import cv2
import csv

# ----------------- 配置 -----------------
ROOT = Path("/root/kidney")
IMG_DIR = ROOT / "data" / "Capture"
GEO_DIR = ROOT / "data" / "geojson"
OUT_ALL = ROOT / "data" / "multiclass" / "all"
OUT_FOLDS = ROOT / "data" / "multiclass"
IMG_EXT = ".jpg"

N_FOLDS = 5
TEST_FRACTION = 0.20  # 5-fold: 20% test per fold
SEED = 42

# 类别映射
CLASS_MAP = {0: "proximal", 1: "distal", 2: "other/rare"}
# ----------------------------------------


def iter_polygons(geom):
    t = geom.get("type")
    if t == "Polygon":
        rings = geom["coordinates"]
        if not rings:
            return []
        return [(rings[0], rings[1:])]
    if t == "MultiPolygon":
        out = []
        for poly in geom["coordinates"]:
            if poly:
                out.append((poly[0], poly[1:]))
        return out
    return []


def ring_to_xy(ring, h, w):
    a = np.asarray(ring, dtype=np.float64)
    xs = np.clip(a[:, 0], 0, w - 1)
    ys = np.clip(a[:, 1], 0, h - 1)
    pts = np.stack([xs, ys], axis=1)
    return np.round(pts).astype(np.int32).reshape(-1, 1, 2)


def rasterize_multiclass(geojson_path, h, w):
    """返回 (label_uint16, class_map_dict) 其中 class_map = {instance_id: class_name}"""
    data = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    feats = data["features"] if data.get("type") == "FeatureCollection" else [data]

    label = np.zeros((h, w), dtype=np.uint16)
    class_map = {}
    next_id = 1
    skipped = 0

    for ft in feats:
        geom = ft.get("geometry", {})
        cls_name = str((ft.get("properties", {}).get("classification") or {}).get("name", "0"))
        cls_int = int(cls_name) if cls_name.isdigit() else 0

        parts = iter_polygons(geom)
        if not parts:
            skipped += 1
            continue

        inst = np.zeros((h, w), dtype=np.uint8)
        for exterior, holes in parts:
            cv2.fillPoly(inst, [ring_to_xy(exterior, h, w)], 1)
            for hole in holes:
                cv2.fillPoly(inst, [ring_to_xy(hole, h, w)], 0)
        inst = inst.astype(bool)
        if not inst.any():
            skipped += 1
            continue

        free = inst & (label == 0)
        label[free] = next_id
        class_map[next_id] = cls_int
        next_id += 1

    return label, class_map, next_id - 1, skipped


def write_class_csv(csv_path, class_map):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "class_id", "class_name"])
        for inst_id, cls_id in sorted(class_map.items()):
            w.writerow([inst_id, cls_id, CLASS_MAP.get(cls_id, "unknown")])


def main():
    OUT_ALL.mkdir(parents=True, exist_ok=True)

    geos = sorted(GEO_DIR.glob("*.geojson"))
    if not geos:
        raise SystemExit(f"未找到 geojson: {GEO_DIR}")

    # K-fold split
    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(geos))

    total_inst = {"proximal": 0, "distal": 0, "other/rare": 0}
    all_items = []  # (stem, n_inst, class_map)

    for i, geo in enumerate(geos):
        stem = geo.stem
        img_path = IMG_DIR / f"{stem}{IMG_EXT}"
        if not img_path.exists():
            print(f"[警告] 缺少原图，跳过: {img_path.name}")
            continue

        img = cv2.imread(str(img_path), -1)
        if img is None:
            print(f"[警告] 无法读取原图: {img_path.name}")
            continue
        h, w = img.shape[:2]

        label, class_map, n_inst, skipped = rasterize_multiclass(geo, h, w)

        # 统计
        for cls_id in class_map.values():
            cat = CLASS_MAP.get(cls_id, "unknown")
            total_inst[cat] = total_inst.get(cat, 0) + 1

        shutil.copy(img_path, OUT_ALL / f"{stem}{IMG_EXT}")
        cv2.imwrite(str(OUT_ALL / f"{stem}_masks.png"), label)
        write_class_csv(OUT_ALL / f"{stem}_class_map.csv", class_map)

        all_items.append((stem, n_inst, class_map))
        warn = f"  (跳过 {skipped})" if skipped else ""
        print(f"{stem:>14} -> {n_inst:3d} 实例 | P:{sum(1 for v in class_map.values() if v==0):3d} "
              f"D:{sum(1 for v in class_map.values() if v==1):3d} O:{sum(1 for v in class_map.values() if v==2)} {warn}")

    # 写 K-fold 划分
    fold_size = len(geos) // N_FOLDS
    for fold in range(N_FOLDS):
        fold_dir = OUT_FOLDS / f"fold_{fold}"
        train_dir = fold_dir / "train"
        test_dir = fold_dir / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        test_start = fold * fold_size
        test_end = min((fold + 1) * fold_size, len(geos))
        test_indices = set(order[test_start:test_end])

        n_train, n_test = 0, 0
        for i, (stem, n_inst, class_map) in enumerate(all_items):
            out_dir = test_dir if i in test_indices else train_dir
            src_img = OUT_ALL / f"{stem}{IMG_EXT}"
            src_mask = OUT_ALL / f"{stem}_masks.png"
            shutil.copy(src_img, out_dir / f"{stem}{IMG_EXT}")
            shutil.copy(src_mask, out_dir / f"{stem}_masks.png")
            if i in test_indices:
                n_test += 1
            else:
                n_train += 1

        print(f"  fold_{fold}: train={n_train}, test={n_test}")

    print(f"\n========== 汇总 ==========")
    print(f"总图: {len(all_items)}, 总实例: {sum(total_inst.values())}")
    for cat, count in total_inst.items():
        print(f"  {cat}: {count}")
    print(f"输出: {OUT_ALL}")
    print(f"K-fold ({N_FOLDS}): {OUT_FOLDS}/fold_*")


if __name__ == "__main__":
    main()
