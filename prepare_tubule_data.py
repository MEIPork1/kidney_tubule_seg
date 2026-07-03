"""
geojson 标注 -> Cellpose 实例标签 + 划分 train/test

数据约定（当前项目）：
  原图     : data/Capture/<stem>.jpg          例如 "Captured 1.jpg"
  标注     : data/geojson/<stem>.geojson      QuPath 风格 FeatureCollection（像素坐标）
  类别     : classification.name = 0/1/2，全部都是肾小管，合并为单类实例分割

输出（Cellpose train_seg 要求图像与掩码同目录、掩码用 _masks 后缀）：
  data/train/<stem>.png
  data/train/<stem>_masks.png   uint16，0=背景，1..N=每个肾小管实例
  data/test/ ...

说明：
  - 不做任何尺寸缩放/resize。保持原图分辨率，Cellpose 训练时自动裁 256 patch、
    并用 normalize99 做强度归一化。
  - MultiPolygon 视为一个实例（多个部分共用一个 ID），内环(holes)挖空。
  - LineString 等非面要素无法填充，跳过并告警。
"""

import json
import shutil
from pathlib import Path

import numpy as np
import cv2

# ----------------- 配置 -----------------
ROOT = Path(__file__).parent
IMG_DIR = ROOT / "data" / "Capture"        # 原图目录
GEO_DIR = ROOT / "data" / "geojson"        # geojson 目录
OUT_TRAIN = ROOT / "data" / "train"
OUT_TEST = ROOT / "data" / "test"
IMG_EXT = ".jpg"                            # 原图扩展名

TEST_FRACTION = 0.15                        # 测试集比例（46*0.15≈7 张）
SEED = 0                                    # 划分随机种子（可复现）
# 想保留的类别名；None 表示全部都算肾小管。本项目 0/1/2 都是肾小管 -> None
KEEP_CLASSES = None                         # 例如只要近端: {"1"}
# ----------------------------------------


def iter_polygons(geom):
    """把 Polygon / MultiPolygon 拆成 [(exterior, [holes...]), ...]，
    每个元素是一个独立填充区域。非面要素返回空。"""
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
    # LineString / Point / 其它 -> 无法填充
    return []


def ring_to_xy(ring, h, w):
    """geojson 环 [[x,y],...] -> cv2 需要的 int32 (N,1,2) 点数组，裁剪到图像范围内。"""
    a = np.asarray(ring, dtype=np.float64)
    xs = np.clip(a[:, 0], 0, w - 1)
    ys = np.clip(a[:, 1], 0, h - 1)
    pts = np.stack([xs, ys], axis=1)
    return np.round(pts).astype(np.int32).reshape(-1, 1, 2)


def rasterize(geojson_path, h, w):
    """读 geojson，返回 uint16 实例标签图 (h, w)。"""
    data = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    feats = data["features"] if data.get("type") == "FeatureCollection" else [data]

    label = np.zeros((h, w), dtype=np.uint16)
    next_id = 1
    skipped = 0
    overlaps = 0

    for ft in feats:
        geom = ft.get("geometry", {})
        cls = (ft.get("properties", {}).get("classification") or {}).get("name")
        if KEEP_CLASSES is not None and str(cls) not in KEEP_CLASSES:
            continue

        parts = iter_polygons(geom)
        if not parts:
            skipped += 1
            continue

        # 一个 feature = 一个实例（MultiPolygon 的多个部分共用 ID）
        inst = np.zeros((h, w), dtype=np.uint8)
        for exterior, holes in parts:
            cv2.fillPoly(inst, [ring_to_xy(exterior, h, w)], 1)
            for hole in holes:
                cv2.fillPoly(inst, [ring_to_xy(hole, h, w)], 0)
        inst = inst.astype(bool)
        if not inst.any():
            skipped += 1
            continue

        # 仅写入当前还是背景的像素，避免覆盖已有实例（统计重叠）
        free = inst & (label == 0)
        if free.sum() < inst.sum():
            overlaps += 1
        label[free] = next_id
        next_id += 1

    return label, next_id - 1, skipped, overlaps


def main():
    OUT_TRAIN.mkdir(parents=True, exist_ok=True)
    OUT_TEST.mkdir(parents=True, exist_ok=True)

    geos = sorted(GEO_DIR.glob("*.geojson"))
    if not geos:
        raise SystemExit(f"未找到 geojson: {GEO_DIR}")

    # 复现的 train/test 划分
    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(geos))
    n_test = max(1, round(len(geos) * TEST_FRACTION))
    test_idx = set(order[:n_test].tolist())

    total_inst = total_skip = 0
    n_train = n_test_done = 0

    for i, geo in enumerate(geos):
        stem = geo.stem                       # "Captured 1"
        img_path = IMG_DIR / f"{stem}{IMG_EXT}"
        if not img_path.exists():
            print(f"[警告] 缺少原图，跳过: {img_path.name}")
            continue

        img = cv2.imread(str(img_path), -1)
        if img is None:
            print(f"[警告] 无法读取原图，跳过: {img_path.name}")
            continue
        h, w = img.shape[:2]

        label, n_inst, skipped, overlaps = rasterize(geo, h, w)
        total_inst += n_inst
        total_skip += skipped

        out_dir = OUT_TEST if i in test_idx else OUT_TRAIN
        if i in test_idx:
            n_test_done += 1
        else:
            n_train += 1

        # 原图直接复制（Cellpose 能读 jpg）；掩码存 uint16 PNG
        shutil.copy(img_path, out_dir / f"{stem}{IMG_EXT}")
        cv2.imwrite(str(out_dir / f"{stem}_masks.png"), label)

        warn = f"  (跳过非面要素 {skipped})" if skipped else ""
        warn += f"  (重叠 {overlaps})" if overlaps else ""
        print(f"{stem:>14} -> {'TEST ' if i in test_idx else 'TRAIN'}  实例 {n_inst:3d}{warn}")

    print("\n========== 汇总 ==========")
    print(f"训练集 {n_train} 张, 测试集 {n_test_done} 张")
    print(f"实例总数 {total_inst}, 跳过非面要素 {total_skip} 个")
    print(f"输出: {OUT_TRAIN}  /  {OUT_TEST}")


if __name__ == "__main__":
    main()
