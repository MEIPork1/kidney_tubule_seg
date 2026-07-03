"""
掩码后处理：提升分割质量。

操作:
  1. 填洞 (binary_fill_holes)
  2. 移除碎片 (< min_area)
  3. 可选：分裂合并实例 (watershed on distance transform)

用法:
  python postprocess_masks.py --mask pred_masks.png --out cleaned_masks.png
"""

import argparse
import numpy as np
from scipy import ndimage
import cv2


def fill_holes(mask):
    """Fill holes in each instance mask."""
    out = np.zeros_like(mask)
    for inst_id in np.unique(mask):
        if inst_id == 0:
            continue
        inst_mask = (mask == inst_id).astype(np.uint8)
        filled = ndimage.binary_fill_holes(inst_mask).astype(np.uint8)
        out[filled > 0] = inst_id
    return out


def remove_small(mask, min_area=50):
    """Remove instances smaller than min_area pixels."""
    out = mask.copy()
    for inst_id in np.unique(mask):
        if inst_id == 0:
            continue
        if (mask == inst_id).sum() < min_area:
            out[mask == inst_id] = 0
    return out


def split_merged(mask, min_distance=15, min_area=100):
    """
    Split potentially merged instances using watershed on distance transform.
    Only splits if the instance is much larger than average.
    """
    areas = [(mask == i).sum() for i in np.unique(mask) if i > 0]
    if len(areas) < 2:
        return mask
    median_area = np.median(areas)

    out = mask.copy()
    next_id = mask.max() + 1

    for inst_id in np.unique(mask):
        if inst_id == 0:
            continue
        inst_mask = (mask == inst_id).astype(np.uint8)
        area = inst_mask.sum()

        # Only attempt split if area > 2.5x median
        if area < 2.5 * median_area:
            continue

        dist = cv2.distanceTransform(inst_mask, cv2.DIST_L2, 5)
        from scipy.ndimage import maximum_filter

        # Find local maxima
        local_max = (dist == maximum_filter(dist, size=min_distance * 2 + 1))
        markers = ndimage.label(local_max)[0]

        if markers.max() < 2:
            continue

        # Watershed
        from skimage.segmentation import watershed
        labels = watershed(-dist, markers, mask=inst_mask)

        # Reassign
        out[inst_mask > 0] = 0
        for new_id in np.unique(labels):
            if new_id == 0:
                continue
            new_mask = (labels == new_id)
            if new_mask.sum() < min_area:
                continue
            out[new_mask] = next_id
            next_id += 1

    return out


def clean_mask(mask, min_area=50, fill=True, split=False):
    """Full post-processing pipeline."""
    if fill:
        mask = fill_holes(mask)
    mask = remove_small(mask, min_area=min_area)
    if split:
        mask = split_merged(mask, min_area=min_area)
    mask = remove_small(mask, min_area=min_area)
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min_area", type=int, default=50)
    parser.add_argument("--no_fill", action="store_true")
    parser.add_argument("--split", action="store_true")
    args = parser.parse_args()

    mask = cv2.imread(args.mask, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise SystemExit(f"Cannot read {args.mask}")

    n_before = len(np.unique(mask)) - 1
    cleaned = clean_mask(mask, min_area=args.min_area,
                        fill=not args.no_fill, split=args.split)
    n_after = len(np.unique(cleaned)) - 1

    cv2.imwrite(args.out, cleaned.astype(np.uint16))
    print(f"  {n_before} → {n_after} instances (removed {n_before - n_after})")


if __name__ == "__main__":
    main()
