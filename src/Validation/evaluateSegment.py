#!/usr/bin/env python3
"""
Evaluate your segmentation model against COCO ground truth.

  1. imageSort.find_and_move_unpaired() pairs model images with COCO entries by
     the ID in the file name and moves model images that have no ground truth.
  2. The COCO annotations (polygons or RLE) of each image are rasterised into a
     binary mask. Your model images are black except for the segment, so
     foreground = any pixel above --threshold (default 0) in any channel.
  3. Precision, recall, F1 and IoU are computed per image and summarised.

Usage (from the project folder, with images/modelSeg and images/groundTruth):
    python evaluateSegment.py
    python evaluateSegment.py --csv results.csv --category-ids 1

From another script:
    from evaluateSegment import evaluate
    results = evaluate()
    print(results["mean"]["f1"], results["mean"]["miou"])

Notes:
  * All annotations of an image are merged (union) into one foreground mask;
    use --category-ids to only include certain COCO categories.
  * Polygons work out of the box. RLE annotations (and exact COCO-style
    rasterisation) need pycocotools:  pip install pycocotools
  * Saving the model's output as JPEG puts compression noise into the black
    background; if so, raise --threshold (e.g. 10) or save as PNG.
  * Zero-division convention: no predicted AND no true foreground -> all scores
    1.0; if only one side is empty, the undefined score is 0.0.
  * Requires: numpy, pillow.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from imageSort import HERE, find_and_move_unpaired

try:
    from pycocotools import mask as mask_utils
except ImportError:  # polygons still work through the PIL fallback
    mask_utils = None


def load_pred_mask(path, threshold=0):
    """Model image -> boolean foreground mask (non-black pixels)."""
    img = Image.open(path)
    if img.mode in ("I;16", "I", "F"):
        arr = np.asarray(img)
    else:
        arr = np.asarray(img.convert("RGB")).max(axis=2)
    return arr > threshold


def polygon_mask(poly, h, w):
    """Rasterise one flat COCO polygon [x1, y1, x2, y2, ...]: a pixel (i, j) is foreground
    if its centre (i + 0.5, j + 0.5) is inside the polygon (even-odd rule)."""
    xs = np.asarray(poly[0::2], dtype=float)
    ys = np.asarray(poly[1::2], dtype=float)
    out = np.zeros((h, w), dtype=bool)
    x0, x1 = max(int(np.floor(xs.min())), 0), min(int(np.ceil(xs.max())), w)
    y0, y1 = max(int(np.floor(ys.min())), 0), min(int(np.ceil(ys.max())), h)
    if x1 <= x0 or y1 <= y0:
        return out
    X, Y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    inside = np.zeros(X.shape, dtype=bool)
    n = len(xs)
    for i in range(n):
        xa, ya, xb, yb = xs[i], ys[i], xs[(i + 1) % n], ys[(i + 1) % n]
        crosses = (ya > Y) != (yb > Y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at_y = (xb - xa) * (Y - ya) / (yb - ya) + xa
        inside ^= crosses & (X < x_at_y)
    out[y0:y1, x0:x1] = inside
    return out


def rle_to_mask(seg):
    """Decode a COCO RLE dict ({"size": [h, w], "counts": ...}, compressed string or
    plain list) without pycocotools. COCO RLE runs go down columns (Fortran order)."""
    h, w = seg["size"]
    counts = seg["counts"]
    if isinstance(counts, (str, bytes)):
        s = counts.encode() if isinstance(counts, str) else counts
        runs, p = [], 0
        while p < len(s):
            x, k, more = 0, 0, True
            while more:
                c = s[p] - 48
                x |= (c & 0x1F) << (5 * k)
                more = bool(c & 0x20)
                p += 1
                k += 1
                if not more and (c & 0x10):
                    x |= -1 << (5 * k)
            if len(runs) > 2:
                x += runs[-2]
            runs.append(x)
        counts = runs
    flat = np.zeros(h * w, dtype=bool)
    pos, value = 0, False
    for run in counts:
        if value:
            flat[pos:pos + run] = True
        pos += run
        value = not value
    return flat.reshape((h, w), order="F")


def annotation_to_mask(ann, h, w):
    """One COCO annotation -> boolean (h, w) mask."""
    seg = ann.get("segmentation")
    if not seg:
        return np.zeros((h, w), dtype=bool)

    if mask_utils is not None:
        if isinstance(seg, list):
            rle = mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
        elif isinstance(seg["counts"], list):          # uncompressed RLE
            rle = mask_utils.frPyObjects(seg, h, w)
        else:                                          # compressed RLE
            counts = seg["counts"]
            rle = {"size": seg["size"],
                   "counts": counts.encode() if isinstance(counts, str) else counts}
        return mask_utils.decode(rle).astype(bool)

    if isinstance(seg, dict):
        mask = rle_to_mask(seg)
        if mask.shape != (h, w):
            raise ValueError(f"RLE size {mask.shape} does not match COCO image size {(h, w)}")
        return mask
    mask = np.zeros((h, w), dtype=bool)
    for poly in seg:
        if len(poly) >= 6:
            mask |= polygon_mask(poly, h, w)
    return mask


def coco_mask(gt_image, annotations, category_ids=None):
    h, w = int(gt_image["height"]), int(gt_image["width"])
    mask = np.zeros((h, w), dtype=bool)
    for ann in annotations:
        if category_ids and ann.get("category_id") not in category_ids:
            continue
        mask |= annotation_to_mask(ann, h, w)
    return mask


def confusion_counts(pred, gt):
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(pred.size - tp - fp - fn)
    return tp, fp, fn, tn


def scores_from_counts(tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else (1.0 if fp == 0 else 0.0)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    iou_fg = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    iou_bg = tn / (tn + fp + fn) if (tn + fp + fn) else 1.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "iou": iou_fg,                   # segment IoU
            "miou": (iou_fg + iou_bg) / 2}   # mean of segment and background IoU


def evaluate(model_dir=HERE / "images" / "modelSeg", gt_dir=HERE / "images" / "groundTruth",
             coco_path=None, out_dir=HERE / "unpaired", id_regex=r"\d+", threshold=0,
             category_ids=None, move=True, resize_gt=False, require_annotations=False):
    """
    Returns {"per_image": [...], "mean": {...}, "pooled": {...}, "skipped": [...]}.
    "mean"   = average of per-image scores (every image counts equally)
    "pooled" = scores from pixel counts summed over all images (big segments count more)
    """
    pairs = find_and_move_unpaired(model_dir, gt_dir, coco_path, out_dir, id_regex,
                                   dry_run=not move, require_annotations=require_annotations)

    per_image, skipped = [], []
    totals = np.zeros(4, dtype=np.int64)
    for pair in pairs:
        pred = load_pred_mask(pair["model_path"], threshold)
        gt = coco_mask(pair["gt_image"], pair["annotations"], category_ids)
        if pred.shape != gt.shape:
            if resize_gt:
                h, w = pred.shape
                gt = np.asarray(Image.fromarray(gt.astype(np.uint8) * 255)
                                .resize((w, h), Image.NEAREST)) > 0
            else:
                skipped.append((pair["model_path"].name, pair["gt_image"]["file_name"],
                                f"size mismatch: model {pred.shape} vs COCO {gt.shape}"))
                continue
        counts = confusion_counts(pred, gt)
        totals += counts
        per_image.append({"id": pair["id"], "model_file": pair["model_path"].name,
                          "gt_file": pair["gt_image"]["file_name"], **scores_from_counts(*counts)})

    keys = ["precision", "recall", "f1", "iou", "miou"]
    mean = {k: float(np.mean([r[k] for r in per_image])) for k in keys} if per_image else {}
    pooled = scores_from_counts(*[int(t) for t in totals]) if per_image else {}
    return {"per_image": per_image, "mean": mean, "pooled": pooled, "skipped": skipped}


def print_report(results):
    print(f"\nEvaluated {len(results['per_image'])} image pairs ({len(results['skipped'])} skipped)")
    for model_name, gt_name, reason in results["skipped"]:
        print(f"  skipped {model_name} / {gt_name}: {reason}")
    if not results["per_image"]:
        return
    labels = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1 score"),
              ("iou", "IoU (segment)"), ("miou", "Mean IoU (segment+background)")]
    print(f"\n{'':32}{'mean per image':>16}{'pooled':>10}")
    for key, label in labels:
        print(f"{label:32}{results['mean'][key]:>16.4f}{results['pooled'][key]:>10.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=HERE / "images" / "modelSeg")
    p.add_argument("--gt-dir", type=Path, default=HERE / "images" / "groundTruth")
    p.add_argument("--coco", type=Path, help="COCO json (default: the single .json in --gt-dir)")
    p.add_argument("--out-dir", type=Path, default=HERE / "unpaired")
    p.add_argument("--id-regex", default=r"\d+")
    p.add_argument("--threshold", type=int, default=0,
                   help="pixel value above which a model pixel counts as foreground (default 0)")
    p.add_argument("--category-ids", type=int, nargs="+",
                   help="only use these COCO category ids (default: all)")
    p.add_argument("--no-move", action="store_true", help="report unpaired images but don't move them")
    p.add_argument("--resize-gt", action="store_true",
                   help="resize the COCO mask to the model image size when sizes differ")
    p.add_argument("--require-annotations", action="store_true",
                   help="treat COCO images with no annotations as having no ground truth")
    p.add_argument("--csv", type=Path, help="write per-image scores to this CSV file")
    a = p.parse_args()

    results = evaluate(a.model_dir, a.gt_dir, a.coco, a.out_dir, a.id_regex, a.threshold,
                       a.category_ids, move=not a.no_move, resize_gt=a.resize_gt,
                       require_annotations=a.require_annotations)
    print_report(results)

    if a.csv and results["per_image"]:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results["per_image"][0].keys()))
            w.writeheader()
            w.writerows(results["per_image"])
        print(f"\nPer-image scores written to {a.csv}")


if __name__ == "__main__":
    main()