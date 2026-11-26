#!/usr/bin/env python3
"""Zero-shot SAM3 evaluation: text prompt -> instance masks on a COCO-format test set.

Usage:
    python sam3_fish_eval.py --ann /path/to/annotations.json
"""
import argparse, contextlib, io, json, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

COCO_KEYS = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
             "AR@1", "AR@10", "AR@100", "AR_small", "AR_medium", "AR_large"]


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def to_np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=np.float32)


def load_gt(path):
    """COCO ground truth with every category (e.g. species) merged into one 'fish' class."""
    with quiet():
        gt = COCO(path)
        for a in gt.dataset["annotations"]:
            a["category_id"] = 1
            a.setdefault("iscrowd", 0)
            a.setdefault("area", a["bbox"][2] * a["bbox"][3])
        gt.dataset["categories"] = [{"id": 1, "name": "fish"}]
        gt.createIndex()
    return gt


def coco_eval(gt, dets, iou_type):
    if not dets:
        return {k: 0.0 for k in COCO_KEYS}
    with quiet():
        E = COCOeval(gt, gt.loadRes(dets), iou_type)
        E.evaluate(); E.accumulate(); E.summarize()
    return {k: float(v) for k, v in zip(COCO_KEYS, E.stats)}


def match_ious(dts, gts, thr):
    """Greedy one-to-one matching (dts sorted by score). Returns IoUs of true positives."""
    if not dts or not gts:
        return []
    ious, hits = np.array(mask_utils.iou(dts, gts, [0] * len(gts))), []
    for d in range(len(dts)):
        g = ious[d].argmax()
        if ious[d, g] >= thr:
            hits.append(float(ious[d, g]))
            ious[:, g] = -1
    return hits


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--images",
        type=Path,
        default=Path("/mnt/c/Users/jakob/Pictures/Fishes"),
        help="folder with test images",
    )
    p.add_argument("--ann", default="/mnt/c/Users/jakob/Pictures/Fishes/_annotations.coco.json",help="COCO-format annotation json")
    p.add_argument("--prompt", default="fish", help="text prompt for SAM3")
    p.add_argument("--score-thr", type=float, default=0.5, help="operating point for P/R/F1, counts, pixel IoU")
    p.add_argument("--iou-thr", type=float, default=0.5, help="IoU needed for a true positive at the operating point")
    p.add_argument("--min-score", type=float, default=0.05, help="lowest score kept (for COCO AP)")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/mnt/c/Users/jakob/Pictures/Segmented_Fishes/Baseline"),
        help="output folder for segmented images",
    )
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    gt = load_gt(args.ann)
    has_masks = all(a.get("segmentation") for a in gt.dataset["annotations"])

    model = build_sam3_image_model()
    try:  # keep low-confidence detections so AP sees the full precision-recall curve
        processor = Sam3Processor(model, confidence_threshold=args.min_score)
    except TypeError:
        processor = Sam3Processor(model)

    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    dets, times, tp_ious, count_err = [], [], [], []
    n_pred = n_gt = inter = union = pred_px = gt_px = 0
    img_ids = sorted(gt.getImgIds())

    for i, img_id in enumerate(img_ids, 1):
        info = gt.loadImgs(img_id)[0]
        image = Image.open(args.images / info["file_name"]).convert("RGB")
        w, h = image.size

        if cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=cuda
        ):
            state = processor.set_image(image)
            out = processor.set_text_prompt(state=state, prompt=args.prompt)
        if cuda: torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

        scores = to_np(out["scores"]).reshape(-1)
        order = np.argsort(-scores)
        scores = scores[order]
        boxes = to_np(out["boxes"]).reshape(-1, 4)[order]                  # xyxy, pixels
        masks = (to_np(out["masks"]).reshape(-1, h, w) > 0.5)[order]        # N x H x W
        xywh = np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], axis=1)
        rles = [mask_utils.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]

        for s, b, r in zip(scores, xywh, rles):
            r["counts"] = r["counts"].decode("ascii")
            dets.append({"image_id": img_id, "category_id": 1, "score": float(s),
                         "bbox": [float(v) for v in b], "segmentation": r})

        # Operating-point metrics (detections with score >= score_thr)
        keep = scores >= args.score_thr
        anns = [a for a in gt.loadAnns(gt.getAnnIds(imgIds=img_id)) if not a["iscrowd"]]
        if has_masks:
            dt_obj = [r for r, k in zip(rles, keep) if k]
            gt_obj = [gt.annToRLE(a) for a in anns]
        else:
            dt_obj, gt_obj = xywh[keep].tolist(), [a["bbox"] for a in anns]
        tp_ious += match_ious(dt_obj, gt_obj, args.iou_thr)
        n_pred += int(keep.sum()); n_gt += len(anns)
        count_err.append(int(keep.sum()) - len(anns))

        if has_masks:  # foreground (semantic) overlap
            pm = masks[keep].any(0)
            gm = np.zeros((h, w), bool)
            for a in anns:
                gm |= gt.annToMask(a).astype(bool)
            inter += int((pm & gm).sum()); union += int((pm | gm).sum())
            pred_px += int(pm.sum()); gt_px += int(gm.sum())

        if i % 50 == 0:
            print(f"{i}/{len(img_ids)} images")

    tp = len(tp_ious); fp = n_pred - tp; fn = n_gt - tp
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    ce = np.array(count_err, dtype=float)
    t = np.array(times[1:] or times)  # drop warm-up image

    metrics = {"prompt": args.prompt, "n_images": len(img_ids), "n_gt_fish": n_gt}
    if has_masks:
        metrics["mask_COCO"] = coco_eval(gt, [{k: v for k, v in d.items() if k != "bbox"} for d in dets], "segm")
    metrics["box_COCO"] = coco_eval(gt, [{k: v for k, v in d.items() if k != "segmentation"} for d in dets], "bbox")
    metrics["operating_point"] = {
        "score_thr": args.score_thr, "iou_thr": args.iou_thr, "TP": tp, "FP": fp, "FN": fn,
        "precision": prec, "recall": rec, "F1": 2 * prec * rec / max(prec + rec, 1e-9),
        "mean_matched_IoU": float(np.mean(tp_ious)) if tp_ious else 0.0}
    if has_masks:
        metrics["pixel"] = {"fg_IoU": inter / max(union, 1), "fg_Dice": 2 * inter / max(pred_px + gt_px, 1)}
    metrics["counting"] = {"MAE": float(np.abs(ce).mean()), "RMSE": float(np.sqrt((ce ** 2).mean())),
                           "bias_pred_minus_gt": float(ce.mean()), "exact_count_acc": float((ce == 0).mean())}
    metrics["speed"] = {"ms_per_image": 1000 * float(t.mean()), "FPS": float(1 / t.mean()),
                        "peak_gpu_mem_GB": torch.cuda.max_memory_allocated() / 1e9 if cuda else None}

    for section, vals in metrics.items():
        if isinstance(vals, dict):
            print(f"\n[{section}]")
            for k, v in vals.items():
                print(f"  {k:<20} {v:.4f}" if isinstance(v, float) else f"  {k:<20} {v}")
        else:
            print(f"{section}: {vals}")

    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.out / "predictions.json").write_text(json.dumps(dets))  # COCO results format
    print(f"\nSaved metrics and predictions to {args.out}/")


if __name__ == "__main__":
    main()
