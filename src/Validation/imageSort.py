#!/usr/bin/env python3
"""
Pair the model's segmentation images with the COCO ground truth and move the
model images that have no ground truth.

Matching is done on the ID written in the file names: the ID in the model
image's file name must equal the ID in a COCO "images" entry's file_name.

Folder layout (defaults, relative to this script):
    images/modelSeg/       <- your model's segmentation images
    images/groundTruth/    <- the COCO annotation .json (and optionally the original images)

Examples:
    python imageSort.py --dry-run
    python imageSort.py
    python imageSort.py --id-regex "case_(\\d+)" --coco images/groundTruth/annotations.json

The ID is the first match of --id-regex in the file name. If the regex has a
capture group, the group is used; otherwise the whole match. Default: the first
run of digits. Leading zeros are ignored ("0042" == "42").

Model images with no COCO entry are moved to <out-dir>/no_ground_truth.
COCO entries with no model image can't be "moved" (they are JSON records), so
they are listed and simply left out of the evaluation.
"""

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def extract_id(name, pattern):
    m = pattern.search(name)
    if not m:
        return None
    key = m.group(1) if m.groups() else m.group(0)
    return (key.lstrip("0") or "0") if key.isdigit() else key


def find_coco_json(gt_dir):
    candidates = sorted(Path(gt_dir).rglob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No .json file found in {gt_dir} - pass the path with --coco")
    if len(candidates) > 1:
        names = ", ".join(str(c) for c in candidates)
        raise ValueError(f"Several .json files in {gt_dir} ({names}) - pass the right one with --coco")
    return candidates[0]


def load_coco(coco_path):
    """Return (images_by_id, annotations_by_image_id) keyed on the ID in the file name."""
    with open(coco_path) as f:
        coco = json.load(f)
    return coco["images"], coco["annotations"]


def index_model_folder(folder, pattern):
    by_id, no_id = defaultdict(list), []
    for f in sorted(Path(folder).iterdir()):
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
            continue
        key = extract_id(f.name, pattern)
        (no_id.append(f) if key is None else by_id[key].append(f))
    return by_id, no_id


def move_files(files, dest, dry_run):
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        target = dest / f.name
        i = 1
        while target.exists():
            target = dest / f"{f.stem}_{i}{f.suffix}"
            i += 1
        print(f"  {'[dry-run] ' if dry_run else ''}{f}  ->  {target}")
        if not dry_run:
            shutil.move(str(f), str(target))


def find_and_move_unpaired(model_dir=HERE / "images" / "modelSeg",
                           gt_dir=HERE / "images" / "groundTruth",
                           coco_path=None, out_dir=HERE / "unpaired", id_regex=r"\d+",
                           dry_run=False, require_annotations=False):
    """
    Move model images that have no ground truth and return the pairs as a list of dicts:
        {"id", "model_path", "gt_image" (COCO image entry), "annotations" (list)}
    require_annotations=True also treats COCO images with zero annotations as "no ground truth".
    """
    pattern = re.compile(id_regex)
    coco_path = Path(coco_path) if coco_path else find_coco_json(gt_dir)
    coco_images, coco_anns = load_coco(coco_path)

    anns_by_image = defaultdict(list)
    for a in coco_anns:
        anns_by_image[a["image_id"]].append(a)

    gt = defaultdict(list)
    gt_noid = []
    for img in coco_images:
        key = extract_id(Path(img["file_name"]).name, pattern)
        (gt_noid.append(img["file_name"]) if key is None else gt[key].append(img))
    model, model_noid = index_model_folder(model_dir, pattern)

    for name, idx in (("model folder", model), ("COCO file", gt)):
        for k, items in idx.items():
            if len(items) > 1:
                print(f"WARNING: ID {k} appears {len(items)} times in {name}; the first one is used")

    def has_gt(k):
        if k not in gt:
            return False
        return bool(anns_by_image[gt[k][0]["id"]]) or not require_annotations

    paired_ids = sorted(k for k in model if has_gt(k))
    model_only = [f for k in model if not has_gt(k) for f in model[k]]
    gt_only = [gt[k][0]["file_name"] for k in gt if k not in model]

    print(f"\nCOCO file: {coco_path}")
    print(f"Paired IDs: {len(paired_ids)}")
    print(f"Model images without ground truth: {len(model_only)}")
    print(f"Ground-truth entries without model image: {len(gt_only)}")
    print(f"Files with no ID in name: {len(model_noid) + len(gt_noid)} (left in place)\n")

    if model_only:
        print("Moving model images with no ground truth:")
        move_files(model_only, Path(out_dir) / "no_ground_truth", dry_run)
    if gt_only:
        print("Ground-truth entries with no model image (excluded from evaluation):")
        for name in gt_only:
            print(f"  {name}")
    for f in list(model_noid) + gt_noid:
        print(f"NOTE: no ID found in {f} - skipped")
    if dry_run:
        print("\nDry run only, nothing was moved.")

    return [{"id": k, "model_path": model[k][0], "gt_image": gt[k][0],
             "annotations": anns_by_image[gt[k][0]["id"]]} for k in paired_ids]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=HERE / "images" / "modelSeg")
    p.add_argument("--gt-dir", type=Path, default=HERE / "images" / "groundTruth")
    p.add_argument("--coco", type=Path, help="COCO json (default: the single .json found in --gt-dir)")
    p.add_argument("--out-dir", type=Path, default=HERE / "unpaired")
    p.add_argument("--id-regex", default=r"\d+")
    p.add_argument("--require-annotations", action="store_true",
                   help="treat COCO images with no annotations as having no ground truth")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    find_and_move_unpaired(a.model_dir, a.gt_dir, a.coco, a.out_dir, a.id_regex,
                           a.dry_run, a.require_annotations)


if __name__ == "__main__":
    main()