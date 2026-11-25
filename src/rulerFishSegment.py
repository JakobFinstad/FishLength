import csv
from pathlib import Path
from PIL import Image
from workflow import measure_image
import numpy as np
import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

INPUT = Path("/mnt/c/Users/jakob/Pictures/ALL_IMAGES_RULER/Measured")
OUTPUT = Path("/mnt/c/Users/jakob/Pictures/SEGMENTED_IMAGES")
OUTPUT.mkdir(parents=True, exist_ok=True)

model = build_sam3_image_model()
processor = Sam3Processor(model)

def segment_image(image, processor, prompts=("ruler", "fish"), min_score=0.5):
    out = {}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        for p in prompts:
            res = processor.set_text_prompt(state=state, prompt=p)
            masks = res["masks"].squeeze(1).cpu().numpy().astype(bool)  # (N,H,W)
            scores = res["scores"].float().cpu().numpy()
            keep = scores >= min_score
            out[p] = masks[keep]
    return out

def segment(InPath=INPUT, OutPath=OUTPUT):
    results = {}
    rows = []
    for path in sorted(InPath.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        image = Image.open(path).convert("RGB")
        per_class = segment_image(image, processor)   # {"ruler": (N,H,W), "fish": (M,H,W)}
        m = measure_image(per_class)
        if m is None:
            print(f"{path.name}: no ruler or no fish, skipped")
        else:
            for i, f in enumerate(m["fish"]):
                rows.append({"image": path.name, "fish": i, "orientation": m["orientation"],
                     "ruler_ok": m["ruler_ok"], **f})
        if rows:                       # <-- add this after the loop
            with open(OutPath / "measurements.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"Wrote {len(rows)} measurements to {OutPath / 'measurements.csv'}")
        results[path.stem] = per_class

        h, w = image.height, image.width
        union = np.zeros((h, w), dtype=bool)

        for name, masks in per_class.items():
            # one mask per class (union of all instances of that class)
            class_mask = masks.any(axis=0) if len(masks) else np.zeros((h, w), dtype=bool)
            Image.fromarray(class_mask.astype(np.uint8) * 255).save(
                OutPath / f"{path.stem}_{name}_mask.png"
            )
            union |= class_mask

        # black-background image for evaluateSegment.py (ruler + fish combined)
        img = np.array(image)
        img[~union] = 0
        Image.fromarray(img).save(OutPath / f"{path.stem}_segmented.png")

        print(f"{path.name}: {len(per_class['ruler'])} ruler(s), {len(per_class['fish'])} fish")

    return results