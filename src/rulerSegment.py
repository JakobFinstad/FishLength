from pathlib import Path
from PIL import Image
import numpy as np
import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

INPUT = Path("/mnt/c/Users/jakob/Pictures/ALL_IMAGES_RULER/Measured")
OUTPUT = Path("/mnt/c/Users/jakob/Pictures/SEGMENTED_IMAGES")
OUTPUT.mkdir(parents=True, exist_ok=True)

model = build_sam3_image_model()
processor = Sam3Processor(model)

for path in INPUT.iterdir():
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

    image = Image.open(path).convert("RGB")

    # SAM 3 inference
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        result = processor.set_text_prompt(state=state, prompt="ruler")

    masks = result["masks"]

    if len(masks) > 0:
        mask = masks.any(dim=0).squeeze().cpu().numpy()

        img = np.array(image)
        img[~mask] = 0

        Image.fromarray(img).save(
            OUTPUT / f"{path.stem}_segmented.png"
        )

    print(f"{path.name}: {len(masks)} ruler(s)")