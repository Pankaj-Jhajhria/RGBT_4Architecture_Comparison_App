from __future__ import annotations

import numpy as np
from PIL import Image

CLASS_NAMES = [
    "Unlabeled", "Car", "Person", "Bike", "Curve",
    "Car stop", "Guardrail", "Color cone", "Bump",
]

# Shared MFNet nine-class palette for masks from all four models.
CLASS_COLORS = np.asarray([
    [40, 40, 40],       # unlabeled / ignored-looking background
    [220, 20, 60],      # car
    [65, 105, 225],     # person
    [218, 165, 32],     # bike
    [138, 43, 226],     # curve
    [255, 69, 0],       # car stop
    [0, 206, 209],      # guardrail
    [255, 127, 80],     # color cone
    [50, 205, 50],      # bump
], dtype=np.uint8)


def colorize_mask(mask: np.ndarray) -> Image.Image:
    mask = np.asarray(mask, dtype=np.int64)
    safe = np.clip(mask, 0, len(CLASS_COLORS) - 1)
    return Image.fromarray(CLASS_COLORS[safe], mode="RGB")


def overlay_mask(rgb_image: Image.Image, mask: np.ndarray, alpha: float = 0.48) -> Image.Image:
    rgb = np.asarray(rgb_image.convert("RGB"), dtype=np.float32)
    color = np.asarray(colorize_mask(mask), dtype=np.float32)
    blended = rgb * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")
