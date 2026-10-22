from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def _as_vector(value, channels: int, fallback):
    if value is None:
        value = fallback
    if isinstance(value, (float, int)):
        values = [float(value)]
    else:
        values = [float(x) for x in value]
    if len(values) == 1:
        values = values * channels
    if len(values) != channels:
        values = [float(x) for x in fallback]
    return values


def _tensor_from_image(image: Image.Image, mean, std, channels: int) -> torch.Tensor:
    mode = "RGB" if channels == 3 else "L"
    arr = np.asarray(image.convert(mode), dtype=np.float32) / 255.0
    if channels == 1:
        arr = arr[..., None]
    arr = np.transpose(arr, (2, 0, 1))
    mean_values = _as_vector(mean, channels, [0.485, 0.456, 0.406] if channels == 3 else [0.5])
    std_values = _as_vector(std, channels, [0.229, 0.224, 0.225] if channels == 3 else [0.25])
    mean_array = np.asarray(mean_values, dtype=np.float32).reshape(channels, 1, 1)
    std_array = np.maximum(np.asarray(std_values, dtype=np.float32), 1e-6).reshape(channels, 1, 1)
    arr = (arr - mean_array) / std_array
    return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)


def prepare_rgb_thermal(model_id: str, rgb_image: Image.Image, thermal_image: Image.Image, metadata: dict, device: str):
    """Return normalized RGB and thermal float tensors on the selected device."""
    rgb = _tensor_from_image(
        rgb_image,
        metadata.get("rgb_mean", [0.485, 0.456, 0.406]),
        metadata.get("rgb_std", [0.229, 0.224, 0.225]),
        3,
    ).to(device)
    thermal = _tensor_from_image(
        thermal_image,
        metadata.get("thermal_mean", [0.5]),
        metadata.get("thermal_std", [0.25]),
        1,
    ).to(device)
    return rgb, thermal
