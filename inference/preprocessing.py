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


def _raw_tensor(image: Image.Image, channels: int) -> torch.Tensor:
    """Convert an image to NCHW float tensor scaled to [0, 1], without normalization."""
    mode = "RGB" if channels == 3 else "L"
    arr = np.asarray(image.convert(mode), dtype=np.float32) / 255.0
    if channels == 1:
        arr = arr[..., None]
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)


def _tensor_from_image(image: Image.Image, mean, std, channels: int) -> torch.Tensor:
    """Original mean/std preprocessing retained unchanged for the other architectures."""
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
    """Prepare image tensors following each model's actual training data pipeline."""
    if model_id == "mfnet":
        # MFNet's dataset.py uses ToTensorV2() then divides all four uint8 channels
        # by 255. It does NOT apply the RGB_MEAN/RGB_STD/THR_MEAN/THR_STD constants.
        # Keep the RGB and thermal tensors raw in [0, 1] so concatenated RGB-T input
        # matches the distribution seen during MFNet training.
        rgb = _raw_tensor(rgb_image, 3)
        thermal = _raw_tensor(thermal_image, 1)
    else:
        # Intentionally preserve the pre-existing code path for CSFNet-2, CMAF,
        # and RTFNet18V2. This MFNet-specific fix does not change their preprocessing.
        rgb = _tensor_from_image(
            rgb_image,
            metadata.get("rgb_mean", [0.485, 0.456, 0.406]),
            metadata.get("rgb_std", [0.229, 0.224, 0.225]),
            3,
        )
        thermal = _tensor_from_image(
            thermal_image,
            metadata.get("thermal_mean", [0.5]),
            metadata.get("thermal_std", [0.25]),
            1,
        )
    return rgb.to(device), thermal.to(device)
