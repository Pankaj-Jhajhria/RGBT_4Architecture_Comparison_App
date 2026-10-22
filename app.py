from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import torch

from inference.model_loaders import MODEL_SPECS, load_model
from inference.preprocessing import prepare_rgb_thermal
from inference.visualization import CLASS_NAMES, CLASS_COLORS, overlay_mask, colorize_mask

ROOT = Path(__file__).resolve().parent
TARGET_SIZE = (640, 480)  # PIL uses (width, height); tensors use (height, width).

st.set_page_config(
    page_title="RGB–Thermal Segmentation | 4-Model Comparison",
    page_icon="▦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1600px;}
    h1, h2, h3 {letter-spacing: -0.025em;}
    .subtle {color: #8b929b; font-size: 0.94rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("RGB–Thermal Semantic Segmentation")
st.markdown(
    "Compare **MFNet, CSFNet-2, ResNet-18 + CMAF, and RTFNet18V2** using the same aligned scene. "
    "Choose whether your data is stored in one four-channel PNG or in two separate images."
)
st.caption(
    "Each model receives RGB and thermal data through its own expected input signature. "
    "Saved checkpoint metrics are training/validation metadata, not scores for the current uploaded image."
)

INPUT_4CH = "One four-channel PNG (RGB + thermal channel)"
INPUT_SEPARATE = "Two separate images (RGB + thermal)"

with st.sidebar:
    st.header("Input settings")
    input_mode = st.radio(
        "How are your inputs stored?",
        [INPUT_4CH, INPUT_SEPARATE],
        index=0,
        help="The MFNet dataset image PNG stores RGB in channels 1–3 and thermal intensity in channel 4.",
    )
    st.divider()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"**Inference device:** `{device.upper()}`")
    st.write("**Model input size:** `640 × 480` pixels")
    if input_mode == INPUT_4CH:
        st.info(
            "Upload the original PNG from `MFNet_dataset/images`, not a file from `labels` or `visual`. "
            "The app extracts RGB from channels 1–3 and thermal intensity from channel 4.",
            icon="ℹ️",
        )
    else:
        st.info(
            "Upload an ordinary RGB frame and its aligned raw thermal-intensity image. "
            "Do not use a ground-truth label or colorized mask as thermal input.",
            icon="ℹ️",
        )
    st.divider()
    st.caption("Fresh mIoU requires a matching ground-truth label mask. Predictions alone do not provide mIoU.")


def _open_uploaded_image(upload) -> Image.Image:
    """Open an upload from its bytes so repeated reads do not depend on file-pointer position."""
    image = Image.open(io.BytesIO(upload.getvalue()))
    image = ImageOps.exif_transpose(image)
    image.load()
    return image.copy()


rgb_image: Image.Image | None = None
thermal_image: Image.Image | None = None
input_error: str | None = None
input_signature_parts: list[object] = [input_mode]

if input_mode == INPUT_4CH:
    four_channel_upload = st.file_uploader(
        "Upload the original four-channel RGB+thermal PNG",
        type=["png", "tif", "tiff"],
        key="upload_four_channel_rgbt",
        help="Expected array shape: (480, 640, 4). The last channel is thermal intensity.",
    )
    if four_channel_upload is not None:
        input_signature_parts.extend([four_channel_upload.name, four_channel_upload.size])
        try:
            opened = _open_uploaded_image(four_channel_upload)
            raw = np.asarray(opened)
            if raw.ndim != 3 or raw.shape[2] != 4:
                raise ValueError(
                    f"This file has shape {raw.shape}, not a four-channel image. "
                    "Choose a four-channel PNG from `MFNet_dataset/images` or switch to separate-image mode."
                )
            if raw.dtype != np.uint8:
                if np.issubdtype(raw.dtype, np.integer) and raw.max(initial=0) <= 255:
                    raw = raw.astype(np.uint8)
                else:
                    raise ValueError(f"Expected 8-bit channel values, but found dtype {raw.dtype}.")

            # Important: the fourth channel is data-bearing thermal intensity in this mode,
            # not an ordinary visual transparency layer.
            rgb_image = Image.fromarray(np.ascontiguousarray(raw[:, :, :3]), mode="RGB")
            thermal_image = Image.fromarray(np.ascontiguousarray(raw[:, :, 3]), mode="L")
            st.success(
                f"Loaded four-channel image: {raw.shape[1]} × {raw.shape[0]} pixels. "
                "RGB uses channels 1–3; thermal uses channel 4."
            )
            thermal_arr = raw[:, :, 3]
            st.caption(
                f"Fourth-channel intensity range: {int(thermal_arr.min())}–{int(thermal_arr.max())}; "
                f"distinct intensity values: {len(np.unique(thermal_arr))}."
            )
            if len(np.unique(thermal_arr)) <= 1:
                input_error = "The fourth channel is constant, so it does not appear to contain usable thermal intensity data."
        except Exception as exc:
            input_error = f"Could not read this as a four-channel RGB+thermal image: {type(exc).__name__}: {exc}"
            st.error(input_error)
else:
    rgb_upload = st.file_uploader(
        "1. Upload visible-light RGB image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key="upload_separate_rgb",
    )
    thermal_upload = st.file_uploader(
        "2. Upload matching raw thermal-intensity image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key="upload_separate_thermal",
    )
    for upload in (rgb_upload, thermal_upload):
        input_signature_parts.extend([upload.name, upload.size] if upload is not None else [None, None])
    if rgb_upload is not None:
        try:
            rgb_image = _open_uploaded_image(rgb_upload).convert("RGB")
        except Exception as exc:
            input_error = f"Could not read the RGB image: {type(exc).__name__}: {exc}"
            st.error(input_error)
    if thermal_upload is not None:
        try:
            thermal_image = _open_uploaded_image(thermal_upload).convert("L")
        except Exception as exc:
            input_error = f"Could not read the thermal image: {type(exc).__name__}: {exc}"
            st.error(input_error)
    if rgb_image is not None and thermal_image is not None:
        thermal_values = np.unique(np.asarray(thermal_image, dtype=np.uint8))
        if len(thermal_values) <= 16:
            st.warning(
                f"The thermal image has only {len(thermal_values)} distinct grayscale values. "
                "It may be a label mask or colorized visualization rather than a raw thermal frame."
            )

if input_error:
    st.error("Correct the input selection before running inference.")

if rgb_image is not None and thermal_image is not None and input_error is None:
    rgb_preview, th_preview = st.columns(2, gap="large")
    with rgb_preview:
        st.image(rgb_image, caption=f"RGB input · {rgb_image.width} × {rgb_image.height}", use_container_width=True)
    with th_preview:
        st.image(thermal_image, caption=f"Thermal input · {thermal_image.width} × {thermal_image.height}", use_container_width=True)
    if rgb_image.size != thermal_image.size:
        st.warning(
            "RGB and thermal dimensions differ. Both will be resized to 640 × 480. "
            "Confirm that the images show the same scene and viewpoint before comparing results."
        )

# Remove results whenever mode or source uploads change.
input_signature = tuple(input_signature_parts)
if st.session_state.get("input_signature") != input_signature:
    st.session_state.pop("model_results", None)
    st.session_state["input_signature"] = input_signature


def _extract_logits(output: object) -> torch.Tensor:
    if torch.is_tensor(output):
        if output.ndim != 4:
            raise ValueError(f"Expected 4D segmentation logits, got shape {tuple(output.shape)}")
        return output
    if isinstance(output, dict):
        for key in ("out", "logits", "pred", "prediction"):
            if key in output and torch.is_tensor(output[key]) and output[key].ndim == 4:
                return output[key]
        candidates = [v for v in output.values() if torch.is_tensor(v) and v.ndim == 4]
        if candidates:
            return candidates[-1]
    if isinstance(output, (tuple, list)):
        candidates = [v for v in output if torch.is_tensor(v) and v.ndim == 4]
        if candidates:
            return candidates[-1]
    raise TypeError(f"Could not locate 4D logits in output type {type(output).__name__}")


can_run = rgb_image is not None and thermal_image is not None and input_error is None
run_button = st.button(
    "Run all four models",
    type="primary",
    use_container_width=True,
    disabled=not can_run,
)

if run_button:
    rgb_resized = rgb_image.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    thermal_resized = thermal_image.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    results: list[dict] = []
    progress = st.progress(0, text="Preparing model inputs…")

    for idx, model_spec in enumerate(MODEL_SPECS):
        progress.progress(idx / len(MODEL_SPECS), text=f"Running {model_spec['name']} ({idx + 1}/{len(MODEL_SPECS)})…")
        started = time.perf_counter()
        try:
            model, checkpoint_info = load_model(model_spec["id"], str(ROOT), device)
            rgb_tensor, thermal_tensor = prepare_rgb_thermal(
                model_spec["id"], rgb_resized, thermal_resized, checkpoint_info, device
            )

            # Feed each architecture in the signature its code was trained to accept.
            # The source image can be four-channel, but CSFNet and RTFNet18V2 use two arguments.
            with torch.inference_mode():
                if model_spec["id"] in ("csfnet", "rtfnet18v2"):
                    output = model(rgb_tensor, thermal_tensor)
                else:
                    output = model(torch.cat((rgb_tensor, thermal_tensor), dim=1))

            logits = _extract_logits(output)
            if logits.shape[-2:] != (TARGET_SIZE[1], TARGET_SIZE[0]):
                logits = torch.nn.functional.interpolate(
                    logits.float(), size=(TARGET_SIZE[1], TARGET_SIZE[0]), mode="bilinear", align_corners=False
                )
            mask = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
            results.append({
                "id": model_spec["id"],
                "ok": True,
                "seconds": time.perf_counter() - started,
                "info": checkpoint_info,
                "mask_image": colorize_mask(mask),
                "overlay_image": overlay_mask(rgb_resized, mask, alpha=0.48),
                "error": None,
            })
        except Exception as exc:
            results.append({
                "id": model_spec["id"],
                "ok": False,
                "seconds": time.perf_counter() - started,
                "info": {},
                "mask_image": None,
                "overlay_image": None,
                "error": f"{type(exc).__name__}: {exc}",
            })
    progress.progress(1.0, text="Finished attempting all four models")
    progress.empty()
    st.session_state["model_results"] = results

results = st.session_state.get("model_results")
if results:
    by_id = {item["id"]: item for item in results}
    successful = sum(1 for item in results if item["ok"])
    st.divider()
    st.subheader("Model predictions")
    if successful == len(MODEL_SPECS):
        st.success("All four models produced a prediction.")
    else:
        st.warning(f"{successful} of {len(MODEL_SPECS)} models produced predictions. Failed model details are shown in their own panels below.")

    # Fixed 2x2 display ensures all four architectures always have a panel.
    for row_start in range(0, len(MODEL_SPECS), 2):
        left, right = st.columns(2, gap="large")
        for column, model_spec in zip((left, right), MODEL_SPECS[row_start:row_start + 2]):
            item = by_id.get(model_spec["id"])
            with column:
                with st.container(border=True):
                    st.subheader(model_spec["name"])
                    if item is None:
                        st.error("No result was recorded for this model. Run all four models again.")
                    elif not item["ok"]:
                        st.error("This model failed to load or run.")
                        st.code(item["error"])
                        st.caption(f"Expected checkpoint: {model_spec['weight_rel']}")
                    else:
                        st.caption(f"Inference: {item['seconds']:.2f} s · {model_spec['short']}")
                        if item["info"].get("display_metric"):
                            st.caption(item["info"]["display_metric"])
                        mask_col, overlay_col = st.columns(2, gap="small")
                        with mask_col:
                            st.image(item["mask_image"], caption="Predicted mask", use_container_width=True)
                        with overlay_col:
                            st.image(item["overlay_image"], caption="RGB overlay", use_container_width=True)

    with st.expander("Shared nine-class legend"):
        legend_cols = st.columns(3)
        for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
            with legend_cols[i % 3]:
                hex_color = "#%02X%02X%02X" % tuple(int(v) for v in color)
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
                    f'<span style="display:inline-block;width:14px;height:14px;background:{hex_color};border:1px solid #888;border-radius:2px;"></span>'
                    f'<span>{name}</span></div>',
                    unsafe_allow_html=True,
                )
