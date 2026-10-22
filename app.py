from __future__ import annotations

import time
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps
import torch

from inference.model_loaders import MODEL_SPECS, load_model
from inference.preprocessing import prepare_rgb_thermal
from inference.visualization import CLASS_NAMES, overlay_mask, colorize_mask

ROOT = Path(__file__).resolve().parent
TARGET_SIZE = (640, 480)  # width, height; model input is 480 x 640

st.set_page_config(
    page_title="RGB–Thermal Segmentation | 4-Model Comparison",
    page_icon="▦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1500px;}
    h1, h2, h3 {letter-spacing: -0.025em;}
    .subtle {color: #626A73; font-size: 0.94rem;}
    .model-title {font-weight: 650; font-size: 1.12rem; margin-bottom: .25rem;}
    div[data-testid="stMetric"] {background: #F6F7F8; border: 1px solid #E6E8EB; padding: .7rem .8rem; border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("RGB–Thermal Semantic Segmentation")
st.markdown(
    "Compare four trained architectures on the **same RGB image and thermal frame**. "
    "Each model uses its own architecture, checkpoint and preprocessing."
)
st.caption("MFNet · CSFNet-2 · ResNet-18 + CMAF · RTFNet18V2")

with st.sidebar:
    st.header("Input settings")
    st.write("Upload a visible-light RGB image and its aligned single-channel thermal image.")
    st.info(
        "Use the raw thermal-intensity image if possible, not a coloured Inferno/Jet display. "
        "The two images should show the same scene and viewpoint.",
        icon="ℹ️",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"**Inference device:** `{device.upper()}`")
    st.write("**Model input:** `480 × 640` pixels")
    st.divider()
    st.caption("Predictions are visual outputs. Without ground-truth masks, this app does not calculate mIoU for the uploaded scene.")

col_rgb, col_thermal = st.columns(2, gap="large")
with col_rgb:
    rgb_upload = st.file_uploader("1. RGB image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="rgb")
with col_thermal:
    thermal_upload = st.file_uploader("2. Thermal image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="thermal")

rgb_image = None
thermal_image = None
if rgb_upload is not None:
    try:
        rgb_image = ImageOps.exif_transpose(Image.open(rgb_upload)).convert("RGB")
    except Exception as exc:
        st.error(f"Could not read the RGB image: {exc}")
if thermal_upload is not None:
    try:
        thermal_image = ImageOps.exif_transpose(Image.open(thermal_upload)).convert("L")
    except Exception as exc:
        st.error(f"Could not read the thermal image: {exc}")

if rgb_image is not None and thermal_image is not None:
    left, right = st.columns(2, gap="large")
    with left:
        st.image(rgb_image, caption=f"RGB input · {rgb_image.width} × {rgb_image.height}", use_container_width=True)
    with right:
        st.image(thermal_image, caption=f"Thermal input · {thermal_image.width} × {thermal_image.height}", use_container_width=True)
    if rgb_image.size != thermal_image.size:
        st.warning(
            "The RGB and thermal image dimensions differ. The app will resize both to 640 × 480. "
            "Check that they are spatially aligned before comparing model outputs."
        )


def _extract_logits(output):
    """Return the 4D logits tensor from common segmentation model outputs."""
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


run_button = st.button(
    "Run all four models",
    type="primary",
    use_container_width=True,
    disabled=(rgb_image is None or thermal_image is None),
)

if run_button:
    rgb_resized = rgb_image.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    thermal_resized = thermal_image.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    st.divider()
    st.subheader("Model predictions")
    st.markdown('<p class="subtle">The same image pair is supplied to every model. Masks use a shared class-colour palette for easier visual comparison.</p>', unsafe_allow_html=True)

    results = []
    progress = st.progress(0, text="Preparing inputs…")
    for idx, spec in enumerate(MODEL_SPECS):
        progress.progress(idx / len(MODEL_SPECS), text=f"Running {spec['name']}…")
        start = time.perf_counter()
        try:
            model, checkpoint_info = load_model(spec["id"], str(ROOT), device)
            rgb_tensor, thermal_tensor = prepare_rgb_thermal(
                spec["id"], rgb_resized, thermal_resized, checkpoint_info, device
            )
            with torch.inference_mode():
                if spec["id"] in ("csfnet", "rtfnet18v2"):
                    output = model(rgb_tensor, thermal_tensor)
                else:
                    # MFNet and CMAF receive a concatenated [RGB, thermal] tensor.
                    output = model(torch.cat((rgb_tensor, thermal_tensor), dim=1))
            logits = _extract_logits(output)
            if logits.shape[-2:] != (TARGET_SIZE[1], TARGET_SIZE[0]):
                logits = torch.nn.functional.interpolate(
                    logits.float(), size=(TARGET_SIZE[1], TARGET_SIZE[0]),
                    mode="bilinear", align_corners=False
                )
            mask = logits.argmax(dim=1)[0].detach().cpu().numpy().astype("uint8")
            elapsed = time.perf_counter() - start
            results.append({
                "spec": spec,
                "mask": mask,
                "seconds": elapsed,
                "info": checkpoint_info,
                "error": None,
            })
        except Exception as exc:
            results.append({
                "spec": spec,
                "mask": None,
                "seconds": time.perf_counter() - start,
                "info": {},
                "error": f"{type(exc).__name__}: {exc}",
            })
    progress.progress(1.0, text="Inference complete")
    progress.empty()

    ok = [item for item in results if item["error"] is None]
    failed = [item for item in results if item["error"] is not None]

    if ok:
        st.success(f"Completed {len(ok)} of {len(MODEL_SPECS)} model predictions.")
        for start_idx in range(0, len(ok), 2):
            model_cols = st.columns(2, gap="large")
            for col, item in zip(model_cols, ok[start_idx:start_idx + 2]):
                with col:
                    spec = item["spec"]
                    mask_img = colorize_mask(item["mask"])
                    overlay_img = overlay_mask(rgb_resized, item["mask"], alpha=0.48)
                    st.markdown(f'<div class="model-title">{spec["name"]}</div>', unsafe_allow_html=True)
                    st.caption(f"Inference: {item['seconds']:.2f} s · {spec['short']}")
                    meta = item["info"].get("display_metric")
                    if meta:
                        st.caption(meta)
                    img_cols = st.columns(2, gap="small")
                    with img_cols[0]:
                        st.image(mask_img, caption="Predicted mask", use_container_width=True)
                    with img_cols[1]:
                        st.image(overlay_img, caption="RGB overlay", use_container_width=True)
                    with st.expander("Class pixel counts"):
                        counts = {CLASS_NAMES[c]: int((item["mask"] == c).sum()) for c in range(len(CLASS_NAMES))}
                        total = max(1, item["mask"].size)
                        st.dataframe(
                            [{"Class": name, "Pixels": count, "Share": f"{count / total:.2%}"}
                             for name, count in counts.items()],
                            hide_index=True, use_container_width=True,
                        )
        st.divider()
        st.subheader("Shared class legend")
        legend_cols = st.columns(3)
        from inference.visualization import CLASS_COLORS
        for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
            with legend_cols[i % len(legend_cols)]:
                hex_color = "#%02X%02X%02X" % tuple(color)
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
                    f'<span style="display:inline-block;width:14px;height:14px;background:{hex_color};border:1px solid #888;border-radius:2px;"></span>'
                    f'<span>{name}</span></div>', unsafe_allow_html=True
                )
    if failed:
        st.warning(f"{len(failed)} model(s) could not run. Other model results are still available.")
        for item in failed:
            with st.expander(f"Error: {item['spec']['name']}", expanded=True):
                st.code(item["error"])
                st.caption(f"Expected checkpoint: {item['spec']['weight_rel']}")


