from __future__ import annotations

import csv
import hashlib
import io
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from inference.model_loaders import MODEL_SPECS, get_model_asset_status, load_model
from inference.preprocessing import prepare_rgb_thermal
from inference.visualization import CLASS_NAMES, CLASS_COLORS, colorize_mask

ROOT = Path(__file__).resolve().parent
TARGET_SIZE = (640, 480)  # PIL size = (width, height); tensors use (height, width).
IGNORE_INDEX = 255

# Source label palette from the MFNet training repository, plus the app's existing
# shared palette so class-ID masks downloaded from this app can be re-evaluated.
MFNET_LABEL_COLORS = np.asarray(
    [
        [0, 0, 0],
        [220, 20, 60],
        [0, 200, 0],
        [30, 144, 255],
        [255, 215, 0],
        [255, 20, 147],
        [0, 255, 255],
        [255, 140, 0],
        [138, 43, 226],
    ],
    dtype=np.uint8,
)

st.set_page_config(
    page_title="RGB-T Lab | Four-Model Benchmark",
    page_icon="RGBT",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --rgbt-bg: #f4f6f9; --rgbt-panel: #ffffff; --rgbt-border: #dbe1e8;
      --rgbt-text: #172337; --rgbt-muted: #627086; --rgbt-accent: #2458a6;
    }
    .stApp { background: var(--rgbt-bg); color: var(--rgbt-text); }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--rgbt-border); }
    [data-testid="stAppViewContainer"] .main .block-container { padding-top: 1.25rem; padding-bottom: 3rem; max-width: 1680px; }
    h1,h2,h3 { color: var(--rgbt-text) !important; letter-spacing: -.02em; }
    h1 { font-size: clamp(1.9rem,2.7vw,2.65rem) !important; font-weight: 750 !important; }
    h2 { font-size: 1.45rem !important; font-weight: 700 !important; }
    h3 { font-size: 1.12rem !important; font-weight: 700 !important; }
    p, label, [data-testid="stMarkdownContainer"] { color: var(--rgbt-text); }
    [data-testid="stCaptionContainer"], .rgbt-muted { color: var(--rgbt-muted) !important; }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid var(--rgbt-border); padding: 12px 14px; border-radius: 8px; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--rgbt-border) !important; border-radius: 10px !important; background: #ffffff; }
    .stButton > button, .stDownloadButton > button { border-radius: 6px; font-weight: 650; border-color: #cbd4df; min-height: 2.55rem; box-shadow: none; }
    .stButton > button[kind="primary"] { background: var(--rgbt-accent); border: 1px solid var(--rgbt-accent); color: #ffffff; }
    .stButton > button:hover, .stDownloadButton > button:hover { border-color: #2458a6; color: #173d78; }
    .stButton > button[kind="primary"]:hover { background: #1c488e; color: #ffffff; }
    [data-testid="stFileUploader"] { border: 1px dashed #c5cfdd; border-radius: 8px; padding: 4px; background: #ffffff; }
    [data-testid="stImage"] img { border-radius: 6px; }
    .rgbt-hero { padding: .25rem 0 1rem 0; margin-bottom: .75rem; border-bottom: 1px solid var(--rgbt-border); background: transparent; }
    .rgbt-eyebrow { color: var(--rgbt-accent); font-size: .72rem; font-weight: 750; text-transform: uppercase; letter-spacing: .09em; margin-bottom: .45rem; }
    .rgbt-hero-title { font-size: clamp(1.8rem,2.6vw,2.6rem); font-weight: 780; line-height: 1.15; letter-spacing: -.03em; color: #172337; }
    .rgbt-hero-sub { color: #526176; font-size: .97rem; max-width: 1050px; margin-top: .55rem; line-height: 1.55; }
    .rgbt-chip { display: inline-block; margin: .65rem .35rem 0 0; padding: .2rem .5rem; border: 1px solid var(--rgbt-border); border-radius: 4px; color: #40516a; background: #ffffff; font-size: .75rem; }
    .rgbt-status { padding: .65rem .75rem; border: 1px solid var(--rgbt-border); border-radius: 7px; background: #ffffff; margin: .35rem 0; }
    .rgbt-status small { color: var(--rgbt-muted); overflow-wrap: anywhere; }
    .rgbt-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 7px; vertical-align: middle; }
    .rgbt-dot-good { background: #27845c; }
    .rgbt-dot-bad { background: #c44545; }
    .rgbt-divider { height: 1px; width: 100%; margin: 1.1rem 0; background: var(--rgbt-border); }
    div[data-testid="stExpander"] { border-color: var(--rgbt-border); border-radius: 8px; background: #ffffff; }
    div[role="radiogroup"] { gap: .4rem; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="rgbt-hero">
      <div class="rgbt-eyebrow">RGB–thermal perception · Research workbench</div>
      <div class="rgbt-hero-title">Semantic Segmentation Benchmark</div>
      <div class="rgbt-hero-sub">Compare four trained architectures on one aligned scene. Review predicted masks and RGB overlays, inspect class coverage, and optionally evaluate against ground-truth labels.</div>
      <span class="rgbt-chip">MFNet</span><span class="rgbt-chip">CSFNet-2</span><span class="rgbt-chip">ResNet-18 + CMAF</span><span class="rgbt-chip">RTFNet18V2</span>
    </section>
    """,
    unsafe_allow_html=True,
)

st.caption("Saved checkpoint metrics describe the recorded validation run. Scene-specific mIoU is only shown when you provide a matching ground-truth mask.")


def _open_uploaded_image(upload: Any) -> Image.Image:
    image = Image.open(io.BytesIO(upload.getvalue()))
    image = ImageOps.exif_transpose(image)
    image.load()
    return image.copy()


def _upload_signature(upload: Any) -> tuple[Any, ...]:
    if upload is None:
        return (None,)
    raw = upload.getvalue()
    return upload.name, len(raw), hashlib.sha1(raw).hexdigest()[:16]


def _dataset_roots() -> list[Path]:
    return [
        ROOT.parent / "MFNet_dataset",
        ROOT / "MFNet_dataset",
        ROOT.parent / "MFNet-dataset" / "ir_seg_dataset",
        ROOT.parent / "MFNet-dataset",
    ]


def _find_dataset_root() -> Path | None:
    for root in _dataset_roots():
        if (root / "images").is_dir() and (root / "labels").is_dir():
            return root
    return None


def _load_dataset_rgba(path: Path) -> tuple[Image.Image, Image.Image, np.ndarray]:
    with Image.open(path) as image:
        raw = np.asarray(image).copy()
    if raw.ndim != 3 or raw.shape[2] != 4:
        raise ValueError(f"Expected a four-channel image, got {raw.shape} from {path}")
    if raw.dtype != np.uint8:
        if np.issubdtype(raw.dtype, np.integer) and raw.max(initial=0) <= 255:
            raw = raw.astype(np.uint8)
        else:
            raise ValueError(f"Expected 8-bit channel values in {path}, got {raw.dtype}")
    return (
        Image.fromarray(np.ascontiguousarray(raw[..., :3]), mode="RGB"),
        Image.fromarray(np.ascontiguousarray(raw[..., 3]), mode="L"),
        raw,
    )


def _read_mask_image(image: Image.Image, source: str = "ground-truth mask") -> np.ndarray:
    arr = np.asarray(image).copy()
    if arr.ndim == 2:
        mask = arr.astype(np.int64)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[..., :3]
        if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2]):
            mask = rgb[..., 0].astype(np.int64)
        else:
            # Support native MFNet palette masks and masks downloaded from this UI.
            mask = np.full(rgb.shape[:2], IGNORE_INDEX, dtype=np.int64)
            palettes = [MFNET_LABEL_COLORS, np.asarray(CLASS_COLORS, dtype=np.uint8)]
            for palette in palettes:
                for class_id, color in enumerate(palette[: len(CLASS_NAMES)]):
                    mask[np.all(rgb == color, axis=-1)] = class_id
            unknown_ratio = float(np.mean(mask == IGNORE_INDEX))
            if unknown_ratio > 0.02:
                raise ValueError(
                    f"{source} looks like a colour image but {unknown_ratio:.1%} of pixels do not match a supported class palette. "
                    "Use the original grayscale labels PNG (values 0–8, optional 255 ignore index)."
                )
    else:
        raise ValueError(f"Unsupported mask shape {arr.shape} in {source}")

    allowed = np.isin(mask, list(range(len(CLASS_NAMES))) + [IGNORE_INDEX])
    if not np.all(allowed):
        invalid = np.unique(mask[~allowed])[:12].tolist()
        raise ValueError(
            f"{source} contains unexpected IDs {invalid}. Expected class IDs 0–8 and optional ignore ID 255. "
            "Do not use a photo or thermal image as ground truth."
        )
    return mask.astype(np.uint8, copy=False)


def _resize_mask(mask: np.ndarray) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8), mode="L")
    return np.asarray(image.resize(TARGET_SIZE, Image.Resampling.NEAREST), dtype=np.uint8)


def _false_color_thermal(image: Image.Image, palette_name: str) -> Image.Image:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if palette_name == "Grayscale":
        return Image.fromarray(gray, mode="L").convert("RGB")
    anchors = {
        "Inferno": [(0, 0, 4), (70, 15, 95), (180, 45, 95), (247, 130, 35), (252, 255, 164)],
        "Magma": [(0, 0, 4), (75, 20, 100), (180, 55, 120), (247, 135, 100), (252, 253, 191)],
        "Viridis": [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
    }[palette_name]
    x = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    xi = gray.astype(np.float32) / 255.0
    color = np.stack(
        [np.interp(xi, x, np.asarray(anchors, dtype=np.float32)[:, c]) for c in range(3)],
        axis=-1,
    ).clip(0, 255).astype(np.uint8)
    return Image.fromarray(color, mode="RGB")


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _mask_id_png(mask: np.ndarray) -> bytes:
    """Export real class IDs in grayscale PNG, not a colorized preview."""
    return _png_bytes(Image.fromarray(mask.astype(np.uint8), mode="L"))


def _foreground_overlay(rgb_image: Image.Image, mask: np.ndarray, alpha: float = 0.48) -> Image.Image:
    """Blend only foreground predictions; class 0/ignore pixels leave RGB untouched."""
    rgb = np.asarray(rgb_image.convert("RGB"), dtype=np.float32)
    colors = np.asarray(colorize_mask(mask), dtype=np.float32)
    result = rgb.copy()
    foreground = (mask > 0) & (mask < len(CLASS_NAMES)) & (mask != IGNORE_INDEX)
    result[foreground] = (1.0 - alpha) * rgb[foreground] + alpha * colors[foreground]
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")


def _class_counts(mask: np.ndarray) -> list[dict[str, Any]]:
    valid = mask[(mask >= 0) & (mask < len(CLASS_NAMES))]
    counts = np.bincount(valid.astype(np.int64), minlength=len(CLASS_NAMES))
    total = int(counts.sum())
    return [
        {"Class": name, "Pixels": int(count), "Coverage": f"{(count / max(total, 1)) * 100:.2f}%"}
        for name, count in zip(CLASS_NAMES, counts)
    ]


def _segmentation_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    valid = (target >= 0) & (target < len(CLASS_NAMES)) & (target != IGNORE_INDEX)
    valid &= (prediction >= 0) & (prediction < len(CLASS_NAMES))
    if not np.any(valid):
        return {"miou": None, "foreground_miou": None, "pixel_accuracy": None, "per_class_iou": []}
    p = prediction[valid].astype(np.int64)
    t = target[valid].astype(np.int64)
    confusion = np.bincount(t * len(CLASS_NAMES) + p, minlength=len(CLASS_NAMES) ** 2).reshape(len(CLASS_NAMES), len(CLASS_NAMES))
    tp = np.diag(confusion).astype(np.float64)
    gt_sum = confusion.sum(axis=1).astype(np.float64)
    pred_sum = confusion.sum(axis=0).astype(np.float64)
    union = gt_sum + pred_sum - tp
    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
    all_valid = np.isfinite(iou)
    fg_valid = all_valid.copy()
    fg_valid[0] = False
    return {
        "miou": float(np.nanmean(iou)) if all_valid.any() else None,
        "foreground_miou": float(np.nanmean(iou[fg_valid])) if fg_valid.any() else None,
        "pixel_accuracy": float(tp.sum() / max(float(confusion.sum()), 1.0)),
        "per_class_iou": [None if not np.isfinite(value) else float(value) for value in iou],
    }


def _zip_results(results: list[dict[str, Any]], rgb: Image.Image, alpha: float) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("input/rgb.png", _png_bytes(rgb.convert("RGB")))
        rows = []
        for item in results:
            if not item.get("ok"):
                continue
            model_id = item["id"]
            mask = item["mask"]
            color_mask = colorize_mask(mask)
            overlay = _foreground_overlay(rgb, mask, alpha=alpha)
            zf.writestr(f"predictions/{model_id}_class_ids.png", _mask_id_png(mask))
            zf.writestr(f"predictions/{model_id}_color_mask.png", _png_bytes(color_mask))
            zf.writestr(f"predictions/{model_id}_overlay.png", _png_bytes(overlay))
            rows.append({
                "model": item.get("name", model_id),
                "inference_seconds": f"{item['seconds']:.4f}",
                "mean_max_probability": f"{item.get('confidence', 0.0):.6f}",
                "checkpoint_metric": item.get("info", {}).get("display_metric") or "",
                "checkpoint": item.get("info", {}).get("actual_checkpoint_path") or "",
            })
        if rows:
            text_buffer = io.StringIO()
            writer = csv.DictWriter(text_buffer, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            zf.writestr("summary.csv", text_buffer.getvalue())
        zf.writestr(
            "README.txt",
            "RGB-T semantic segmentation outputs. *_class_ids.png contains grayscale class IDs 0-8 (255 is ignore if present). "
            "*_color_mask.png is for visual inspection. *_overlay.png blends the prediction over the RGB input. "
            "Checkpoint metrics are recorded validation metadata; they do not evaluate this uploaded image.\n",
        )
    return archive.getvalue()


def _extract_logits(output: object) -> torch.Tensor:
    if torch.is_tensor(output):
        if output.ndim != 4:
            raise ValueError(f"Expected 4D segmentation logits, got shape {tuple(output.shape)}")
        return output
    if isinstance(output, dict):
        for key in ("out", "logits", "pred", "prediction"):
            if key in output and torch.is_tensor(output[key]) and output[key].ndim == 4:
                return output[key]
        candidates = [value for value in output.values() if torch.is_tensor(value) and value.ndim == 4]
        if candidates:
            return candidates[-1]
    if isinstance(output, (tuple, list)):
        candidates = [value for value in output if torch.is_tensor(value) and value.ndim == 4]
        if candidates:
            return candidates[-1]
    raise TypeError(f"Could not locate 4D logits in output type {type(output).__name__}")


def _make_histogram_image(values: np.ndarray) -> Image.Image:
    """Render the thermal histogram with Pillow (no browser chart/CSS assets required)."""
    from PIL import ImageDraw, ImageFont

    values = np.asarray(values, dtype=np.uint8).reshape(-1)
    hist = np.bincount(values, minlength=256)
    width, height = 900, 270
    left, right, top, bottom = 70, 22, 22, 44
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    plot_w, plot_h = width - left - right, height - top - bottom
    max_count = max(int(hist.max()), 1)

    for i in range(5):
        y = top + round(plot_h * i / 4)
        count = round(max_count * (1 - i / 4))
        draw.line((left, y, width - right, y), fill="#e7ebf0", width=1)
        draw.text((8, y - 5), f"{count:,}", fill="#667085", font=font)
    draw.line((left, top, left, height - bottom), fill="#8a96a8", width=1)
    draw.line((left, height - bottom, width - right, height - bottom), fill="#8a96a8", width=1)

    bar_w = plot_w / 256
    for intensity, count in enumerate(hist):
        bar_h = (int(count) / max_count) * plot_h
        x0 = left + int(intensity * bar_w)
        x1 = left + max(int(intensity * bar_w + bar_w) - 1, int(intensity * bar_w))
        y0 = top + plot_h - int(bar_h)
        draw.rectangle((x0, y0, x1, top + plot_h), fill="#4774a8")

    for value in (0, 64, 128, 192, 255):
        x = left + round(plot_w * value / 255)
        draw.line((x, height - bottom, x, height - bottom + 4), fill="#8a96a8", width=1)
        draw.text((x - 8, height - bottom + 8), str(value), fill="#667085", font=font)
    draw.text((width // 2 - 55, height - 17), "Thermal intensity (0–255)", fill="#475467", font=font)
    return canvas


# -----------------------------------------------------------------------------
# Sidebar controls and model-artifact status
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Workspace settings")
    device_options = ["Auto"] + (["CUDA"] if torch.cuda.is_available() else []) + ["CPU"]
    device_choice = st.selectbox("Inference device", device_options, index=0)
    device = "cuda" if device_choice == "CUDA" or (device_choice == "Auto" and torch.cuda.is_available()) else "cpu"
    st.caption(f"Active device: **{device.upper()}**")
    if device == "cuda":
        try:
            st.caption(torch.cuda.get_device_name(0))
        except Exception:
            pass
    overlay_alpha = st.slider("Overlay opacity", min_value=0.10, max_value=0.85, value=0.48, step=0.05)
    thermal_palette = st.selectbox("Thermal preview palette", ["Grayscale", "Inferno", "Magma", "Viridis"], index=0)
    show_diagnostics = st.toggle("Show input diagnostics", value=True)
    show_pixel_tables = st.toggle("Show per-class pixel counts", value=True)
    sidebar_action_slot = st.empty()
    st.divider()
    st.markdown("### Checkpoint status")
    for spec in MODEL_SPECS:
        asset = get_model_asset_status(spec["id"], ROOT)
        if asset["available"]:
            badge = '<span class="rgbt-dot rgbt-dot-good"></span>'
            suffix = " · fallback" if asset.get("is_legacy") else ""
        else:
            badge = '<span class="rgbt-dot rgbt-dot-bad"></span>'
            suffix = " · missing"
        st.markdown(
            f'<div class="rgbt-status">{badge}<b>{spec["name"]}</b>{suffix}<br>'
            f'<small>{asset.get("variant", "")}</small><br><small>{asset.get("checkpoint", "")}</small></div>',
            unsafe_allow_html=True,
        )
        if spec["id"] == "cmaf" and asset.get("is_legacy"):
            st.warning(
                "CMAF is currently using the older scratch baseline. To activate the newer pretrained v2.1 model, "
                "place its matching best_checkpoint.pt at weights/cmaf_resnet18_cmaf_best_checkpoint.pt "
                "or keep it in the adjacent cmaf-my-model-worktree checkpoint directory.",
                icon="⚠️",
            )
    st.divider()
    st.caption("All four models use the same 640 × 480 RGB-T scene. The thermal preview palette changes display only, never model input values.")

# -----------------------------------------------------------------------------
# Input selection: native 4-channel image, paired sensor images, or local dataset sample
# -----------------------------------------------------------------------------
mode = st.radio(
    "Choose input source",
    ["MFNet 4-channel PNG", "Separate RGB + thermal", "Local MFNet sample"],
    horizontal=True,
    help="The local sample option reads from the neighbouring MFNet_dataset folder when it exists.",
)

rgb_image: Image.Image | None = None
thermal_image: Image.Image | None = None
input_error: str | None = None
input_signature_parts: list[Any] = [mode]
ground_truth_mask: np.ndarray | None = None
input_raw: np.ndarray | None = None
sample_name: str | None = None
source_description = mode

if mode == "MFNet 4-channel PNG":
    four_upload = st.file_uploader(
        "Upload a native 4-channel RGB + thermal PNG",
        type=["png", "tif", "tiff"],
        key="rgbt_four_channel_upload",
        help="Use a file from MFNet_dataset/images, not labels/ or visual/.",
    )
    input_signature_parts.append(_upload_signature(four_upload))
    if four_upload is not None:
        try:
            opened = _open_uploaded_image(four_upload)
            input_raw = np.asarray(opened).copy()
            if input_raw.ndim != 3 or input_raw.shape[2] != 4:
                raise ValueError(f"Expected a 4-channel array [H,W,4], got {input_raw.shape}.")
            if input_raw.dtype != np.uint8:
                if np.issubdtype(input_raw.dtype, np.integer) and input_raw.max(initial=0) <= 255:
                    input_raw = input_raw.astype(np.uint8)
                else:
                    raise ValueError(f"Expected 8-bit channel data, got {input_raw.dtype}.")
            rgb_image = Image.fromarray(np.ascontiguousarray(input_raw[..., :3]), mode="RGB")
            thermal_image = Image.fromarray(np.ascontiguousarray(input_raw[..., 3]), mode="L")
            sample_name = Path(four_upload.name).stem
            source_description = f"Native four-channel input · {four_upload.name}"
            values = np.unique(input_raw[..., 3])
            if len(values) <= 1:
                input_error = "The fourth channel is constant and does not appear to contain thermal intensities."
        except Exception as exc:
            input_error = f"Could not read the file as RGB+thermal: {type(exc).__name__}: {exc}"

elif mode == "Separate RGB + thermal":
    col_rgb, col_thermal = st.columns(2, gap="large")
    with col_rgb:
        rgb_upload = st.file_uploader("Visible-light RGB image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="paired_rgb_upload")
    with col_thermal:
        thermal_upload = st.file_uploader("Aligned raw thermal-intensity image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="paired_thermal_upload")
    input_signature_parts.extend([_upload_signature(rgb_upload), _upload_signature(thermal_upload)])
    if rgb_upload is not None:
        try:
            rgb_image = _open_uploaded_image(rgb_upload).convert("RGB")
        except Exception as exc:
            input_error = f"Could not read RGB image: {type(exc).__name__}: {exc}"
    if thermal_upload is not None:
        try:
            thermal_image = _open_uploaded_image(thermal_upload).convert("L")
        except Exception as exc:
            input_error = f"Could not read thermal image: {type(exc).__name__}: {exc}"
    if thermal_image is not None and len(np.unique(np.asarray(thermal_image))) <= 16:
        st.warning("The thermal image has very few unique grayscale values. Check that it is raw thermal data, not a label mask or colorized preview.")

else:
    dataset_root = _find_dataset_root()
    if dataset_root is None:
        st.warning(
            "Could not find MFNet_dataset next to this app. Expected a sibling folder with images/ and labels/. "
            "Select one of the upload modes above to continue."
        )
    else:
        image_paths = sorted((dataset_root / "images").glob("*.png"))
        names = [path.stem for path in image_paths]
        default_idx = names.index("00001D") if "00001D" in names else 0
        sample_name = st.selectbox("Choose an MFNet sample", names, index=default_idx)
        sample_path = dataset_root / "images" / f"{sample_name}.png"
        input_signature_parts.append((str(sample_path), sample_path.stat().st_size, sample_path.stat().st_mtime_ns))
        try:
            rgb_image, thermal_image, input_raw = _load_dataset_rgba(sample_path)
            source_description = f"Local MFNet sample · {sample_name}"
            label_path = dataset_root / "labels" / f"{sample_name}.png"
            if label_path.is_file():
                with Image.open(label_path) as label_image:
                    ground_truth_mask = _read_mask_image(label_image, str(label_path))
        except Exception as exc:
            input_error = f"Could not load local sample: {type(exc).__name__}: {exc}"

if input_error:
    st.error(input_error)

# Optional ground-truth mask for user uploads. Local dataset mode automatically pairs labels/<sample>.png.
if mode != "Local MFNet sample":
    with st.expander("Optional: attach ground-truth labels to calculate scene mIoU", expanded=False):
        label_upload = st.file_uploader(
            "Ground-truth mask PNG (class IDs 0–8; 255 may be ignored)",
            type=["png", "tif", "tiff", "jpg", "jpeg"],
            key="ground_truth_upload",
            help="Use labels/<sample>.png. Do not upload the visual/ colourized preview.",
        )
        if label_upload is not None:
            try:
                label_image = _open_uploaded_image(label_upload)
                ground_truth_mask = _read_mask_image(label_image, label_upload.name)
                st.success(f"Ground-truth mask loaded: {label_image.width} × {label_image.height}.")
            except Exception as exc:
                ground_truth_mask = None
                st.error(f"Could not use this as a ground-truth label mask: {type(exc).__name__}: {exc}")

# Show source pair and basic sensor diagnostics.
if rgb_image is not None and thermal_image is not None and input_error is None:
    if rgb_image.size != thermal_image.size:
        st.warning(f"RGB size {rgb_image.size} differs from thermal size {thermal_image.size}. Both will be resized to 640 × 480; confirm scene alignment.")
    left, right = st.columns(2, gap="large")
    with left:
        st.image(rgb_image, caption=f"RGB frame · {rgb_image.width} × {rgb_image.height}", use_container_width=True)
    with right:
        st.image(_false_color_thermal(thermal_image, thermal_palette), caption=f"Thermal intensity · {thermal_image.width} × {thermal_image.height} · {thermal_palette}", use_container_width=True)
    if ground_truth_mask is not None:
        gt_preview = colorize_mask(_resize_mask(ground_truth_mask))
        st.image(gt_preview, caption="Ground truth · provided label IDs shown using the app palette", width=420)
    if show_diagnostics:
        rgb_array = np.asarray(rgb_image.convert("RGB"), dtype=np.uint8)
        thermal_array = np.asarray(thermal_image.convert("L"), dtype=np.uint8)
        diag1, diag2, diag3, diag4 = st.columns(4)
        diag1.metric("RGB dimensions", f"{rgb_image.width} × {rgb_image.height}")
        diag2.metric("Thermal range", f"{thermal_array.min()}–{thermal_array.max()}")
        diag3.metric("Thermal unique values", f"{len(np.unique(thermal_array)):,}")
        diag4.metric("GT mask attached", "Yes" if ground_truth_mask is not None else "No")
        with st.expander("Thermal intensity histogram", expanded=False):
            st.image(_make_histogram_image(thermal_array), use_container_width=True)
            st.caption("Distribution of raw thermal pixel intensities (0–255). This visualization does not modify model inputs.")

# Keep the main action immediately after the input controls. Long previews/diagnostics
# should never push the only run button out of sight.
input_signature = tuple(input_signature_parts)
if st.session_state.get("rgbt_input_signature") != input_signature:
    st.session_state.pop("rgbt_model_results", None)
    st.session_state["rgbt_input_signature"] = input_signature

can_run = rgb_image is not None and thermal_image is not None and input_error is None
# The sidebar action remains accessible while users scroll through large previews and diagnostics.
with sidebar_action_slot.container():
    sidebar_run_button = st.button(
        "Run comparison",
        type="primary",
        use_container_width=True,
        disabled=not can_run,
        key="run_models_sidebar",
        help="Run MFNet, CSFNet-2, CMAF, and RTFNet18V2 on the current input.",
    )

st.markdown("### Run comparison")
button_col, note_col = st.columns([1, 2], vertical_alignment="center")
with button_col:
    main_run_button = st.button(
        "Run all four models",
        type="primary",
        use_container_width=True,
        disabled=not can_run,
        key="run_models_main",
        help="Run the same RGB–thermal scene through each architecture.",
    )
with note_col:
    if can_run:
        st.caption("Input ready. Each model runs independently; a failure in one panel does not stop the others.")
    else:
        st.caption("Select a valid four-channel PNG, an aligned RGB–thermal pair, or a local MFNet sample to enable inference.")
run_button = bool(sidebar_run_button or main_run_button)

if run_button:
    rgb_resized = rgb_image.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    thermal_resized = thermal_image.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    results: list[dict[str, Any]] = []
    progress = st.progress(0, text="Preparing the four architectures…")
    run_started = time.perf_counter()

    for idx, spec in enumerate(MODEL_SPECS):
        progress.progress(idx / len(MODEL_SPECS), text=f"Running {spec['name']} · {idx + 1}/{len(MODEL_SPECS)}")
        started = time.perf_counter()
        try:
            model, checkpoint_info = load_model(spec["id"], str(ROOT), device)
            rgb_tensor, thermal_tensor = prepare_rgb_thermal(spec["id"], rgb_resized, thermal_resized, checkpoint_info, device)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            with torch.inference_mode():
                if spec["id"] in ("csfnet", "rtfnet18v2"):
                    output = model(rgb_tensor, thermal_tensor)
                else:
                    output = model(torch.cat((rgb_tensor, thermal_tensor), dim=1))
            logits = _extract_logits(output)
            if logits.shape[-2:] != (TARGET_SIZE[1], TARGET_SIZE[0]):
                logits = F.interpolate(logits.float(), size=(TARGET_SIZE[1], TARGET_SIZE[0]), mode="bilinear", align_corners=False)
            probabilities = torch.softmax(logits.float(), dim=1)
            confidence = float(probabilities.max(dim=1).values.mean().item())
            mask = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
            results.append({
                "id": spec["id"],
                "name": spec["name"],
                "ok": True,
                "seconds": time.perf_counter() - started,
                "confidence": confidence,
                "info": checkpoint_info,
                "mask": mask,
                "error": None,
            })
        except Exception as exc:
            results.append({
                "id": spec["id"],
                "name": spec["name"],
                "ok": False,
                "seconds": time.perf_counter() - started,
                "confidence": 0.0,
                "info": {},
                "mask": None,
                "error": f"{type(exc).__name__}: {exc}",
            })
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    total_seconds = time.perf_counter() - run_started
    progress.progress(1.0, text="Inference finished")
    progress.empty()
    st.session_state["rgbt_model_results"] = results
    st.session_state["rgbt_run_seconds"] = total_seconds
    st.session_state["rgbt_run_source"] = source_description

results = st.session_state.get("rgbt_model_results")
if results:
    st.markdown('<div class="rgbt-divider"></div>', unsafe_allow_html=True)
    st.markdown("## Prediction results")
    successes = [item for item in results if item["ok"]]
    failed = len(results) - len(successes)
    total_seconds = float(st.session_state.get("rgbt_run_seconds", sum(item["seconds"] for item in results)))
    gt_eval = _resize_mask(ground_truth_mask) if ground_truth_mask is not None else None

    summary1, summary2, summary3, summary4 = st.columns(4)
    summary1.metric("Models succeeded", f"{len(successes)} / {len(results)}")
    summary2.metric("Total wall time", f"{total_seconds:.2f} s")
    summary3.metric("Input resolution", "640 × 480")
    summary4.metric("Ground-truth evaluation", "Enabled" if gt_eval is not None else "Not attached")
    st.caption(f"Results for: {st.session_state.get('rgbt_run_source', source_description)}")
    if failed:
        st.warning(f"{failed} model(s) did not produce an output. Each failed architecture shows its own error below.")
    else:
        st.success("All four architectures returned a mask.")

    by_id = {item["id"]: item for item in results}
    for row_start in range(0, len(MODEL_SPECS), 2):
        columns = st.columns(2, gap="large")
        for col, spec in zip(columns, MODEL_SPECS[row_start:row_start + 2]):
            item = by_id.get(spec["id"])
            with col:
                with st.container(border=True):
                    heading_col, status_col = st.columns([4, 1])
                    with heading_col:
                        st.markdown(f"### {spec['name']}")
                    with status_col:
                        st.markdown("🟢 **Ready**" if item and item["ok"] else "🔴 **Failed**")
                    if not item or not item.get("ok"):
                        st.error("This model could not produce a prediction.")
                        if item and item.get("error"):
                            st.code(item["error"])
                        st.caption(f"Expected checkpoint: {spec.get('weight_rel', 'see model loader')}")
                        if spec["id"] == "cmaf":
                            st.info("The newer CMAF v2.1 uses its matching checkpoint. Do not point it at the old 34.56% scratch-baseline weights.")
                        continue

                    info = item.get("info", {})
                    st.caption(f"Inference **{item['seconds']:.2f} s** · mean max softmax probability **{item['confidence']:.1%}**")
                    if info.get("cmaf_variant"):
                        st.caption(f"Active CMAF variant: {info['cmaf_variant']}")
                    if info.get("display_metric"):
                        st.markdown(f'<div class="rgbt-muted">{info["display_metric"]}</div>', unsafe_allow_html=True)
                    mask = item["mask"]
                    mask_preview = colorize_mask(mask)
                    overlay_preview = _foreground_overlay(rgb_image.resize(TARGET_SIZE, Image.Resampling.BILINEAR), mask, alpha=overlay_alpha)
                    image_mask, image_overlay = st.columns(2, gap="small")
                    with image_mask:
                        st.image(mask_preview, caption="Predicted semantic mask", use_container_width=True)
                    with image_overlay:
                        st.image(overlay_preview, caption=f"RGB overlay · opacity {overlay_alpha:.2f}", use_container_width=True)

                    object_coverage = float(np.mean(mask > 0) * 100.0)
                    metric_a, metric_b = st.columns(2)
                    metric_a.metric("Foreground coverage", f"{object_coverage:.2f}%")
                    metric_b.metric("Unique predicted classes", f"{len(np.unique(mask))} / {len(CLASS_NAMES)}")
                    if gt_eval is not None:
                        metrics = _segmentation_metrics(mask, gt_eval)
                        if metrics["miou"] is not None:
                            g1, g2, g3 = st.columns(3)
                            g1.metric("Scene mIoU", f"{metrics['miou']:.2%}")
                            g2.metric("Foreground mIoU", f"{metrics['foreground_miou']:.2%}" if metrics["foreground_miou"] is not None else "N/A")
                            g3.metric("Pixel accuracy", f"{metrics['pixel_accuracy']:.2%}" if metrics["pixel_accuracy"] is not None else "N/A")
                        else:
                            st.warning("No valid target pixels were found; scene metrics were not computed.")

                    if show_pixel_tables:
                        with st.expander(f"{spec['name']} · class pixel counts", expanded=False):
                            st.dataframe(_class_counts(mask), hide_index=True, use_container_width=True)
                            if gt_eval is not None:
                                metrics = _segmentation_metrics(mask, gt_eval)
                                if metrics["per_class_iou"]:
                                    iou_rows = []
                                    for class_name, value in zip(CLASS_NAMES, metrics["per_class_iou"]):
                                        iou_rows.append({"Class": class_name, "IoU": "N/A" if value is None else f"{value:.2%}"})
                                    st.markdown("**Per-class IoU against the attached ground truth**")
                                    st.dataframe(iou_rows, hide_index=True, use_container_width=True)

                    down1, down2 = st.columns(2)
                    with down1:
                        st.download_button(
                            "Download class-ID mask",
                            data=_mask_id_png(mask),
                            file_name=f"{spec['id']}_class_ids.png",
                            mime="image/png",
                            key=f"download_mask_{spec['id']}",
                            use_container_width=True,
                            help="Grayscale PNG with original class IDs 0–8, not a colorized image.",
                        )
                    with down2:
                        st.download_button(
                            "Download overlay",
                            data=_png_bytes(overlay_preview),
                            file_name=f"{spec['id']}_overlay.png",
                            mime="image/png",
                            key=f"download_overlay_{spec['id']}",
                            use_container_width=True,
                        )

    st.markdown("### Cross-model summary")
    comparison_rows = []
    for item in results:
        row = {
            "Model": item["name"],
            "Status": "Success" if item["ok"] else "Failed",
            "Inference (s)": round(item["seconds"], 3),
            "Mean max probability": round(item.get("confidence", 0.0), 4) if item["ok"] else None,
            "Checkpoint metric": item.get("info", {}).get("display_metric", "") if item["ok"] else "",
        }
        if item["ok"] and gt_eval is not None:
            metrics = _segmentation_metrics(item["mask"], gt_eval)
            row["Scene mIoU"] = None if metrics["miou"] is None else round(metrics["miou"], 4)
            row["Foreground mIoU"] = None if metrics["foreground_miou"] is None else round(metrics["foreground_miou"], 4)
        comparison_rows.append(row)
    st.dataframe(comparison_rows, hide_index=True, use_container_width=True)

    if successes and rgb_image is not None:
        st.download_button(
            "⬇  Download all successful results (ZIP)",
            data=_zip_results(results, rgb_image.resize(TARGET_SIZE, Image.Resampling.BILINEAR), overlay_alpha),
            file_name="rgbt_four_model_predictions.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
            help="Includes class-ID masks, color previews, overlays, and a CSV summary.",
        )

    with st.expander("Shared class legend", expanded=False):
        legend_columns = st.columns(3)
        for idx, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
            with legend_columns[idx % 3]:
                hex_color = "#%02X%02X%02X" % tuple(int(channel) for channel in color)
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:9px;padding:.35rem 0;">'
                    f'<span style="display:inline-block;width:15px;height:15px;background:{hex_color};border:1px solid rgba(255,255,255,.3);border-radius:4px;"></span>'
                    f'<span>{name}</span></div>',
                    unsafe_allow_html=True,
                )

with st.expander("How to read the results", expanded=False):
    st.markdown(
        "- **Predicted semantic mask:** each pixel is assigned one of nine classes. The class-ID PNG download preserves the numeric labels.\n"
        "- **RGB overlay:** the mask is blended with the original visible-light image; changing opacity only changes the display.\n"
        "- **Mean max probability:** average maximum softmax probability over pixels. This is a confidence summary, not an accuracy score.\n"
        "- **Saved validation mIoU:** metadata recorded when the checkpoint was evaluated. It is not calculated on the uploaded scene.\n"
        "- **Scene mIoU:** computed only when a matching ground-truth mask is provided; this depends on the supplied label map and alignment."
    )

st.markdown('<div class="rgbt-divider"></div><div class="rgbt-muted">RGB-T Lab · Four-model comparison · Same input pair, independent model adapters</div>', unsafe_allow_html=True)
