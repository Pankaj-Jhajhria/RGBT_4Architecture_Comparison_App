from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

MODEL_SPECS = [
    {
        "id": "mfnet",
        "name": "MFNet",
        "short": "Dual-stream MiniInception CNN",
        "source_rel": "architectures/mfnet/model.py",
        "weight_rel": "weights/mfnet_best_checkpoint.pt",
    },
    {
        "id": "csfnet",
        "name": "CSFNet-2",
        "short": "STDC2 + cosine-similarity attention fusion",
        "source_rel": "architectures/csfnet/models/csfnet.py",
        "weight_rel": "weights/csfnet2_best_model.pth",
    },
    {
        "id": "cmaf",
        "name": "ResNet-18 + CMAF",
        "short": "Pretrained dual-stream ResNet-18 + 5-scale cross-modal attention",
        "source_rel": "architectures/cmaf/my_model_v2.py",
        "weight_rel": "weights/cmaf_resnet18_cmaf_best_checkpoint.pt",
        "legacy_source_rel": "architectures/cmaf/my_model.py",
        "legacy_weight_rel": "weights/cmaf_best_checkpoint.pt",
    },
    {
        "id": "rtfnet18v2",
        "name": "RTFNet18V2",
        "short": "RTFNet-inspired dual ResNet-18 + additive fusion",
        "source_rel": "architectures/rtfnet18v2/model19_v2.py",
        "weight_rel": "weights/rtfnet18v2_best_fp16.pt",
    },
]


def _is_real_checkpoint(path: Path) -> bool:
    """Ignore missing files and Git LFS pointer text files without deserializing weights."""
    try:
        if not path.is_file() or path.stat().st_size < 1024:
            return False
        with path.open("rb") as file:
            head = file.read(160)
        return not head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def find_cmaf_v2_checkpoint(root: Path) -> Path | None:
    """Find the new ResNet-18+CMAF checkpoint; app-local weight wins over the worktree."""
    candidates = [
        root / "weights/cmaf_resnet18_cmaf_best_checkpoint.pt",
        # Existing worktree path used by an earlier setup.
        root.parent / "cmaf-my-model-worktree/checkpoints/resnet18_crossmodal_attention_v2_1_final/best_checkpoint.pt",
        root.parent / "cmaf-my-model-worktree/checkpoints/resnet18_crossmodal_attention_v2_1/best_checkpoint.pt",
        # Dedicated upstream clone containing the verified epoch-22 v2.1 checkpoint.
        root.parent / "CMAF_ResNet18_Upstream/checkpoints/resnet18_crossmodal_attention_v2_1_final/best_checkpoint.pt",
        root.parent / "CMAF_ResNet18_Upstream/checkpoints/resnet18_crossmodal_attention_v2_1/best_checkpoint.pt",
        root / "architectures/cmaf/checkpoints/resnet18_crossmodal_attention_v2_1_final/best_checkpoint.pt",
        root / "architectures/cmaf/checkpoints/resnet18_crossmodal_attention_v2_1/best_checkpoint.pt",
    ]
    # Catch a renamed final run directory inside either known local clone.
    for checkpoints_root in (
        root.parent / "cmaf-my-model-worktree/checkpoints",
        root.parent / "CMAF_ResNet18_Upstream/checkpoints",
    ):
        if checkpoints_root.is_dir():
            candidates.extend(sorted(checkpoints_root.glob("resnet18_crossmodal_attention_v2_1*/best_checkpoint.pt")))
    for candidate in candidates:
        if _is_real_checkpoint(candidate):
            return candidate.resolve()
    return None


def get_model_asset_status(model_id: str, root: str | Path) -> dict[str, Any]:
    """Inspect checkpoint availability for the UI without loading model tensors."""
    root = Path(root)
    if model_id == "cmaf":
        v2_source = root / "architectures/cmaf/my_model_v2.py"
        v2_checkpoint = find_cmaf_v2_checkpoint(root)
        if v2_source.is_file() and v2_checkpoint is not None:
            return {
                "available": True,
                "variant": "CMAF ResNet-18 v2.1",
                "checkpoint": str(v2_checkpoint),
                "is_legacy": False,
            }
        legacy_source = root / "architectures/cmaf/my_model.py"
        legacy_checkpoint = root / "weights/cmaf_best_checkpoint.pt"
        if legacy_source.is_file() and _is_real_checkpoint(legacy_checkpoint):
            return {
                "available": True,
                "variant": "Legacy scratch CMAF baseline",
                "checkpoint": str(legacy_checkpoint.resolve()),
                "is_legacy": True,
            }
        return {
            "available": False,
            "variant": "CMAF checkpoint missing",
            "checkpoint": str(root / "weights/cmaf_resnet18_cmaf_best_checkpoint.pt"),
            "is_legacy": False,
        }

    spec = next((item for item in MODEL_SPECS if item["id"] == model_id), None)
    if spec is None:
        return {"available": False, "variant": "Unknown model", "checkpoint": "", "is_legacy": False}
    checkpoint = root / spec["weight_rel"]
    source = root / spec["source_rel"]
    available = _is_real_checkpoint(checkpoint) and source.is_file()
    return {
        "available": available,
        "variant": spec["name"],
        "checkpoint": str(checkpoint.resolve()),
        "is_legacy": False,
    }

def _load_python_file(module_name: str, source_path: Path):
    if not source_path.is_file():
        raise FileNotFoundError(f"Model source file not found: {source_path}")
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_checkpoint(path: Path, device: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}\nPut the checkpoint at the path shown in the app's repository layout.")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint dictionary, got {type(checkpoint).__name__}")
    return checkpoint


def _state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model_state_dict", "model", "state_dict", "model_state"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
            return value
    # Permit a pure state_dict checkpoint too.
    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise KeyError(f"Could not find a model state dictionary. Keys: {list(checkpoint)[:30]}")


def _saved_metric(model_id: str, checkpoint: dict[str, Any]) -> str | None:
    epoch = checkpoint.get("best_epoch", checkpoint.get("epoch"))
    epoch_text = f" · checkpoint epoch {epoch}" if epoch is not None else ""
    if model_id == "cmaf" and checkpoint.get("architecture_id") == "resnet18_dual_stream_cross_modal_attention_v2_1":
        metrics = checkpoint.get("best_val_metrics", {}) or {}
        miou = metrics.get("miou", checkpoint.get("best_metric_val"))
        fg = metrics.get("miou_foreground")
        pixel_acc = metrics.get("pixel_acc")
        if miou is not None:
            result = f"Saved validation mIoU: {float(miou):.2%}"
            if fg is not None:
                result += f" · foreground: {float(fg):.2%}"
            if pixel_acc is not None:
                result += f" · pixel accuracy: {float(pixel_acc):.2%}"
            return result + epoch_text
    if model_id == "csfnet":
        metrics = checkpoint.get("metrics", {})
        miou = metrics.get("mIoU")
        fg = metrics.get("mIoU_fg")
        if miou is not None:
            result = f"Saved validation mIoU: {float(miou):.2%}"
            if fg is not None:
                result += f" · foreground: {float(fg):.2%}"
            return result + epoch_text
    if model_id == "rtfnet18v2":
        metrics = checkpoint.get("val_metrics", {})
        miou = metrics.get("miou_all9")
        fg = metrics.get("miou_fg8")
        if miou is not None:
            result = f"Saved validation mIoU (9 classes): {float(miou):.2%}"
            if fg is not None:
                result += f" · foreground: {float(fg):.2%}"
            return result + epoch_text
    score = checkpoint.get("best_metric_val")
    if score is not None:
        return f"Recorded best validation metric: {float(score):.2%}" + epoch_text
    return None


def _config_value(config: Any, names: tuple[str, ...], default: Any) -> Any:
    """Recursively find a saved config field, matching case and punctuation insensitively."""
    wanted = {"".join(ch for ch in name.upper() if ch.isalnum()) for name in names}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                normalized = "".join(ch for ch in str(key).upper() if ch.isalnum())
                if normalized in wanted:
                    return value
            for value in node.values():
                found = walk(value)
                if found is not None:
                    return found
        return None

    result = walk(config)
    return default if result is None else result


def _normalization_metadata(model_id: str, checkpoint: dict[str, Any]) -> dict[str, Any]:
    snapshot = checkpoint.get("config_snapshot", {})
    rgb_mean = _config_value(snapshot, ("RGB_MEAN", "MEAN_RGB"), [0.485, 0.456, 0.406])
    rgb_std = _config_value(snapshot, ("RGB_STD", "STD_RGB"), [0.229, 0.224, 0.225])
    if model_id == "csfnet":
        # These values are documented in the team's attached CSFNet README.
        return {
            "rgb_mean": [0.2215, 0.2586, 0.2299],
            "rgb_std": [0.2280, 0.2318, 0.2367],
            "thermal_mean": [0.3954],
            "thermal_std": [0.1848],
            "normalization_source": "CSFNet README training statistics",
        }
    if model_id == "rtfnet18v2":
        args = checkpoint.get("args", {})
        mean = checkpoint.get("thermal_mean")
        std = checkpoint.get("thermal_std")
        if mean is None:
            mean = 0.5
        if std is None:
            std = 0.25
        return {
            "rgb_mean": [0.485, 0.456, 0.406],
            "rgb_std": [0.229, 0.224, 0.225],
            "thermal_mean": [float(mean)],
            "thermal_std": [max(float(std), 1e-6)],
            "normalization_source": "ImageNet RGB statistics and thermal statistics saved in checkpoint",
            "args": args,
        }
    # MFNet and CMAF use their checkpoint snapshots where available. Thermal aliases
    # cover the common INF/IR/thermal config names used by these dataset loaders.
    thermal_mean = _config_value(
        snapshot,
        ("THERMAL_MEAN", "INFRARED_MEAN", "INFRA_MEAN", "IR_MEAN", "INF_MEAN"),
        [0.5],
    )
    thermal_std = _config_value(
        snapshot,
        ("THERMAL_STD", "INFRARED_STD", "INFRA_STD", "IR_STD", "INF_STD"),
        [0.25],
    )
    # A config may store a single scalar or a list; preprocessing handles both.
    return {
        "rgb_mean": rgb_mean,
        "rgb_std": rgb_std,
        "thermal_mean": thermal_mean,
        "thermal_std": thermal_std,
        "normalization_source": "statistics found in checkpoint configuration, with MFNet defaults if absent",
    }


def _normalize_model_key(key: str, root: Path, device: str):
    if key == "mfnet":
        source = root / "architectures/mfnet/model.py"
        weight_path = root / "weights/mfnet_best_checkpoint.pt"
        checkpoint = _load_checkpoint(weight_path, device)
        module = _load_python_file("app_mfnet_model", source)
        snapshot_model = checkpoint.get("config_snapshot", {}).get("model", {})
        dropout = float(snapshot_model.get("DROPOUT_RATE", 0.3))
        class_count = int(snapshot_model.get("NUM_CLASSES", 9))
        model = module.build_model(num_classes=class_count, dropout_rate=dropout)
        model.load_state_dict(_state_dict(checkpoint), strict=True)
        meta = _normalization_metadata(key, checkpoint)

    elif key == "csfnet":
        source_dir = root / "architectures/csfnet"
        source = source_dir / "models/csfnet.py"
        weight_path = root / "weights/csfnet2_best_model.pth"
        if not source.is_file():
            raise FileNotFoundError(f"CSFNet source is missing: {source}")
        # CSFNet's source uses absolute imports such as `from models.stdc import ...`.
        # Put its own source root first on sys.path so these resolve to the copied source.
        source_dir_text = str(source_dir.resolve())
        if source_dir_text in sys.path:
            sys.path.remove(source_dir_text)
        sys.path.insert(0, source_dir_text)
        from models.csfnet import build_csfnet  # type: ignore
        checkpoint = _load_checkpoint(weight_path, device)
        model = build_csfnet(backbone="STDCNet1446", pretrained=False, num_classes=9, dataset="MFNet")
        model.load_state_dict(_state_dict(checkpoint), strict=True)
        meta = _normalization_metadata(key, checkpoint)

    elif key == "cmaf":
        # Prefer the newer pretrained ResNet-18 + 5-scale CMAF checkpoint. If it is
        # not available, keep the existing scratch baseline working rather than
        # breaking the other model comparisons.
        v2_source = root / "architectures/cmaf/my_model_v2.py"
        v2_weight_path = find_cmaf_v2_checkpoint(root)
        if v2_weight_path is not None:
            if not v2_source.is_file():
                raise FileNotFoundError(
                    f"CMAF v2 checkpoint found at {v2_weight_path}, but its source is missing: {v2_source}"
                )
            checkpoint = _load_checkpoint(v2_weight_path, "cpu")
            module = _load_python_file("app_cmaf_resnet18_v2_model", v2_source)
            expected_arch = getattr(module, "ARCHITECTURE_ID", None)
            actual_arch = checkpoint.get("architecture_id") or checkpoint.get("config_snapshot", {}).get("architecture_id")
            if actual_arch and expected_arch and actual_arch != expected_arch:
                raise RuntimeError(
                    f"CMAF v2 checkpoint architecture mismatch: checkpoint={actual_arch!r}, "
                    f"source={expected_arch!r}. Use a checkpoint from the matching CMAF v2.1 branch."
                )
            model = module.build_model(num_classes=9, in_channels=4, pretrained=False)
            model.load_state_dict(_state_dict(checkpoint), strict=True)
            meta = {
                "architecture_variant": "cmaf_resnet18_v2_1",
                "cmaf_variant": "ResNet-18 + CMAF v2.1 (pretrained dual-stream)",
                "normalization_source": "raw RGB-T values scaled to [0,1]; normalization is applied inside my_model_v2.forward",
                "actual_checkpoint_path": str(v2_weight_path),
            }
        else:
            legacy_source = root / "architectures/cmaf/my_model.py"
            legacy_weight_path = root / "weights/cmaf_best_checkpoint.pt"
            if not legacy_source.is_file():
                raise FileNotFoundError(f"Legacy CMAF source is missing: {legacy_source}")
            checkpoint = _load_checkpoint(legacy_weight_path, device)
            module = _load_python_file("app_cmaf_legacy_my_model", legacy_source)
            snapshot_model = checkpoint.get("config_snapshot", {}).get("model", {})
            class_count = int(snapshot_model.get("NUM_CLASSES", 9))
            in_channels = int(snapshot_model.get("IN_CHANNELS", 4))
            base_channels = int(snapshot_model.get("BASE_CHANNELS", 16))
            model = module.build_model(num_classes=class_count, in_channels=in_channels, base_channels=base_channels)
            model.load_state_dict(_state_dict(checkpoint), strict=True)
            meta = _normalization_metadata(key, checkpoint)
            meta.update({
                "architecture_variant": "cmaf_legacy_baseline",
                "cmaf_variant": "Legacy scratch CMAF baseline",
                "actual_checkpoint_path": str(legacy_weight_path.resolve()),
            })

    elif key == "rtfnet18v2":
        source = root / "architectures/rtfnet18v2/model19_v2.py"
        weight_path = root / "weights/rtfnet18v2_best_fp16.pt"
        checkpoint = _load_checkpoint(weight_path, device)
        module = _load_python_file("app_rtfnet18v2_model", source)
        args = checkpoint.get("args", {})
        model = module.build_model(
            num_classes=9,
            pretrained=False,  # Never download ImageNet weights at inference time.
            head=args.get("head", "conv1x1"),
            thermal_init=args.get("thermal_init", "mean"),
            freeze_encoder_bn=(args.get("bn_mode", "train") == "freeze-encoder"),
            device=device,
        )
        model.load_state_dict(_state_dict(checkpoint), strict=True)
        meta = _normalization_metadata(key, checkpoint)
    else:
        raise KeyError(f"Unknown model key: {key}")

    model.to(device).eval()
    meta["display_metric"] = _saved_metric(key, checkpoint)
    meta["epoch"] = checkpoint.get("best_epoch", checkpoint.get("epoch"))
    meta["normalization_source"] = meta.get("normalization_source", "not specified")
    return model, meta


# Cache each model separately so a single broken checkpoint does not prevent the others loading.
try:
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def load_model(model_id: str, root_string: str, device: str):
        return _normalize_model_key(model_id, Path(root_string), device)
except ImportError:  # pragma: no cover - allows syntax/static inspection without Streamlit installed
    def load_model(model_id: str, root_string: str, device: str):
        return _normalize_model_key(model_id, Path(root_string), device)
