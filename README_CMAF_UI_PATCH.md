# RGBT Comparison App — CMAF v2.1 and UI patch

This patch updates the Streamlit UI and adds an isolated loader for the newer pretrained ResNet-18 + 5-scale CMAF v2.1 model. It intentionally keeps the existing CSFNet-2 and RTFNet18V2 source, checkpoint paths, tensor signatures, and preprocessing functions unchanged. The existing legacy CMAF loader remains available if the v2.1 checkpoint is not found.

## Apply

Extract this ZIP into the root of `RGBT_4Architecture_Comparison_App`, overwriting `app.py`, `inference/model_loaders.py`, and `inference/preprocessing.py`. The ZIP adds `architectures/cmaf/my_model_v2.py`; it does not overwrite the existing `architectures/cmaf/my_model.py` or any weights.

Restart Streamlit:

```powershell
python -m streamlit run app.py
```

## CMAF checkpoint selection

The loader prefers the new checkpoint in this order:

1. `weights/cmaf_resnet18_cmaf_best_checkpoint.pt`
2. `../cmaf-my-model-worktree/checkpoints/resnet18_crossmodal_attention_v2_1_final/best_checkpoint.pt`
3. `../cmaf-my-model-worktree/checkpoints/resnet18_crossmodal_attention_v2_1/best_checkpoint.pt`
4. A matching `resnet18_crossmodal_attention_v2_1*/best_checkpoint.pt` directly under the adjacent CMAF worktree's `checkpoints` folder.

If the new checkpoint is outside those paths, copy it into the app's `weights` folder using:

```powershell
Copy-Item "..\cmaf-my-model-worktree\checkpoints\resnet18_crossmodal_attention_v2_1_final\best_checkpoint.pt" ".\weights\cmaf_resnet18_cmaf_best_checkpoint.pt"
```

If that exact source path differs, use the actual checkpoint path in the command. Keep the existing `weights/cmaf_best_checkpoint.pt`; it is the older scratch baseline and is retained only as a fallback. The newer v2.1 architecture must be paired with its own matching checkpoint. The loader validates the `architecture_id` when it is present and loads the state dictionary strictly.

The newer CMAF model normalizes RGB and thermal channels inside `my_model_v2.forward`. Its app input therefore stays scaled to `[0,1]`, matching the uploaded `dataset.py`. The old baseline keeps its former path. MFNet remains raw `[0,1]`; the CSFNet-2 and RTFNet18V2 preprocessing code paths are preserved.

## UI additions

- Dark, responsive comparison workbench and model checkpoint status.
- Native 4-channel upload, separate aligned RGB/thermal upload, and local MFNet sample mode.
- Thermal false-colour preview and input-channel diagnostics.
- Adjustable overlay opacity, model timing and mean maximum probability summary.
- Optional ground-truth mask upload for scene mIoU, foreground mIoU, pixel accuracy and per-class IoU.
- Per-model class pixel counts and downloadable class-ID masks/overlays.
- One ZIP export containing successful masks, overlays and CSV metadata.

Checkpoint validation metrics are shown as saved metadata, not as a score for the current uploaded scene. Scene mIoU is only computed when an aligned ground-truth mask is supplied.
