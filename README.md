# RGB–Thermal Four-Architecture Comparison App

Streamlit inference UI for the MFNet, CSFNet-2, ResNet-18 + CMAF, and RTFNet18V2 models. The app accepts one RGB image and its aligned single-channel thermal image, then displays each model's predicted nine-class mask and RGB overlay.

## Expected local layout

This app package is intended to be copied into the existing `RGBT_4Architecture_Comparison_App` project, preserving the `architectures/` and `weights/` directories already created there.

```text
app.py
requirements.txt
.gitignore
inference/
  model_loaders.py
  preprocessing.py
  visualization.py
architectures/
  mfnet/model.py
  csfnet/models/csfnet.py, stdc.py, fusion.py, ...
  cmaf/my_model.py
  rtfnet18v2/model19_v2.py
weights/
  mfnet_best_checkpoint.pt
  csfnet2_best_model.pth
  cmaf_best_checkpoint.pt
  rtfnet18v2_best_fp16.pt
```

The app package does **not** contain checkpoints. Keep the checkpoint files in your local `weights/` folder; `.gitignore` excludes them from normal Git pushes.

## Run locally

From the project root, activate your existing PyTorch environment and install Streamlit if needed:

```powershell
python -m pip install streamlit
python -m streamlit run app.py
```

The code imports PyTorch, torchvision, NumPy and Pillow. Install a mutually compatible `torch` and `torchvision` pair for your system using the official PyTorch instructions if your environment does not already have them.

## Inputs

- RGB: colour visible-light image.
- Thermal: raw single-channel thermal intensity image, not a colour-mapped heatmap when possible.
- Both should be spatially aligned. The app resizes each input to 640 × 480 pixels.

## Important interpretation notes

- Each model has a separate loader and normalization metadata. MFNet and CMAF take concatenated four-channel tensors; CSFNet-2 and RTFNet18V2 take RGB and thermal tensors separately.
- Displayed saved validation metrics come from checkpoint metadata. They are not recalculated on uploaded input images and are not necessarily directly comparable across separate runs.
- The app can show predictions without the original dataset. Computing new mIoU requires ground-truth masks.
- Before online deployment, store model weights in suitable model storage and download/cache them at runtime; do not commit the large checkpoint files to the app repository.
