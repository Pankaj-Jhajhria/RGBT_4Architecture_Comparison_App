# CMAF checkpoint path fix (safe, targeted patch)

The upstream checkpoint has been verified by its metadata:
- architecture: `resnet18_dual_stream_cross_modal_attention_v2_1`
- best epoch: 22
- validation mIoU: 0.555502 (55.55%)
- state dictionary: 345 tensors

The app currently searches for the newer checkpoint under `cmaf-my-model-worktree`, but the verified checkpoint is in the sibling folder `CMAF_ResNet18_Upstream`. This patch adds that location to checkpoint discovery. It changes only `inference/model_loaders.py`; it does not modify `app.py`, model source files, preprocessing, weights, or any other model path.

## Install

1. Stop Streamlit (`Ctrl+C`).
2. Extract this ZIP into the root of `RGBT_4Architecture_Comparison_App`, allowing the file `inference/model_loaders.py` to be overwritten.
3. Restart with `python -m streamlit run app.py`.
4. In the sidebar, verify the CMAF status says `CMAF ResNet-18 v2.1`, and its checkpoint path contains `CMAF_ResNet18_Upstream`.

The loader uses the v2.1 architecture and its matching checkpoint when discovered. Its state dictionary is loaded with `strict=True`. If architecture or key compatibility fails, CMAF should show an error rather than silently loading the legacy weights. MFNet, CSFNet-2, and RTFNet18V2 loaders and preprocessing are unchanged.
