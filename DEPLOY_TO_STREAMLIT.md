# Deploy the RGB-T 4-Model Comparison App to Streamlit Community Cloud

This kit adds deployment configuration. It does not contain the application source, architecture directories, or model weights. Extract these files into the root of your existing `RGBT_4Architecture_Comparison_App` folder and overwrite `requirements.txt` if prompted.

## 1. Prepare the correct checkpoints

From the app root in PowerShell, run:

```powershell
.\prepare_streamlit.ps1
```

The script copies your verified CMAF v2.1 checkpoint from the adjacent `CMAF_ResNet18_Upstream` checkout into:

`weights\cmaf_resnet18_cmaf_best_checkpoint.pt`

It verifies that all four active checkpoints are present and look like real files rather than Git LFS pointer text files. It configures Git LFS but does not commit or push anything.

The deployment needs these files:

- `weights/mfnet_best_checkpoint.pt`
- `weights/csfnet2_best_model.pth`
- `weights/cmaf_resnet18_cmaf_best_checkpoint.pt` (the verified v2.1 checkpoint, not the old 34.56% scratch checkpoint)
- `weights/rtfnet18v2_best_fp16.pt`

Do not include the full MFNet dataset; uploaded images are enough for inference.

## 2. Check your GitHub remote before pushing

```powershell
git remote -v
git branch --show-current
git status --short
```

Confirm this is your **personal app repository**, not the shared/team model repository. Do not switch branches or push to a team repo for deployment.

If the personal GitHub repo is not created yet, create it on GitHub first. If `origin` is missing, add your personal repo as `origin`. If `origin` points to the wrong repository, stop and correct it before pushing.

## 3. Stage source and model weights

The project `.gitignore` may ignore `.pt` and `.pth` files, so force-add only the four selected active checkpoints. Git LFS tracks the binary content; do not commit the models as ordinary Git blobs.

```powershell
git lfs install
git lfs track "weights/*.pt" "weights/*.pth"

git add .gitattributes requirements.txt .streamlit/config.toml .gitignore

git add app.py inference architectures

git add -f weights/mfnet_best_checkpoint.pt

git add -f weights/csfnet2_best_model.pth

git add -f weights/cmaf_resnet18_cmaf_best_checkpoint.pt

git add -f weights/rtfnet18v2_best_fp16.pt

# Review carefully before committing
git lfs ls-files
git status --short
git diff --cached --stat
```

Make sure the staged files do not contain a dataset folder, `.venv`, `__pycache__`, old checkpoints, training logs, or unrelated secrets. If everything is correct:

```powershell
git commit -m "Prepare RGB-T four-model app for Streamlit deployment"
git push -u origin main
```

Use the actual current branch name shown by `git branch --show-current` if it is not `main`; don't push to a team branch.

GitHub LFS has included storage/bandwidth quotas, and the active four weights total about 0.65 GiB. That is within the documented 10 GiB free Git LFS allowance at the time this kit was created, but repeated clones/redeployments consume bandwidth. Streamlit Community Cloud documentation states that Git LFS files work with deployed repositories.

## 4. Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io and sign in with the GitHub account that owns the personal app repository.
2. Click **Create app**.
3. Select **From existing repo** (wording may vary).
4. Repository: `Pankaj-Jhajhria/RGBT_4Architecture_Comparison_App`.
5. Branch: `main` (or the branch you pushed).
6. Main file path: `app.py`.
7. In **Advanced settings**, choose Python **3.12**.
8. Deploy.

No secrets are required when all four model checkpoints are publicly available in this public app repo through Git LFS. Do not put access tokens in Python source or commit `.streamlit/secrets.toml`.

## 5. First-deployment test

After the app finishes building, use the app's **Checkpoint status** panel. It should report the newer `CMAF ResNet-18 v2.1` checkpoint, not `Legacy scratch CMAF baseline`. Upload one native four-channel `images/*.png` sample and run all four models.

If a checkpoint shows as missing or the app reports an LFS pointer, open the Streamlit app's **Manage app → Logs** view and inspect the deployment logs. Do not upload the old CMAF baseline under the v2.1 filename.

## Notes

- Community Cloud is a CPU environment for this project; inference may be slower than the local GPU.
- The checkpoint's recorded validation mIoU is not a score for the currently uploaded image. To calculate per-image or dataset mIoU, a matching ground-truth mask is required.
- The kit pins CPU PyTorch wheels to keep the deployment from installing CUDA-only dependencies unnecessarily. It does not change model architecture or preprocessing code.
