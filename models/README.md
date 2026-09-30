# Released model bundles

The checkpoints are on Hugging Face (`rohitm9/surfaceWaterGlobal`, tag `v2-labelclass`).
From the repository root:

```bash
python scripts/download_model.py --output-dir models/s1aef_resnet34
python scripts/download_model.py --variant s1_only --output-dir models/s1_only_control
python scripts/download_model.py --variant fused_previous_year --output-dir models/s1aef_previous_year
```

The resulting layout is:

```text
models/s1aef_resnet34/                       # --run-dir for the primary S1+AEF model
  band_stats.npz
  models/s1aef_bottleneck_resnet34_best.pth
models/s1_only_control/
  variants/s1_only_k0_seed42/                # --run-dir for the S1-only control
    band_stats.npz
    models/s1dw_resnet34_best.pth
models/s1aef_previous_year/
  variants/s1_aef_previous_year_k16_seed42/  # --run-dir for the previous-year model
    band_stats.npz
    models/s1aef_bottleneck_resnet34_best.pth
```

Both files are required for each model. `band_stats.npz` holds the per-band training means
and standard deviations used by inference (67 bands for the S1+AEF models, 3 for the S1-only
control). The downloader verifies the SHA-256 hashes recorded in `model_registry.json`, which
also records the training run of each checkpoint.

The Google Drive folder listed under `superseded` in `model_registry.json` holds only the
superseded `v1-mixed-labels` checkpoint. Do not commit checkpoints to Git.
