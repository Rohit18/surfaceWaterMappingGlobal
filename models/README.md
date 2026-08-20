# Released model bundle

Run `python scripts/download_model.py --output-dir models/s1aef_resnet34` from the
repository root. The resulting layout is:

```text
models/s1aef_resnet34/
  band_stats.npz
  models/
    s1aef_bottleneck_resnet34_best.pth
```

Both files are required. `band_stats.npz` contains the 67 training-band means and
standard deviations used by inference. The downloader verifies the hashes recorded
in `model_registry.json`.

The Hugging Face and Google Drive copies are mirrors of the same primary AEF model.
Do not commit checkpoints to Git.
