# surfaceWaterMappingGlobal

Global surface-water mapping with co-registered Sentinel-1 (S1) SAR and AlphaEarth Foundations (AEF) annual embeddings. The supplied model predicts a per-pixel Dynamic World (DW) binary water probability.

The primary model is a FastAI ResNet34 U-Net. It receives three S1 bands plus 64 AEF bands. A learned 1x1 bottleneck projects the 64 AEF bands to 16 channels before fusion. The training script can also run the S1-only baseline.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/train.py` | Train or benchmark the S1-only or S1+AEF ResNet34 U-Net. |
| `src/infer.py` | Tiled, probability-only inference for S1-only or S1+AEF models. |
| `examples/` | Sample manifest and the data-preparation recipe. |
| `models/` | Model bundle location. Checkpoints are deliberately ignored by Git. |
| `models/model_registry.json` | One place to add the Google Drive URL once the model is uploaded. |

## Setup

Python 3.10+ and a CUDA-capable GPU are recommended. The code also runs on CPU, but inference is substantially slower.

```bash
conda create -n surface-water python=3.10 -y
conda activate surface-water
pip install -r requirements.txt
```

## Data contract

Training pairs must use identical file names in the input directories:

```text
data/
  s1/       # 3-band, co-registered GeoTIFFs
  aef/      # matching 64-band AEF GeoTIFFs
  labels/   # matching single-band binary DW water masks (0=other, 1=water)
```

For inference, `--scenes-root` is searched recursively for `s1_YYYY-MM-DD.tif` files. Each must have three S1 bands. `--aef-path` is one 64-band AEF GeoTIFF; it is reprojected to each S1 scene grid when needed.

## Prepare training data

The model does not download source imagery itself. Prepare each training tile as a co-registered S1/AEF/DW triplet, using one fixed projected 10 m grid per tile. The critical rule is that all three rasters have the same CRS, transform, width, height, pixel alignment, and filename.

1. Define an area of interest and a projected output grid (typically the local UTM CRS at 10 m resolution).
2. Export or preprocess Sentinel-1 GRD to that grid. Stack exactly three float bands in this order: `VV`, `VH`, `angle`.
3. Export the matching annual AlphaEarth Foundation embedding, retaining all 64 raw embedding bands. Reproject it to the same grid; do not use a PCA-reduced AEF product with the supplied 64-band model.
4. Export Dynamic World to the same grid and derive a single-band binary target: DW class `0` (water) becomes `1`; every other valid class becomes `0`.
5. Name all three files identically, for example `tile_0001.tif`, and store them in `data/s1/`, `data/aef/`, and `data/labels/` respectively.
6. Validate a small set of triplets visually before training. Misaligned rasters are the fastest way to teach a segmentation model surrealist geography.

Use the directory convention above for filename-based matching, or use an explicit manifest as shown in [`examples/training_manifest.csv`](examples/training_manifest.csv). See [`examples/prepare_data.md`](examples/prepare_data.md) for the raster checklist and examples.

## Train

Train the S1+AEF model:

```bash
python src/train.py \
  --input_dir data/s1 \
  --aef_dir data/aef \
  --label_dir data/labels \
  --artifact_dir runs/s1aef_resnet34 \
  --n_s1_bands 3 --n_aef_bands 64 --n_proj_bands 16 \
  --loss ce_dice --batch_size 4 --crop_size 512
```

For the S1-only baseline, set `--n_aef_bands 0 --n_proj_bands 0` and omit `--aef_dir`. Each run saves its checkpoint to `runs/<run>/models/`, along with `band_stats.npz`, the selected threshold, metric tables, and an evaluation summary. Keep these files together: inference needs both the checkpoint and `band_stats.npz`.

## Use the trained model

The trained-model link is intentionally a placeholder until the checkpoint is uploaded. Update the `google_drive_url` field in [`models/model_registry.json`](models/model_registry.json) with a direct-download Google Drive URL and download the **whole model bundle** into `models/s1aef_resnet34/`:

```text
models/s1aef_resnet34/
  models/s1aef_bottleneck_resnet34_best.pth
  band_stats.npz
  s1aef_bottleneck_resnet34_eval_summary.json
```

The recommended checkpoint to upload is:

```text
/pscratch/sd/r/rohit9/S1ML/training_runs/s1aef_bottleneck_resnet34_crop512/
  dw_label_water_pm1d_recovered/bestproj16_ce_dice/train_53123937/
    models/s1aef_bottleneck_resnet34_best.pth
```

Also copy `band_stats.npz` and `s1aef_bottleneck_resnet34_eval_summary.json` from that same `train_53123937` directory. The checkpoint alone cannot reproduce inference normalization.

## Infer

After placing the bundle above in `models/s1aef_resnet34/`, run:

```bash
python src/infer.py \
  --model-kind s1aef \
  --scenes-root data/inference/s1 \
  --aef-path data/inference/alphaearth_2025.tif \
  --run-dir models/s1aef_resnet34 \
  --output-root outputs \
  --tile 512 --overlap 64 --batch-size 4
```

This writes georeferenced water-probability GeoTIFFs to `outputs/s1_aef_tta/probabilities/`. By default, four flip test-time augmentation passes are averaged. Add `--no-tta` for faster inference. Add `--validate-only` to check data, model, and normalization files without running the model.

## Model hosting

Do not commit `.pth` checkpoints to Git. Upload the three-file bundle above to Google Drive, set sharing to allow the intended users to download it, then paste its direct-download link into `models/model_registry.json`. The exact link is then visible in the repository without placing large artifacts in Git history.
