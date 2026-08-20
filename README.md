# Extending Dynamic World Surface Water Mapping to Sentinel-1 with AlphaEarth Embeddings

Code, model, sample metadata, and reproducibility instructions for the paper
*Extending Dynamic World Surface Water Mapping to Sentinel-1 with AlphaEarth Embeddings*.

The model predicts a georeferenced per-pixel water probability from a near-real-time
Sentinel-1 observation and an annual AlphaEarth Foundations (AEF) embedding. Dynamic
World supplies weak supervision during training only.

## What is released

- The complete S1-only and S1+AEF training implementation (`src/train.py`).
- Tiled, four-flip-TTA inference (`src/infer.py`).
- Dataset validation and sample-index generation utilities (`scripts/`).
- The exact paper split and reconstruction metadata for every usable training tile
  (`metadata/training_samples.csv`). No imagery is redistributed.
- Metadata for the 53 independent GSWD evaluation scenes
  (`metadata/evaluation_samples.csv`).
- Paper-level and training-run results (`results/`).
- One primary trained S1+AEF checkpoint and its normalization statistics on
  [Hugging Face](https://huggingface.co/rohitm9/surfaceWaterGlobal). The same model
  is mirrored on Google Drive.

Only the primary AEF model is published. The width, training-size, and seed ablations
can be retrained from this code; releasing every ablation checkpoint would add storage
without being necessary to use or reproduce the proposed method.

## Model

| Component | Paper configuration |
| --- | --- |
| Dynamic input | Sentinel-1 GRD `VV`, `VH`, incidence angle |
| Annual context | 64-band AEF v1 annual int8 embedding |
| AEF projection | Bias-free 1x1 convolution, 64 to 16 channels |
| Segmentation model | ImageNet-pretrained ResNet-34 U-Net, Mish activations |
| Loss | Equal-weight cross-entropy + Dice |
| Optimizer | Ranger (RAdam + Lookahead) |
| Schedule | Frozen 2 epochs, partial 2, full 12, low-rate tail 10 |
| Training crop | 512 x 512, batch size 4, seed 42 |
| Reported map threshold | 0.30 with four-flip TTA |

## Install

Python 3.10 or 3.11 and a CUDA-capable GPU are recommended.

```bash
conda create -n surface-water python=3.11 -y
conda activate surface-water
python -m pip install -r requirements.txt
```

## Download the released model

```bash
python scripts/download_model.py --output-dir models/s1aef_resnet34
```

The downloader obtains only the checkpoint and `band_stats.npz`, then verifies their
SHA-256 hashes. Inference normalization is checkpoint-specific, so the statistics file
is required even when the weights are already available elsewhere.

## Inference

Prepare one or more three-band S1 GeoTIFFs named `s1_YYYY-MM-DD.tif` and one 64-band
AEF GeoTIFF. S1 band order must be `VV`, `VH`, `angle`; AEF bands remain in source order.

```bash
python src/infer.py \
  --model-kind s1aef \
  --scenes-root data/inference/s1 \
  --aef-path data/inference/alphaearth_2025.tif \
  --run-dir models/s1aef_resnet34 \
  --output-root outputs \
  --tile 512 --overlap 64 --batch-size 4
```

Outputs are float32 water-probability GeoTIFFs. Apply threshold 0.30 to reproduce the
paper's primary evaluation protocol. Validate the threshold for operational use in a
new region.

## Recreate the training dataset

The source imagery is too large and remains subject to the source providers' terms.
Instead, `metadata/training_samples.csv` records the SWORD node ID, S1 date, AEF year,
exact raster grid, centroid, and paper split for each tile. This is the most useful
lightweight release: it preserves *what was sampled* rather than only describing the
sampling conceptually.

The reconstruction sequence is:

1. Authenticate Google Earth Engine and obtain the S1/Dynamic World pair for each row.
2. Stack S1 `VV`, `VH`, and incidence angle on the recorded grid.
3. Convert Dynamic World label class 0 to water=1 and all other valid classes to 0;
   retain samples with at least 90% valid Dynamic World coverage.
4. Fetch the specified annual AEF v1 embedding from Source Cooperative and reproject
   all 64 bands to the same grid using nearest-neighbor resampling.
5. Preserve the common filename across `s1/`, `aef/`, and `labels/`.

Reconstruct one indexed sample (Earth Engine authentication required):

```bash
python scripts/reconstruct_sample.py \
  --tile-name S1_20151004_24280100821.tif \
  --ee-project YOUR_EARTH_ENGINE_PROJECT \
  --output-root data
```

The command records the resolved Earth Engine image IDs, AEF source COG URLs, and
output checksums in `data/provenance/`. Run it per row or distribute rows across a
batch system. Source archives can change independently; retain the provenance files.

See [`examples/prepare_data.md`](examples/prepare_data.md) for the complete raster
contract. Verify a reconstruction before training:

```bash
python scripts/validate_dataset.py \
  --s1-dir data/s1 --aef-dir data/aef --label-dir data/labels \
  --expected-index metadata/training_samples.csv
```

The released checkpoint's actual input set contains 4,678 triplets. The seeded paper
split contains 3,742 training and 936 validation tiles. These counts and all validation
tile names are independently recoverable from the released run artifacts.

AEF v1 annual coverage begins in 2017. The index therefore records `aef_year=2017`
for the 98 retained S1 samples acquired in 2015-2016; later samples use their
acquisition year. This behavior is explicit in the metadata and reconstruction code.

## Train the paper model

```bash
python src/train.py \
  --input_dir data/s1 \
  --aef_dir data/aef \
  --label_dir data/labels \
  --artifact_dir runs/paper_s1aef_seed42 \
  --n_s1_bands 3 --n_aef_bands 64 --n_proj_bands 16 \
  --loss ce_dice --batch_size 4 --crop_size 512 \
  --seed 42 --split_seed 42 --split_mode paper_tile
```

`paper_tile` exactly reproduces the published tile-level split. `grouped` is also
available for future experiments and keeps rows with the same `grid_id`/`group_id` in
one partition; it is not the protocol used to train the released model.

For the matched S1-only control, omit `--aef_dir` and use
`--n_aef_bands 0 --n_proj_bands 0`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/train.py` | Model, losses, augmentation, training schedule, validation, ablations |
| `src/infer.py` | Geospatial tiled inference and four-flip TTA |
| `scripts/download_model.py` | Download and checksum the released AEF model bundle |
| `scripts/build_sample_index.py` | Regenerate the public sample index from local tiles |
| `scripts/reconstruct_sample.py` | Rebuild an indexed triplet from public source archives |
| `scripts/validate_dataset.py` | Validate bands, grids, names, and optional sample index |
| `src/evaluate.py`, `scripts/compare_methods.py` | Pooled/per-scene metrics, bootstrap CIs, paired tests |
| `metadata/` | Training and independent-evaluation sampling metadata |
| `results/` | Machine-readable paper and training-run summaries |
| `models/model_registry.json` | Model locations, filenames, and checksums |

## Data sources

- Sentinel-1 GRD and Dynamic World: Google Earth Engine public collections.
- AlphaEarth Foundations v1 annual embeddings: Source Cooperative.
- Training sampling frame: SWORD v16 river nodes.
- Independent labels: the public Global Surface Water Dataset (GSWD) PlanetScope-based
  annotations cited in the paper.
- Comparator: OPERA DSWx-S1 from NASA PO.DAAC.

Source datasets retain their own licenses and terms. This repository does not grant
rights to redistribute them.

## Citation

Use the metadata in [`CITATION.cff`](CITATION.cff). Please update the DOI and final
bibliographic fields after publication.

## License

Original code is released under Apache License 2.0. The checkpoint uses the same
license. Third-party datasets and software retain their original terms.
