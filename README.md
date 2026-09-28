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
- The exact paper split and raster-grid metadata for every usable training tile
  (`metadata/training_samples.csv`). No imagery is redistributed.
- Metadata for the 53 independent GSWD evaluation scenes
  (`metadata/evaluation_samples.csv`).
- Paper-level and training-run results (`results/`).
- Three trained checkpoints with their normalization statistics, validation summaries and
  thresholds on [Hugging Face](https://huggingface.co/rohitm9/surfaceWaterGlobal), tag
  `v2-labelclass`: the primary S1+AEF model, the matched S1-only control, and the
  previous-year AEF model.
- The independent S1S2-Water evaluation against OPERA DSWx-S1
  (`external_validation/s1s2_water/`).

The width, training-size and seed ablations are not published as checkpoints; they can be
retrained from this code. The checkpoints released in August 2026 were trained on a mixture
of two Dynamic World water products; they remain available at tag `v1-mixed-labels` and are
superseded by `v2-labelclass`, which uses the DW `label` class only.

Note on the trainer: `src/train.py` here has advanced since the released checkpoints were
produced (it gained ignore-index handling for partially observed labels). The Hugging Face
release ships the exact trainer that produced the published weights.

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
| Training target | Dynamic World `label` band, class 0 (water) |
| Training tiles | 5,278 triplets: 4,222 training, 1,056 validation |
| Reported map threshold | 0.30 with four-flip TTA |
| Validation-selected threshold | 0.50 with four-flip TTA (of 0.30 and 0.50) |

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
python scripts/download_model.py --variant s1_only --output-dir models/s1_only_control
python scripts/download_model.py --variant fused_previous_year --output-dir models/s1aef_previous_year
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
Instead, `metadata/training_samples.csv` records the SWORD node ID, legacy indexed date, AEF year,
exact raster grid, centroid, and paper split for each tile. This is the most useful
lightweight release: it preserves *what was sampled* rather than only describing the
sampling conceptually.

The legacy indexed date came from the original filename and does not, by itself,
identify the exact Sentinel-1 source scene. Use a generation provenance manifest when
exact acquisition timing matters. If the original raster still exists,
`scripts/recover_s1_raster_provenance.py` can instead verify its source scene by exact
VV/VH/angle pixel matching. A catalog lookup by date alone is an inference and is not
equivalent to recorded or raster-matched provenance.

For the footprint coverage inventory (SWORD sampling frame, SWOT PLD lake overlap,
JRC water persistence, GSHHG coastal proximity, and S1/Dynamic World timing), see
[`docs/sample_inventory_audit.md`](docs/sample_inventory_audit.md) and the audit
commands in [`examples/prepare_data.md`](examples/prepare_data.md).

The reconstruction sequence is:

1. Authenticate Google Earth Engine and obtain an S1/Dynamic World pair on the recorded
   grid. Exact reconstruction additionally requires the original scene provenance.
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

The command performs date-based catalog inference, then records the resolved Earth
Engine image IDs, AEF source COG URLs, and output checksums in `data/provenance/`.
It can fail when the legacy date has no same-day S1 acquisition, and a result should
not be described as an exact historical reconstruction without source-scene records.
Source archives can change independently; retain all new provenance files.

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

### Retrain with the open-water supplement

`outputs/audit/supplement_samples.csv` defines 600 additional lake, reservoir,
coastal/estuarine, and seasonal-water grids with exact Sentinel-1 and Dynamic World
scene IDs. The supplement is not materialized until the export workflow is run.

On NERSC, the complete paper-refresh dependency chain is:

```bash
scripts/submit_paper_retrain.sh
```

The workflow exports and validates the 600 new S1/Dynamic World/AEF triplets, builds
an immutable 5,278-tile manifest, preserves the existing 4,678-tile split, and adds
480/120 new train/validation tiles. It then runs the matched three-seed AEF-width and
training-size experiments, reruns the 53-scene independent evaluation for S1-only,
acquisition-year AEF, and previous-year AEF, and writes manuscript-facing result
tables under `results/paper_retrain_openwater_v1/`.

Training consumes `triplets.csv` with `--triplet_manifest` and `fixed_split.csv` with
`--fixed_split_manifest`; this prevents a retrain from silently reshuffling the
published validation population.

### Label-class rerun

The runs behind `results/paper_retrain_openwater_v1/` resolved about 65% of their tiles to a
binarized Dynamic World water probability rather than the `label` band. All 33 runs were repeated
with the `label` class only, keeping the same chips, split, seeds and hyperparameters, and those
runs produce the released checkpoints and every number in `results/paper_labelclass_v1/58321212/`.
The analyses accept either run set:

```bash
python scripts/complete_paper_analyses.py sweep --run-set labelclass_v1 --out results/.../threshold_sweep.json
python scripts/complete_paper_analyses.py paired --run-set mixed_v1 --out /tmp/paired_old.json
python scripts/make_figure3_iou_distribution.py --run-set labelclass_v1 --out-dir results/.../figures
```

## Results

Independent 53-scene PlanetScope reference set, threshold 0.30
(`results/paper_labelclass_v1/58321212/paper_metrics.csv`):

| Method | Precision | Recall | Pooled water IoU | Per-scene mean [95% CI] |
| --- | ---: | ---: | ---: | --- |
| S1 + AEF (acquisition year) | 0.892 | 0.948 | 0.851 | 0.756 [0.689, 0.816] |
| S1 + AEF (previous year) | 0.882 | 0.946 | 0.840 | 0.741 [0.672, 0.803] |
| S1 only | 0.933 | 0.582 | 0.558 | 0.443 [0.355, 0.533] |
| OPERA DSWx-S1 | 0.858 | 0.852 | 0.746 | 0.591 [0.508, 0.670] |

`results/paper_labelclass_v1/58321212/completion_results.md` documents every number, including the
threshold sweep, the train/evaluation proximity screen, the matched-valid-mask comparison, the paired
tests against OPERA and the WorldCover error analysis. `results/paper_retrain_openwater_v1/57725872/`
holds the superseded mixed-label results.

Independent S1S2-Water benchmark, 15 test scenes, 30 m grid, mean over three seeds
(`external_validation/s1s2_water/`):

| Method | Pooled water IoU, threshold 0.50 | Threshold 0.30 |
| --- | ---: | ---: |
| S1 + AEF | 0.953 | 0.941 |
| OPERA DSWx-S1 (v1.2 with the upstream boundary fix) | 0.869 | 0.869 |
| S1 only | 0.822 | 0.794 |

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
| `scripts/complete_paper_analyses.py` | Threshold sweep, proximity screen, matched mask, paired tests vs OPERA |
| `scripts/make_figure3_iou_distribution.py` | Per-scene IoU distribution figure |
| `external_validation/s1s2_water/` | Independent S1S2-Water evaluation against OPERA DSWx-S1 |
| `jobs/` | Slurm job scripts for the training, inference and evaluation chain |
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
