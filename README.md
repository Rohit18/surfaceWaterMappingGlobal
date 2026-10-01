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
- The split and raster-grid metadata for the 4,678 tiles of the original training sample
  (`metadata/training_samples.csv`). No imagery is redistributed.
- The centroid, split, sampling frame, S1 date and AEF year of all 5,278 training tiles of
  the released checkpoints, including the 600-tile open-water supplement, as GeoJSON points
  (`metadata/training_sample_centroids.geojson`). The raster grids of the supplement tiles
  are not published.
- Metadata for the 53 independent GSWD evaluation scenes
  (`metadata/evaluation_samples.csv`).
- The 10 m evaluation protocol of the paper: input export, inference wrappers, scoring,
  ablations, figures and the resolution-matched OPERA comparison (`scripts/eval_10m/`,
  `jobs/eval_10m/`).
- Paper-level and training-run results (`results/`). The manuscript's numbers are in
  `results/paper/`.
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
| Optimizer | Ranger (RAdam + Lookahead; fastai `ranger`), decoupled weight decay 0.01, gradient clipping 1.0, fp16 |
| Schedule | Four `fit_one_cycle` stages with peak learning rates: encoder frozen, 2 epochs at 1e-4; partial unfreeze, 2 epochs at slice(5e-7, 1e-5); full fine-tuning, 10 epochs at slice(6e-8, 3e-6); tail, 10 epochs at slice(1.5e-7, 1.5e-6). `--finetune_epochs 12` includes the 2 partial-unfreeze epochs |
| Early stopping and checkpoint | Within each stage, on validation water IoU (patience 4, min_delta 0.001); each stage starts from the previous stage's saved epoch; the checkpoint is the saved epoch of the final stage (see below) |
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

The downloader obtains only the checkpoint and `band_stats.npz` from the Hugging Face tag
`v2-labelclass` (commit `472736653d5450ae3d47b6198f1411ed507628ad`), then verifies their
SHA-256 hashes. The hashes in `scripts/download_model.py` are the same as those in
[`models/model_registry.json`](models/model_registry.json); `tests/test_release_metadata.py`
checks this. Inference normalization is
checkpoint-specific, so the statistics file is required even when the weights are already
available elsewhere. The variants keep their Hugging Face paths, so their `--run-dir` for
`src/infer.py` is the `variants/...` subdirectory of the output directory:

| `--variant` | Model | Training run (job / task) | Training tiles (train / valid) | `--run-dir` after the commands above | Checkpoint SHA-256 |
| --- | --- | --- | --- | --- | --- |
| `fused_current` (default) | S1 + acquisition-year AEF, k=16 | `58321212` / `width_k16_seed42` | 5,278 (4,222 / 1,056) | `models/s1aef_resnet34` | `dd0252e2…688b06ec` |
| `s1_only` | Matched S1-only control, k=0 | `58321212` / `width_k0_seed42` | 5,278 (4,222 / 1,056) | `models/s1_only_control/variants/s1_only_k0_seed42` | `a5eab4f0…a57b6e68` |
| `fused_previous_year` | S1 + previous-year AEF, k=16 | `58321217` / `width_k16_seed42` | 5,140 (4,119 / 1,021) | `models/s1aef_previous_year/variants/s1_aef_previous_year_k16_seed42` | `5bd31240…d6fb2717` |

Hashes are abbreviated; the full checkpoint and `band_stats.npz` hashes are in `models/model_registry.json`.

All three are seed 42 of the label-class runs described in [Label-class rerun](#label-class-rerun),
trained with the configuration in [Model](#model). The same Hugging Face tag also holds, for each
model, `validation_eval_summary.json` and `validation_threshold_sweep.csv`; for the primary model it
also holds `MODEL_INFO.json` (inputs, band order, thresholds, hashes), `aef_bottleneck_report.json`,
and the exact `src/train.py` that produced the weights (SHA-1
`cd991e7424ae8d74c4737bd339f1ca66e563e506`). Download them with `huggingface_hub` or from the
Hugging Face web page. The evaluation tables and numbers on the Hugging Face tag come from the
earlier 3 m-input protocol; the manuscript's numbers are in `results/paper/`.
The `v1-mixed-labels` checkpoint (training run `train_53123937`,
`results/released_model_training.json`) is superseded and is not used for any number in the paper.

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

Outputs are float32 water-probability GeoTIFFs on the grid of the S1 input. The model was
trained on 10 m inputs; provide S1 and AEF on a 10 m grid. The paper reports water at
probability >= 0.30. Validate the threshold for operational use in a new region.

`src/infer.py` shares its tiling, TTA and normalisation code with the inference wrappers
used for the paper, but it was not used for any paper number. To reproduce the paper, use
the manifest-driven wrappers in `scripts/eval_10m/runtime/` as described in
[Reproducing the paper](#reproducing-the-paper).

## Recreate the training dataset

The source imagery is too large and remains subject to the source providers' terms.
Instead, `metadata/training_samples.csv` records the SWORD node ID, legacy indexed date, AEF year,
exact raster grid, centroid, and paper split for each of the 4,678 tiles of the original
sample (see [Training sample centroids](#training-sample-centroids) for all 5,278 tiles). This is the most useful
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

`metadata/training_samples.csv` covers the original 4,678 triplets (3,742 training, 936
validation), which the superseded `v1-mixed-labels` checkpoint was trained on. The released
`v2-labelclass` checkpoints were trained on these 4,678 tiles, with their split unchanged, plus
the 600-tile open-water supplement (480 training, 120 validation), giving 5,278 tiles (4,222 /
1,056).

### Training sample centroids

`metadata/training_sample_centroids.geojson` has one point per tile for all 5,278 tiles
(RFC 7946, WGS 84 longitude/latitude, 6 decimal places). The point is the centre of the
tile's raster grid; tiles are 10.5-11.5 km on a side (supplement tiles 11 km) at 10 m. Properties:

| Property | Content |
| --- | --- |
| `tile_name` | Tile filename, shared by the S1, AEF and label rasters |
| `sample_set` | `sword_river_node` (4,678 original tiles) or `open_water_supplement` (600) |
| `sampling_frame` | `SWORD_v16`, `SWOT_PLD_2.02` (200), `JRC_GSW1.4` (150), `GDW_v1.0` (125) or `GSHHG_2.3.7` (125) |
| `primary_target` | `river_targeted`, `non_river_natural_lake`, `high_water_seasonal_or_ephemeral`, `confirmed_reservoir` or `coastal_or_estuarine` |
| `source_feature_id` | Feature ID in the sampling frame (SWORD node ID for the original tiles) |
| `s1_date` | Sentinel-1 date |
| `s1_date_basis` | `legacy_indexed_date` for the original tiles (the filename date; see the caveat above) or `s1_acquisition_utc` for the supplement, whose scene IDs were recorded |
| `aef_year` | Year of the AEF embedding used by the acquisition-year models |
| `split` | `train` or `valid`, identical in all released models |
| `in_previous_year_model` | Whether the tile is in the 5,140-tile previous-year training set (which uses `aef_year` - 1) |

`scripts/build_training_centroids.py` writes the file from `metadata/training_samples.csv`,
the supplement table and the two training manifests. The 4,678 original tiles have the same
split and centroids as `metadata/training_samples.csv`; `tests/test_release_metadata.py`
checks this and the counts above.

AEF v1 annual coverage begins in 2017. The index therefore records `aef_year=2017`
for the 98 retained S1 samples acquired in 2015-2016; later samples use their
acquisition year. This behavior is explicit in the metadata and reconstruction code.
The previous-year (t-1) training manifest (5,140 tiles) omits the 137 tiles acquired in
2015-2017, which have no previous-year embedding, and one tile whose 2021 embedding
covers less than 90% of it.

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

### Training schedule details

`src/train.py` runs four stages (`build_schedule`), each a separate fastai `fit_one_cycle` call with the
optimizer state reset. A `slice(a, b)` learning rate is spread geometrically over the three parameter groups
(encoder stem and layers 1-2; layers 3-4; decoder, head and AEF projection). `SaveModelCallback` and
`EarlyStoppingCallback` are rebuilt for every stage and both use `min_delta=0.001`, so an epoch is saved only when
its validation water IoU (argmax) beats the best value of the current stage by more than 0.001. After each stage the
saved weights are reloaded. The released checkpoint is the saved epoch of the final (tail) stage; it is not
necessarily the highest validation IoU reached during the run. In the label-class runs, early stopping ended the
fine-tuning stage in 20 of 33 runs and the tail stage in 30 of 33 (14-24 of 24 planned epochs). The paper runs used
`--eval_thresholds 0.30 0.50`; every run selects 0.50 (TTA) on the Dynamic World validation split. Training used
one NVIDIA A100-SXM4-40GB per run (recorded in the job logs), PyTorch 2.6.0+cu124 with cuDNN 9.1.0 and fastai 2.7.19.
`torch.optim.AdamW` appears only in the batch-size probe of `--mode benchmark`, which the paper runs do not use.

This command trains on the 4,678 tiles of `metadata/training_samples.csv`. The released
`v2-labelclass` models were instead trained from the 5,278-tile manifest with
`--triplet_manifest` and `--fixed_split_manifest` (see
[Retrain with the open-water supplement](#retrain-with-the-open-water-supplement)).

`paper_tile` exactly reproduces the published tile-level split. `grouped` is also
available for future experiments and keeps rows with the same `grid_id`/`group_id` in
one partition; it is not the protocol used to train the released model.

For the matched S1-only control, omit `--aef_dir` and use
`--n_aef_bands 0 --n_proj_bands 0`.

### Retrain with the open-water supplement

`outputs/audit/supplement_samples.csv` defines 600 additional lake, reservoir,
coastal/estuarine, and seasonal-water grids with exact Sentinel-1 and Dynamic World
scene IDs. It is written by `scripts/sample_water_feature_supplement.py` and is not
tracked in this repository. The centroids, split and dates of the 600 tiles used for the
released models are in `metadata/training_sample_centroids.geojson`; their full table
(scene IDs, raster grids) is not published.
The supplement is not materialized until the export workflow is run.

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
with the `label` class only, keeping the same chips, split, seeds and hyperparameters. Those runs
produced the released checkpoints. `results/paper_labelclass_v1/58321212/` holds their evaluation under
the earlier 3 m-input protocol, which is superseded by the 10 m protocol of the paper (see
[Results](#results)). The 3 m-input analyses accept either run set:

```bash
python scripts/complete_paper_analyses.py sweep --run-set labelclass_v1 --out results/.../threshold_sweep.json
python scripts/complete_paper_analyses.py paired --run-set mixed_v1 --out /tmp/paired_old.json
python scripts/make_figure3_iou_distribution.py --run-set labelclass_v1 --out-dir results/.../figures
```

## Results

All values are those of the paper. Every number, and the file and field it comes
from, is listed in [`results/paper/README.md`](results/paper/README.md).

Independent 53-scene PlanetScope (GSWD) reference set. Inference at 10 m, probabilities resampled
bilinearly to the 3 m reference grid, threshold 0.30, seed 42, common valid pixels (valid in the
reference, every model run and OPERA; 51,897,488 pixels, 93.4% of the reference pixels):

| Method | Precision | Recall | Dice | Pooled IoU | Per-scene mean [95% CI] |
| --- | ---: | ---: | ---: | ---: | --- |
| S1 only | 0.883 | 0.854 | 0.869 | 0.768 | 0.617 [0.527, 0.702] |
| S1 + AEF (t) | 0.891 | 0.950 | 0.920 | 0.851 | 0.741 [0.669, 0.806] |
| OPERA DSWx-S1 | 0.859 | 0.851 | 0.855 | 0.747 | 0.591 [0.508, 0.670] |

Seeds 42-44, pooled IoU (mean +/- sample SD): 0.764 +/- 0.005 (S1 only), 0.850 +/- 0.001 (S1 + AEF).
Per-scene wins out of 53 scenes: S1 + AEF > S1 only in 44, S1 + AEF > OPERA in 48, S1 only > OPERA
in 36 (Wilcoxon signed-rank p = 8.4e-7, 7.4e-10 and 0.015).

**Embedding year.** With the previous year's embedding, AEF(t-1), at inference the pooled IoU is 0.846
(per-scene 0.732 [0.662, 0.797]). A model trained and applied with AEF(t-1) reaches 0.848
(0.734 [0.663, 0.800]).

**Ablations** (Table II; per-scene IoU, threshold 0.30, common valid pixels, mean +/- sample SD over
three seeds; `results/paper/table_II/`):

| AEF width k | 0 (S1 only) | 1 | 2 | 3 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Per-scene IoU | 0.624 +/- 0.007 | 0.730 +/- 0.006 | 0.737 +/- 0.002 | 0.741 +/- 0.004 | 0.741 +/- 0.002 | 0.743 +/- 0.001 | 0.741 +/- 0.001 |

| Training tiles (k = 16) | 1,000 | 2,000 | 3,000 | 4,222 |
| --- | ---: | ---: | ---: | ---: |
| Per-scene IoU | 0.666 +/- 0.011 | 0.740 +/- 0.001 | 0.742 +/- 0.002 | 0.741 +/- 0.001 |

The `*_std` fields and `.tex` tables in that folder use the population SD, which is lower in the third
decimal for k = 0, k = 1 and 1,000 tiles; see its README.

Independent S1S2-Water benchmark, 15 test scenes, 30 m grid, pooled water IoU, mean +/- SD over three
seeds (`external_validation/s1s2_water/`):

| Method | Threshold 0.30 | Threshold 0.50 |
| --- | ---: | ---: |
| S1 + AEF | 0.941 +/- 0.001 | 0.953 +/- 0.001 |
| S1 only | 0.794 +/- 0.026 | 0.822 +/- 0.024 |
| OPERA DSWx-S1 (v1.2 with the upstream boundary fix) | 0.869 | 0.869 |

**Resolution-matched comparison with OPERA** (Supplementary Table S2). OPERA DSWx-S1 is a 30 m product.
On the 53 GSWD scenes the S1 + AEF - OPERA difference in pooled IoU is 0.104 under the paper's scoring,
0.106 with both scored at 30 m (model probabilities area-averaged from 10 m), 0.095 when the model is
given OPERA's treatment (a binary 30 m map scored at 3 m), 0.091 on 30 m cells that are entirely water
or entirely land, and 0.103 on OPERA's native grid (46 scenes). Values and the per-scene OPERA grid
offsets are in `results/paper/table_S2/`.

### Superseded (3 m-input protocol)

Earlier versions of the manuscript ran the networks on the 3 m reference grid (inputs resampled to 3 m)
and scored each method on its own valid pixels. Those values are kept as provenance for earlier versions
and are not the paper's results:

| Method (superseded, 3 m inputs, threshold 0.30) | Precision | Recall | Pooled water IoU | Per-scene mean [95% CI] |
| --- | ---: | ---: | ---: | --- |
| S1 + AEF (acquisition year) | 0.892 | 0.948 | 0.851 | 0.756 [0.689, 0.816] |
| S1 + AEF (previous year) | 0.882 | 0.946 | 0.840 | 0.741 [0.672, 0.803] |
| S1 only | 0.933 | 0.582 | 0.558 | 0.443 [0.355, 0.533] |
| OPERA DSWx-S1 | 0.858 | 0.852 | 0.746 | 0.591 [0.508, 0.670] |

Source: `results/paper_labelclass_v1/58321212/` (`paper_metrics.csv`, `completion_results.md`).
`results/paper_retrain_openwater_v1/57725872/` holds the older mixed-label results. The top-level
`results/paper_metrics.csv` and `results/ablation_metrics.csv` are from the first release (20 August
2026; mixed-label models, 3,742 training tiles) and are also superseded; they are kept unchanged. See
[`results/README.md`](results/README.md).

## Reproducing the paper

The protocol is: 10 m inference -> bilinear resampling of the probabilities to the 3 m reference grid ->
threshold 0.30 -> scoring on the common valid mask. The commands, in order, with the environment
variables they need, are in [`scripts/eval_10m/README.md`](scripts/eval_10m/README.md):

1. Export the 10 m S1 and AEF inputs and write the inference manifests (`build_10m_inputs.py`,
   `verify_10m_inputs.py`, `make_10m_manifests.py`).
2. Run inference with the paper's wrappers (`jobs/eval_10m/submit_10m_inference.sh`,
   `jobs/eval_10m/submit_ablation_10m_inference.sh`). S1-only uses
   `runtime/infer_intercomparison_s1_tiles.py`, S1 + AEF uses `runtime/infer_pnw_s1aef_tiles.py`.
3. Resample and score: `score_10m.py` (Table I and paired tests), `score_ablations.py`,
   `aggregate_ablations.py` and `make_table_ii.py` (Table II).
4. Figures and analyses: `export_53_scene_layers_10m.py` and `make_figure1b_10m.py` (Fig. 1B), the
   WorldCover error analysis, and `g0_inventory.py`, `gswd_rescore.py`, `gswd_summary.py`,
   `s1s2_rescore.py`, `s1s2_summary.py` (Supplementary Table S2).

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/train.py` | Model, losses, augmentation, training schedule, validation, ablations |
| `src/infer.py` | Geospatial tiled inference and four-flip TTA |
| `scripts/download_model.py` | Download and checksum a released checkpoint and its `band_stats.npz` |
| `scripts/build_sample_index.py` | Regenerate the public sample index from local tiles |
| `scripts/reconstruct_sample.py` | Rebuild an indexed triplet from public source archives |
| `scripts/validate_dataset.py` | Validate bands, grids, names, and optional sample index |
| `src/evaluate.py`, `scripts/compare_methods.py` | Pooled/per-scene metrics, bootstrap CIs, paired tests |
| `scripts/eval_10m/`, `jobs/eval_10m/` | 10 m evaluation protocol of the paper: inputs, inference, scoring, ablations, figures, resolution-matched OPERA comparison |
| `scripts/complete_paper_analyses.py` | Threshold sweep, proximity screen, matched mask, paired tests vs OPERA (3 m-input protocol) |
| `scripts/make_figure3_iou_distribution.py` | Per-scene IoU distribution figure (3 m-input protocol) |
| `external_validation/s1s2_water/` | Independent S1S2-Water evaluation against OPERA DSWx-S1 |
| `jobs/` | Slurm job scripts for the training, inference and evaluation chain |
| `metadata/` | Training and independent-evaluation sampling metadata |
| `scripts/build_training_centroids.py` | Write `metadata/training_sample_centroids.geojson` |
| `results/paper/` | Machine-readable values of the paper |
| `results/` | Earlier (superseded) paper and training-run summaries; see `results/README.md` |
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

Original code is released under Apache License 2.0. The checkpoints use the same
license. Third-party datasets and software retain their original terms.
