# 10 m evaluation protocol (manuscript v11)

These scripts produced every number of manuscript v11 (29 September 2026) that is computed on the 53 PlanetScope
(GSWD) reference scenes: Table I and its paired tests, Table II (width and training-size ablations), the Fig. 1B panel
values, the WorldCover error analysis, and the resolution-matched comparison with OPERA DSWx-S1 (Supplementary
Table S2). The numbers are in [`results/paper_v11_10m/`](../../results/paper_v11_10m/README.md).

## Protocol

1. **Inputs at 10 m.** For each scene, Sentinel-1 GRD (`VV`, `VH`, `angle`; the manifest's exact `scene_id`, Earth
   Engine `COPERNICUS/S1_GRD`) and the AlphaEarth v1 annual embedding (acquisition year, and the previous year for the
   t-1 runs) are exported on a 768 x 768 grid of 10 m pixels. The grid is in the label CRS, has its edges at integer
   multiples of 10 m, and is centred on the ~3 km label footprint. 10 m is the training resolution.
2. **Inference at 10 m.** Tile 512, overlap 64, batch 4, four-flip TTA, fp16, band statistics from each run
   directory, with a `--validate-only` pass first.
3. **Resampling.** Each 10 m probability raster is reprojected to the 3 m label grid with `rasterio.warp.reproject`,
   `Resampling.bilinear`, nodata -9999 (`score_10m.resample_to_label`).
4. **Threshold.** Water where `probability >= 0.30`, applied on the 3 m grid. 0.50 is reported as a secondary value.
5. **Common valid mask.** A pixel is scored when the label is 0 or 1 and the pixel is valid in OPERA, in the four
   seed-42 10 m rasters (S1-only, S1+AEF(t), S1+AEF(t-1) swap, S1+AEF(t-1) trained) and in the corresponding four 3 m
   rasters of the earlier protocol. That is 51,897,488 pixels, 93.4% of the reference pixels. The 10 m rasters remove
   no pixel from the earlier mask. Every method is scored on this one pixel set.
6. **Statistics** (`scoring.py`). Pooled precision, recall, Dice and IoU sum TP/FP/FN over scenes. The per-scene mean
   has a bootstrap 95% CI: `np.random.default_rng(42)`, 10,000 resamples of scene indices, 2.5% and 97.5% quantiles.
   The scene list is sorted as strings (SID03, ..., SID100, SID14, ...). The CI depends on this order: numeric order
   moves the endpoints by up to 0.003. Paired tests use `scipy.stats.wilcoxon` with its defaults, and wins, losses and
   ties are counted from the sign of the per-scene IoU difference.
7. **Seeds.** Seed means are reported with the sample SD (n - 1) over seeds 42, 43 and 44. The `*_std` fields
   written by `aggregate_ablations.py` and the `table_II_10m*.tex` files use the population SD (`statistics.pstdev`,
   as in `scripts/aggregate_paper_retrain.py`). Manuscript v11 Table II uses the sample SD. The two agree to three
   decimals except for k = 0, k = 1 and 1,000 tiles.

The earlier protocol ran the networks on 3 m inputs (the S1 and AEF layers resampled to the label grid). Its results
are in `results/paper_labelclass_v1/58321212/` and are superseded. Running at 10 m changes S1-only most (pooled IoU
0.567 -> 0.768 on the common mask) and changes S1+AEF little (0.853 -> 0.851).

## Which inference code reproduces the paper

The paper numbers come from the wrappers in `runtime/`, called by `jobs/eval_10m/run_*_inference.sh`:

| Model | Wrapper | Core |
| --- | --- | --- |
| S1-only (k = 0) | `runtime/infer_intercomparison_s1_tiles.py` | `runtime/infer_s1_probability_models.py` |
| S1+AEF (k > 0) | `runtime/infer_pnw_s1aef_tiles.py --n-proj-bands <k>` | `runtime/infer_s1_probability_models.py` |

Both read an input manifest (one row per scene, with `s1_path` and `aef_path`). They build the network from each
training run's pinned trainer (`--train-script <run_dir>/runtime/train.py`, sha1
`cd991e7424ae8d74c4737bd339f1ca66e563e506` in all 33 label-class runs) and load the run's checkpoint and band
statistics. The same wrappers and arguments were used for the earlier 3 m protocol
(`jobs/paper_retrain_infer_array.sbatch`).

`src/infer.py` is a cleaned-up copy of the same core (same tiling, TTA, normalisation and nodata handling) with a
different command line: dated S1 scenes plus a single AEF raster, no manifest. It lacks the core's `--aef-ablation`
option and does not pass `use_permanent_prior`, `prior_dropout_p` and `target_source_id` when it instantiates the
model. It was not used for any paper number, and it has not been checked to give bit-identical probabilities. Use
`src/infer.py` to map new scenes; use the wrappers here to reproduce the paper.

The jobs point `--run-dir` at the training run directories (`OPENWATER_RUNS`, `TMINUS1_RUNS`). Running them against
a bundle downloaded with `scripts/download_model.py` instead has not been tested.

## Environment variables

The check runs hard-coded NERSC paths. The repository copies read these variables instead (`paths.py`):

| Variable | Contents | Used by |
| --- | --- | --- |
| `SWM_GSWD_DIR` | GSWD reference data: `labels/<SID>.tif`, `PS/` (PlanetScope), `S1_Intercomparison_All_Scenes.csv` | scoring, figures |
| `SWM_S1ML` | `intercomparison_s1aef/` (`manifests/`, `predictions/paper_labelclass_v1/` 3 m rasters, `worldcover/`, `cache/`), `intercomparison_opera/` (`grouped/`, `binary_masks/`), `training_runs/paper_labelclass_v1/` (evaluation reports) | inputs, scoring, ablations, WorldCover |
| `SWM_EVAL10M` | output root of the main 10 m run: `inputs_10m/`, `predictions_10m/`, `predictions_10m_on_3m/`, `task1_10m/manifests/`, `task1_10m/results/` | all |
| `SWM_ABLATION10M` | output root of the ablation, Fig. 1B and WorldCover run | ablations, figures, WorldCover |
| `SWM_RESOLUTION` | output root of the resolution-matched rescoring (`gswd/`, `s1s2water/`) | Supplementary Table S2 |
| `SWM_S1S2` | S1S2-Water validation root (`evaluation/grids_30m.csv`, `work/model/`, `work/opera/`), see `external_validation/s1s2_water/` | Supplementary Table S2 |
| `ENV_PREFIX` | conda environment with `requirements.txt` (the paper used PyTorch 2.6.0+cu124) | inference jobs |
| `OPENWATER_RUNS`, `TMINUS1_RUNS` | run directories of training arrays 58321212 and 58321217 | inference jobs |

On NERSC the paper used `SWM_GSWD_DIR=$HOME/Workspace/data/S1_Intercomparison_All_Scenes_subset`,
`SWM_S1ML=$PSCRATCH/S1ML`, `SWM_S1S2=$PSCRATCH/surface_water_validation` and
`SWM_EVAL10M`, `SWM_ABLATION10M`, `SWM_RESOLUTION` = `$PSCRATCH/S1ML/revision_checks_20260929`,
`revision_checks_20260930` and `revision_checks_resolution`.

## Reproducing the paper

Run from the repository root with the variables above set. `E=scripts/eval_10m`.

```bash
E=scripts/eval_10m
mkdir -p $SWM_EVAL10M/task1_10m $SWM_ABLATION10M/{ablations_10m,worldcover_10m/inputs,fig1B/ancillary} $SWM_RESOLUTION/gswd

# 1. 10 m inputs (Earth Engine for S1; Source Cooperative for AlphaEarth)
python $E/build_10m_inputs.py --out-root $SWM_EVAL10M/inputs_10m --products grid
python $E/build_10m_inputs.py --out-root $SWM_EVAL10M/inputs_10m --products s1 aef_t aef_tminus1 --ee-project <PROJECT>
python $E/verify_10m_inputs.py --inputs-root $SWM_EVAL10M/inputs_10m --out-csv $SWM_EVAL10M/task1_10m/verify_10m_inputs.csv
python $E/build_10m_daymosaic_control.py --inputs-root $SWM_EVAL10M/inputs_10m   # control for SID05/21/61/63 only
python $E/make_10m_manifests.py --inputs-root $SWM_EVAL10M/inputs_10m --out $SWM_EVAL10M/task1_10m

# 2. Inference (one GPU node; DRY_RUN=1 prints the salloc command)
DRY_RUN=0 bash jobs/eval_10m/submit_10m_inference.sh            # k0, k16 (seeds 42-44), t-1 runs, controls
DRY_RUN=0 bash jobs/eval_10m/submit_ablation_10m_inference.sh   # k = 1, 2, 3, 4, 8 and 1,000-3,000 tiles

# 3. Table I, paired tests (resampling to 3 m happens here)
python $E/score_10m.py                          # -> $SWM_EVAL10M/task1_10m/results/

# 4. Table II
python $E/score_ablations.py --res 10m --out $SWM_ABLATION10M/ablations_10m/per_scene_counts_10m.csv
python $E/score_ablations.py --res 3m --out $SWM_ABLATION10M/ablations_10m/per_scene_counts_3m.csv
python $E/aggregate_ablations.py --counts $SWM_ABLATION10M/ablations_10m/per_scene_counts_3m.csv \
  --out-prefix $SWM_ABLATION10M/ablations_10m/ablation_3m --check-3m
python $E/aggregate_ablations.py --counts $SWM_ABLATION10M/ablations_10m/per_scene_counts_10m.csv \
  --out-prefix $SWM_ABLATION10M/ablations_10m/ablation_10m
python $E/make_table_ii.py

# 5. WorldCover error analysis (ESA WorldCover 2021 v200 tiles cached under $SWM_S1ML/intercomparison_s1aef/worldcover/)
python $E/make_worldcover_10m_inputs.py
EV=$SWM_S1ML/training_runs/paper_labelclass_v1/openwater/reports/58321212/evaluation
python $E/analyze_s1_vs_s1aef_worldcover_errors.py --threshold 0.30 --skip-download \
  --s1-csv $SWM_ABLATION10M/worldcover_10m/inputs/width_k0_seed42_current_tta_10m_per_sample_inputs.csv \
  --s1aef-csv $SWM_ABLATION10M/worldcover_10m/inputs/width_k16_seed42_current_tta_10m_per_sample_inputs.csv \
  --s1-summary $EV/width_k0_seed42_current_tta_paper030/width_k0_seed42_current_tta_paper030_confusion_summary.json \
  --s1aef-summary $EV/width_k16_seed42_current_tta_paper030/width_k16_seed42_current_tta_paper030_confusion_summary.json \
  --output-dir $SWM_ABLATION10M/worldcover_10m/inference_10m
python $E/worldcover_shares.py $SWM_ABLATION10M/worldcover_10m/inference_10m

# 6. Fig. 1B (Natural Earth 1:50m countries is used only to assign continents)
curl -sSfL -o $SWM_ABLATION10M/fig1B/ancillary/ne_50m_admin_0_countries.geojson \
  https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson
python $E/export_53_scene_layers_10m.py --images-dir $SWM_ABLATION10M/images_10m/planetscope_53_scenes
python $E/check_exported_png_iou.py
python $E/make_figure1b_10m.py                  # -> fig1B/figure1B_10m_{current,alt1,alt2}.{png,pdf}, panel_iou.csv

# 7. Supplementary Table S2: resolution-matched comparison with OPERA
python $E/g0_inventory.py
python $E/gswd_rescore.py --procs 8
python $E/gswd_summary.py
DRY_RUN=0 PYTHON_BIN=$(command -v python) bash jobs/eval_10m/submit_s1s2_rescore.sh
python $E/s1s2_summary.py
```

Optional checks: `reproduce_paper_common_mask.py` recomputes the earlier 3 m Table I from the 3 m rasters, and
`submit_hwcheck_10m_inference.sh` with `compare_hwcheck.py` reran k0 and k16 seed 42 on A100-80GB nodes (bit-identical
probabilities to the A100-40GB run).

The committed `manifests/` are the manifests of the paper run. Their `s1_path`, `aef_path` and `label_path` columns
are absolute NERSC paths; `make_10m_manifests.py` writes new ones for your own `inputs_10m/`. `grids_10m.csv` lists
the 10 m grid of every scene.

## Changes from the check-run copies

Paths only. Hard-coded NERSC paths were replaced by the environment variables above, by repository-relative
argparse defaults (in the `runtime/` wrappers and the helper modules, whose path constants are only argparse
defaults), or by `required=True` for arguments the paper run always passed
(`analyze_s1_vs_s1aef_worldcover_errors.py`). `s1s2_summary.py` reads the repository copy of
`s1s2water_per_scene_all_methods.csv`, which is identical (md5) to the file the check run read.
`s1s2_rescore.py` imports the repository's `external_validation/s1s2_water/evaluation/evaluate_intercomparison.py`,
which is identical to the copy the check run used.

`jobs/eval_10m/run_10m_inference.sh` also no longer ends with `grep`: job 59050165 was recorded by Slurm as FAILED
(exit 1) only because that `grep` found no error rows. It now prints the number of error rows and exits 0, as the
later ablation script already did.

On 29 September 2026 the repository copies were rerun on NERSC against the check-run 10 m probabilities (read
only), with fresh output folders, so the 10 m -> 3 m resampling was also recomputed. The outputs were compared
with the check-run outputs:

| Step | Result |
| --- | --- |
| `score_10m.py` | `summary_thr0.30.json`, `summary_thr0.50.json`, `per_scene_counts_long.csv`, `per_scene_3m_vs_10m.csv` byte-identical |
| `score_ablations.py` (3 m and 10 m), `aggregate_ablations.py`, `make_table_ii.py` | per-scene counts, per-run CSVs, `table_II_10m.tex`, `table_II_10m_thr030_common.tex`, `table_II_3m_vs_10m.md` byte-identical |
| `make_worldcover_10m_inputs.py`, `analyze_s1_vs_s1aef_worldcover_errors.py`, `worldcover_shares.py` | `worldcover_error_classes.csv` byte-identical; `chip_delta_iou.csv`, `class_shares.json`, `summary.json`, `report.md` equal except for recorded file paths and timestamps |
| `export_53_scene_layers_10m.py`, `check_exported_png_iou.py`, `make_figure1b_10m.py` | `panel_iou.csv`, `scene_selection_rule.md`, `scene_table.csv` byte-identical; IoUs from the exported PNGs equal the scoring in 106 of 106 scene-model pairs |
| `g0_inventory.py`, `gswd_rescore.py`, `gswd_summary.py`, `results_table.py` | `opera_grid_offsets.csv`, `opera_products_per_scene.csv`, per-scene counts and support, `summary_thr030.json`, `summary_thr050.json`, `reproduction_check.json` byte-identical; Markdown tables equal row for row |
| `submit_s1s2_rescore.sh`, `s1s2_rescore.py`, `s1s2_summary.py` | interactive CPU job 59100957: `per_scene_counts_long.csv`, `per_scene_support.csv`, `opera_grid_offsets.csv`, `summary_thr030.json`, `summary_thr050.json` byte-identical; `reproduction_check.json` equal except for the recorded path of the supplement CSV; Markdown tables equal row for row |

The input export (`build_10m_inputs.py`, Earth Engine and Source Cooperative) and the GPU inference jobs were not
rerun.

| Repository file | Source (check run) | Source md5 | Edit |
| --- | --- | --- | --- |
| `runtime/infer_intercomparison_s1_tiles.py` | 29 Sep run: `task1_10m/runtime/` | `a2bdf98abb0ef553af319fea87fa08e9` | paths only |
| `runtime/infer_pnw_s1aef_tiles.py` | 29 Sep run: `task1_10m/runtime/` | `f1c9e3595d0c6cfbb255ed7a6a1e8dcd` | paths only |
| `runtime/infer_s1_probability_models.py` | 29 Sep run: `task1_10m/runtime/` | `b05427354f5bbff133a703be6af210a1` | paths only |
| `fetch_alphaearth_embeddings.py` | NERSC `Workspace/code/` (imported by `build_10m_inputs.py`) | `9e766f42ce764968cb78e4c41e892f21` | unchanged |
| `select_intercomparison_nearest_s1.py` | NERSC `Workspace/code/` (imported by `build_10m_inputs.py`) | `521a56b3fcd09faadca8219ba8ca39a0` | paths only |
| `download_intercomparison_s1aef_label_products.py` | NERSC `Workspace/code/` (imported by `build_10m_daymosaic_control.py`) | `f17f5886fefcebe352bc819cd7a78172` | paths only |
| `evaluate_intercomparison_opera_binary.py` | resolution run: `scripts/orig/` | `5dbea85e96317983108d34d5dd73d1b6` | paths only |
| `make_permanent_water_comparison_figures.py` | 30 Sep run: `scripts/` | `6e2e9ffec9f84b6b91486cb1904a7724` | paths only |
| `analyze_s1_vs_s1aef_worldcover_errors.py` | 30 Sep run: `scripts/` | `1deef133ae0f63ceaf70c616f9724b83` | paths only |
| `build_10m_inputs.py` | 29 Sep run: `task1_10m/scripts/` | `e102956ca0b84eef96c0d900428abbd9` | paths only |
| `build_10m_daymosaic_control.py` | 29 Sep run: `task1_10m/scripts/` | `d1f79006be42bc3667e209934d958e2b` | paths only |
| `verify_10m_inputs.py` | 29 Sep run: `task1_10m/scripts/` | `4fe0c69035c9be6add71aff26655d9fc` | paths only |
| `make_10m_manifests.py` | 29 Sep run: `task1_10m/scripts/` | `f6dbf5db6f0f3769117904f09893bac3` | paths only |
| `scoring.py` | 29 Sep run: `task1_10m/scripts/` (same file in the 30 Sep and resolution runs) | `5aaae64d3872bf9ac3305af9f5dcdb97` | paths only |
| `score_10m.py` | 29 Sep run: `task1_10m/scripts/` (same file in the 30 Sep and resolution runs) | `e0cbfcc305a4a88106e183a9df4e133e` | paths only |
| `reproduce_paper_common_mask.py` | 29 Sep run: `task1_10m/scripts/` | `94aa027d3b247d9be2fcec25765bd2f4` | unchanged |
| `score_ablations.py` | 30 Sep run: `scripts/` | `81895723b7fe703103e6d7a7bb582841` | paths only |
| `aggregate_ablations.py` | 30 Sep run: `scripts/` | `0662eee07139bfab48cdda6dada032cf` | paths only |
| `make_table_ii.py` | 30 Sep run: `scripts/` | `1974cf94dde8984d708d8795d87538f6` | paths only |
| `compare_hwcheck.py` | 30 Sep run: `scripts/` | `32335adefe083d5e25bc1985848ef6ae` | paths only |
| `export_53_scene_layers_10m.py` | 30 Sep run: `scripts/` | `57238c87aa4de8e148e3f520de58e2ed` | paths only |
| `check_exported_png_iou.py` | 30 Sep run: `scripts/` | `1d352f7830a32f19a880ab0c141d48a4` | paths only |
| `make_figure1b_10m.py` | 30 Sep run: `scripts/` | `b19c2f856c394eb4ff5e2ddf78b29f35` | unchanged |
| `make_worldcover_10m_inputs.py` | 30 Sep run: `scripts/` | `d090e3d165bfda1ea8d4ab2a4851601d` | paths only |
| `worldcover_shares.py` | 30 Sep run: `scripts/` | `a8af4d3954e2a3ba4e823c677b113a27` | unchanged |
| `worldcover_tables_md.py` | 30 Sep run: `scripts/` | `ccacd71f0c10cf59e33b26e18746a805` | paths only |
| `g0_inventory.py` | resolution run: `scripts/` | `1721739247b218fa7755e83619b6e266` | paths only |
| `gswd_rescore.py` | resolution run: `scripts/` | `f75c386cd5735dd757af52afd78d6260` | paths only |
| `gswd_summary.py` | resolution run: `scripts/` | `acaa752405aca9a31fe29dee496cfaf8` | paths only |
| `s1s2_rescore.py` | resolution run: `scripts/` | `5cfeaf4ebeb10399b8df1aa553832aa2` | paths only |
| `s1s2_summary.py` | resolution run: `scripts/` | `6c7596810b0881f748a111bee92f0ce3` | paths only |
| `results_table.py` | resolution run: `scripts/` | `56f36b0e8a383e2bb62e20d84a0e29d8` | unchanged |
| `jobs/eval_10m/run_10m_inference.sh` | 29 Sep run: `task1_10m/scripts/` | `005653bac081b19854902ab8392cbf08` | paths; final `grep` replaced |
| `jobs/eval_10m/run_ablation_10m_inference.sh` | 30 Sep run: `scripts/` | `f3e9444368b9033ba00c343a6d4913cb` | paths only |
| `jobs/eval_10m/run_hwcheck_10m_inference.sh` | 30 Sep run: `scripts/` | `f501710de06d97cbb950331e48260709` | paths only |
| `jobs/eval_10m/submit_10m_inference.sh` | 29 Sep run: `task1_10m/scripts/` | `74901f07ca8dce2e751b0267599bd616` | paths only |
| `jobs/eval_10m/submit_ablation_10m_inference.sh` | 30 Sep run: `scripts/` | `5b10c67a13a8eb9f18cced8f421dc705` | paths only |
| `jobs/eval_10m/submit_hwcheck_10m_inference.sh` | 30 Sep run: `scripts/` | `b517ffe34e98fee0457e89640f747dd1` | paths only |
| `jobs/eval_10m/submit_s1s2_rescore.sh` | resolution run: `scripts/` | `7bc0206ef76ce9b2e2e428ccd2cd9c8f` | paths only |

The seven files in `manifests/` are byte-identical copies of the 29 Sep run's `task1_10m/manifests/*.csv` and
`task1_10m/grids_10m.csv`.
