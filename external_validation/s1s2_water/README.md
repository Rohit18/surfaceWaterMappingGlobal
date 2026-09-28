# External validation on S1S2-Water

Independent, non-flood evaluation of the released models against OPERA DSWx-S1 on the S1S2-Water benchmark
(Wieland et al., 2024, IEEE JSTARS 17:1084-1099; data at https://zenodo.org/records/11278238). The benchmark
provides expert-corrected water masks for Sentinel-1 scenes of 100 km x 100 km; none of its data was used for
training, model selection or threshold selection.

The scripts run on a Slurm cluster and reference absolute paths of the machine they were run on
(`/pscratch/sd/r/rohit9/surface_water_validation`). They are published as the record of what was executed; adapt
the roots before reuse.

## Scenes

The 16 official test scenes are 23, 25, 29, 31, 33, 35, 47, 53, 57, 75, 77, 78, 80, 82, 88 and 89. Scene 31 is
excluded: its metadata lists no source Sentinel-1 product and its image does not follow the documented encoding,
so the acquisition cannot be reproduced (`results/scene31_exclusion_audit.json`). The remaining 15 are evaluated.

## Pipeline

| Step | Script | Output |
| --- | --- | --- |
| Download the benchmark test assets from Zenodo by HTTP range | `inputs/remote_zip.py` | S1 image, water mask, valid mask per scene |
| Freeze one 30 m evaluation grid per scene, before any method is run | `inputs/freeze_evaluation_grids.py` | `results/grids_30m.csv` |
| Sentinel-1 model input from Earth Engine `COPERNICUS/S1_GRD` at 10 m, exact GRD products | `inputs/export_gee_s1_scene.py` | `inputs_gee10m/s1_<date>.tif` |
| AlphaEarth annual embedding on the same grid | `inputs/download_scene_aef.py --pixel-size 10` | `inputs_gee10m/alphaearth_<year>.tif` |
| Model inference (four-flip TTA) | `jobs/run_models_labelclass_array.sbatch`, `jobs/run_*_interactive.sh` | water probability per scene |
| OPERA: SLC and orbit retrieval, ancillary selection, RTC-S1 v1.0.4, DSWx-S1 v1.2 | `opera/*.py`, `jobs/run_rtc_array.sbatch`, `jobs/run_dswx_array.sbatch` | DSWx B02 BWTR products |
| Scoring of all methods on common pixels | `evaluation/evaluate_labelclass_models.py`, `evaluation/evaluate_intercomparison.py`, `evaluation/s1s2water_supplement.py` | `results/*.csv`, `results/s1s2water_summary.json` |

`EVALUATION_DESIGN.md` is the evaluation policy, written and frozen on 2026-09-14 before any method output was
inspected.

## Two points that affect reproduction

1. **Sentinel-1 source.** The benchmark ships its own Sentinel-1 rasters, processed differently from the Earth
   Engine collection the models were trained on (6-10 m pixels; on scene 57 the benchmark VH is 1.96 dB lower).
   Using the benchmark rasters as model input lowers the S1-only pooled IoU from 0.847 to 0.750, and the fused
   model from 0.951 to 0.949 (seed 42, 30 m, threshold 0.50). The reported evaluation uses Earth Engine inputs,
   as in the paper's own pipeline.
2. **DSWx-SAR v1.2 defect.** In the tagged v1.2 release, the final pass of `remove_false_water_bimodality_parallel`
   excludes water bodies that touch the image edge, which removed the dominant water body on scene 57 (water IoU
   0.052). The upstream commit `e7f20a86ba6757d698220fec00713dea600356d0`, merged after the tag, fixes it; all
   products here were generated with that correction. Report OPERA results as "v1.2 with the upstream boundary
   fix", not as unmodified v1.2.

## Results

30 m grid, common valid pixels of all methods, pooled water IoU (mean over three training seeds):

| Method | Threshold 0.50 | Threshold 0.30 |
| --- | ---: | ---: |
| S1 + AEF | 0.953 | 0.941 |
| OPERA DSWx-S1 (corrected) | 0.869 | 0.869 |
| S1-only | 0.822 | 0.794 |

Per-scene values are in `results/s1s2water_supplementary_table.csv`; all methods and both thresholds are in
`results/s1s2water_per_scene_all_methods.csv`.
