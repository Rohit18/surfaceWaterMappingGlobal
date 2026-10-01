# Paper results (10 m inference)

Machine-readable values behind the paper. The files are copies of the check-run outputs of
28-29 September 2026 on NERSC and were not recomputed for this folder. The scripts that produced them are in
[`scripts/eval_10m/`](../../scripts/eval_10m/README.md). Rerunning those scripts from the repository against the same
predictions gave the same values; that README lists the comparison for each step.

Protocol: 10 m inference -> bilinear resampling to the 3 m label grid -> threshold 0.30 (`>=`) -> common valid mask
(51,897,488 pixels, 93.4% of the reference pixels, 53 scenes). Seed 42 unless stated. Seed spreads are the sample SD
(n - 1) over seeds 42, 43 and 44.

## Where each number is

| Manuscript item | File | Field |
| --- | --- | --- |
| Table I, S1 only | `table_I/summary_thr0.30.json` | `common_mask.methods["S1-only [10m]"]`: `precision`, `recall`, `dice`, `pooled_water_iou`, `per_scene_mean`, `per_scene_ci95` |
| Table I, S1 + AEF (t) | same | `common_mask.methods["S1+AEF(t) [10m]"]` |
| Table I, OPERA DSWx-S1 | same | `common_mask.methods["OPERA DSWx-S1"]` |
| Table I at threshold 0.50 | `table_I/summary_thr0.50.json` | same keys |
| Seeds 42-44, pooled IoU | `table_I/summary_thr0.30.json` | `common_mask.methods["S1-only seed {42,43,44} [10m]"]` and `["S1+AEF(t) seed {42,43,44} [10m]"]`, `pooled_water_iou`; mean and sample SD of the three values |
| Paired tests (wins, mean and median difference, Wilcoxon p) | `table_I/summary_thr0.30.json` | `common_mask.paired_10m["S1+AEF(t) [10m] vs S1-only [10m]"]`, `["S1+AEF(t) [10m] vs OPERA DSWx-S1"]`, `["S1-only [10m] vs OPERA DSWx-S1"]`, `["S1+AEF(t) [10m] vs S1+AEF(t-1) swap [10m]"]`; other seeds: `paired_seed43`, `paired_seed44` |
| Embedding year: AEF(t-1) at inference | `table_I/summary_thr0.30.json` | `common_mask.methods["S1+AEF(t-1) swap [10m]"]` |
| Embedding year: trained and applied with AEF(t-1) | same | `common_mask.methods["S1+AEF(t-1) trained [10m]"]` |
| Per-scene IoU (all models, 3 m and 10 m inference, controls) | `table_I/per_scene_3m_vs_10m.csv`; counts in `table_I/per_scene_counts_long.csv` | one row per scene and threshold; `mask == "common"` for the paper's pixel set |
| Table II, per-scene IoU (threshold 0.30, common mask) | `table_II/ablation_10m_summary.json` | `configs[]` with `convention == "thr030_common"`: mean of `per_scene_mean_iou_seeds`, SD = `statistics.stdev` of the same list |
| Table II, recall and pooled IoU | same | `pooled_recall_seeds`, `pooled_iou_seeds` |
| Table II, original convention (threshold 0.50, own valid pixels) | same | `convention == "thr050_own"` |
| Table II per run and per scene | `table_II/ablation_10m_per_run.csv`, `table_II/per_scene_counts_10m.csv` | |
| Table II, 3 m and 10 m side by side | `table_II/table_II_3m_vs_10m.md` | |
| WorldCover error analysis | `worldcover/inference_10m/class_shares.json`, `worldcover/worldcover_3m_vs_10m.md` | `chip_groups` (24 / 26 / 3 scenes), class shares of corrected and added FN/FP |
| WorldCover per scene | `worldcover/inference_10m/chip_delta_iou.csv`, `worldcover_error_classes.csv` | |
| Fig. 1B panel IoUs | `fig1B/panel_iou.csv` | one row per version (`current`, `alt1`, `alt2`) and panel; four decimals |
| Fig. 1B scene selection | `fig1B/scene_selection_rule.md`, `fig1B/scene_table.csv` | |
| Supplementary Table S2 (GSWD) | `table_S2/gswd/summary_thr030.json` | `<variant>.vs_opera["S1+AEF(t) seed 42"].pooled_iou_diff`; variants `paper` (main, 0.104), `g1_average` (i, 0.106), `g2` (ii, 0.095), `g3_30m` (iii, 0.091), `g4` (iv, 0.103) |
| Supplementary Table S2, readable | `table_S2/gswd/summary_table_thr030.md`, `key_numbers.txt` | |
| Supplementary Table S2 (S1S2-Water) | `table_S2/s1s2water/summary_thr030.json` | variants `paper`, `s1_average`, `opera_union`, `s3_pure_*`, `refcut0{25,75}_*` |
| OPERA grid offsets | `table_S2/gswd/opera_grid_offsets.csv`, `table_S2/s1s2water/opera_grid_offsets.csv` | |
| S1S2-Water table (threshold 0.30 primary, 0.50 secondary) | `external_validation/s1s2_water/results/s1s2water_summary.json` | `models.s1_aef_t030.water_iou`, `models.s1_only_t030.water_iou` (0.30); `models.s1_aef.water_iou`, `models.s1_only.water_iou` (0.50); `opera.water_iou`; `mean` and `sd` (sample SD) |

## Values

| Method (threshold 0.30) | Precision | Recall | Dice | Pooled IoU | Per-scene mean [95% CI] |
| --- | ---: | ---: | ---: | ---: | --- |
| S1 only | 0.883 | 0.854 | 0.869 | 0.768 | 0.617 [0.527, 0.702] |
| S1 + AEF (t) | 0.891 | 0.950 | 0.920 | 0.851 | 0.741 [0.669, 0.806] |
| S1 + AEF (t-1) at inference | 0.886 | 0.949 | 0.917 | 0.846 | 0.732 [0.662, 0.797] |
| S1 + AEF (t-1) trained and applied | 0.889 | 0.948 | 0.918 | 0.848 | 0.734 [0.663, 0.800] |
| OPERA DSWx-S1 | 0.859 | 0.851 | 0.855 | 0.747 | 0.591 [0.508, 0.670] |

| Table II configuration | Per-scene IoU, mean +/- sample SD |
| --- | --- |
| k = 0 (S1 only) | 0.624 +/- 0.007 |
| k = 1 | 0.730 +/- 0.006 |
| k = 2 | 0.737 +/- 0.002 |
| k = 3 | 0.741 +/- 0.004 |
| k = 4 | 0.741 +/- 0.002 |
| k = 8 | 0.743 +/- 0.001 |
| k = 16 | 0.741 +/- 0.001 |
| 1,000 training tiles | 0.666 +/- 0.011 |
| 2,000 | 0.740 +/- 0.001 |
| 3,000 | 0.742 +/- 0.002 |
| 4,222 (full) | 0.741 +/- 0.001 |

**Standard deviation convention.** `ablation_10m_summary.json` (`*_std` fields), `table_II_10m.tex`,
`table_II_10m_thr030_common.tex` and `table_II_3m_vs_10m.md` give the population SD (`statistics.pstdev`), which is
what `aggregate_ablations.py` writes. The paper uses the sample SD. The two differ at three decimals for k = 0
(0.006 population, 0.007 sample), k = 1 (0.005, 0.006) and 1,000 tiles (0.009, 0.011). The seed values are stored,
so either SD can be computed from them. `tests/test_paper_results.py` checks the sample SD.

## Not included

- Fig. 1B images. `scripts/eval_10m/make_figure1b_10m.py` renders three versions (`current`, `alt1`, `alt2`), each
  about 4 MB as PDF and 2.9 MB as PNG. They are not committed; `fig1B/panel_iou.csv` holds the values printed on all
  three.
- Probability rasters, 10 m inputs and the exported per-scene PNG layers.
