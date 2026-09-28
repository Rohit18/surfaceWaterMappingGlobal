# Results needed to complete the surface water manuscript

Generated 2026-09-09. Closes the five outstanding analyses identified against
`manuscripts/prism_upload_surface_water_2026-08-12/main.tex`.

Sources:

- Supplemented run `paper_retrain_openwater_v1/57725872` (5,278 tiles, 4,222 / 1,056 split)
- Prior-year training run `paper_tminus1_v1/57952761` (seeds 42, 43, 44)
- OPERA comparator `S1ML/intercomparison_opera/grouped/sid_permanent_water_per_sample_metrics.csv`
- Audit outputs under `outputs/audit/`

---

## 1. Threshold sweep — resolves the blocking abstract number

Computed directly from the saved probability rasters by histogram accumulation
(1e-5 bin resolution). The method reproduces every published 0.30 and 0.50
confusion count exactly, so the sweep is on the same footing as the evaluation
pipeline.

| Configuration | Optimal threshold | IoU at optimum | IoU at 0.01 | IoU at 0.30 |
|---|---:|---:|---:|---:|
| S1-only (k0, seed 42) | **0.0405** | **0.6844** | 0.6528 | 0.6021 |
| S1+AEF(t) (k16, seed 42) | 0.1500 | 0.8500 | 0.8365 | 0.8488 |
| S1+AEF(t−1) swap | 0.1750 | 0.8410 | 0.8260 | 0.8400 |
| S1+AEF(t−1) trained | 0.2750 | 0.8398 | 0.8132 | 0.8397 |

At the optimum the S1-only control reaches precision 0.830, recall 0.796,
Dice 0.813; its per-scene mean IoU is 0.5457.

**Manuscript effect.** The abstract's "0.66 for a S1-only control after threshold
optimization" becomes **0.68**, and §III-A's "0.663 at its reference-set-optimal
threshold of 0.01" becomes **0.684 at 0.0405**.

**Worth adding.** The preselected 0.30 threshold costs the fused model only
0.0012 IoU against its own optimum (0.8488 vs 0.8500), whereas the S1-only
control gains 0.082 from tuning. The fixed operating point is therefore near
optimal for the model being proposed and generous to the control — a useful
reply to any reviewer who suspects the threshold was chosen to flatter the
fused model.

---

## 2. Train/evaluation proximity — §II-D

Great-circle distance from each of the 53 evaluation scene centroids to the
nearest of the 5,278 training centroids.

| Distance | Scenes | From legacy | From supplement |
|---|---:|---:|---:|
| ≤ 5 km | 5 | 5 | 0 |
| ≤ 11 km | 8 | 8 | 0 |
| ≤ 20 km | 10 | 10 | 0 |

This reproduces the manuscript's counts exactly. Scenes within 11 km: SID17,
SID21, SID53, SID64, SID78, SID79, SID98, SID99. **No supplement sample lies
near any evaluation scene** — the closest is 27.9 km, and none falls inside the
20 km band — so the 600 new samples add no leakage exposure.

Excluding the eight proximate scenes:

| Configuration | Pooled, 53 | Pooled, 45 | Δ | Per-scene, 53 | Per-scene, 45 |
|---|---:|---:|---:|---:|---:|
| S1-only | 0.6021 | 0.6157 | +0.0136 | 0.4777 | 0.4830 |
| S1+AEF(t) | 0.8488 | **0.8492** | **+0.0004** | 0.7460 | 0.7541 |
| S1+AEF(t−1) swap | 0.8400 | 0.8400 | −0.0000 | 0.7320 | 0.7399 |
| S1+AEF(t−1) trained | 0.8397 | 0.8403 | +0.0005 | 0.7357 | 0.7439 |
| OPERA DSWx-S1 | 0.7461 | 0.7436 | −0.0025 | 0.5905 | 0.5947 |

**Manuscript effect.** The §II-D sentence keeps its structure and conclusion:
0.851 → 0.852 becomes **0.8488 → 0.8492**. Add that the supplement was screened
and contributes no sample within 27 km of an evaluation scene.

---

## 3. WorldCover error anatomy — §III-B

Re-run with `analyze_s1_vs_s1aef_worldcover_errors.py` against the supplemented
k=16 model and its matched S1-only control, at the paper's own 0.30 threshold.
Output: `Workspace/reports/worldcover_error_analysis_openwater_v1/`.

The previous version used train-3000, seed-42 models at threshold 0.01, which
matched nothing else in the paper. This version uses the headline model at the
headline threshold, so the caveat about a reduced-data variant can be dropped.

| Quantity | Old (train-3000, thr 0.01) | New (full, thr 0.30) |
|---|---:|---:|
| Scenes improved / unchanged / worse | 39 / 13 / 1 | **42 / 9 / 2** |
| Mean S1-only per-scene IoU | 0.5426 | 0.4788 |
| Mean S1+AEF per-scene IoU | 0.7481 | 0.7460 |
| Mean ΔIoU | 0.2055 | 0.2672 |

Corrected **false negatives** by WorldCover class:

| Class | Old | New |
|---|---:|---:|
| Permanent water | 80.2% | **90.3%** |
| Mangroves | 10.1% | 5.3% |
| Bare / sparse vegetation | — | 1.6% |
| Tree cover | — | 1.1% |

Corrected **false positives** by WorldCover class:

| Class | Old | New |
|---|---:|---:|
| Grassland | 51% | **43.7%** |
| Bare / sparse vegetation | 32% | 34.5% |
| Tree cover | — | 10.9% |
| Cropland | 9% | 5.3% |

Note: this analysis intersects the S1-only and fused valid masks, so its
S1-only mean (0.4788) differs marginally from the standalone evaluation
(0.4777), which uses the larger S1-only valid mask.

---

## 4. Figure 3 regenerated

The original figure script was not kept with the manuscript package. It has been
rewritten as `scripts/make_figure3_iou_distribution.py` and now includes the
trained prior-year configuration as a fifth row.

Output: `results/paper_retrain_openwater_v1/57725872/figures/figure3_iou_distribution.{pdf,png}`
plus `_values.json`.

| Row | Per-scene mean | Bootstrap 95% CI |
|---|---:|---|
| S1 only | 0.4777 | [0.3882, 0.5664] |
| OPERA DSWx-S1 | 0.5905 | [0.5089, 0.6675] |
| S1 + AEF(t−1), swap | 0.7320 | [0.6609, 0.7977] |
| S1 + AEF(t−1), trained | 0.7357 | [0.6642, 0.8017] |
| S1 + AEF(t) | 0.7460 | [0.6738, 0.8110] |

Figure 1 panel B keeps its scene selection: SID23, SID46, SID53 and SID100 are
still the four largest-gain scenes. Only the printed IoU labels change
(0.884, 0.821, 0.842, 0.808 for the fused model).

---

## 5. OPERA comparator — recomputed

The manuscript printed OPERA per-scene 0.527 [0.441, 0.612]. That value could
not be reproduced from any surviving artifact, and it was already present in the
first commit of `results/paper_metrics.csv` (20 Aug 2026). Candidates checked,
all on the same 53 scenes:

| Source | Per-scene mean |
|---|---:|
| `intercomparison_opera` per-sample, full valid mask | **0.5905** |
| `intercomparison_opera_bilinear`, full valid mask | 0.5941 |
| `permanent_water_..._opera_figures/figure_manifest.csv`, restricted mask | 0.6832 |
| Combined 53 SID + 60 post-flood scenes | 0.5061 |
| Manuscript value | 0.527 |

The pooled OPERA row always reconciled exactly with the first source, so only
the per-scene column was unaccounted for.

`scripts/aggregate_paper_retrain.py` previously copied the OPERA row forward
verbatim. It now loads the OPERA summary and per-sample metrics through the same
`paper_row()` path as the model rows, so all four rows of Table I — including
the bootstrap CI, which uses the script's own RNG convention — come from one
code path. The recomputed row:

| Field | Old | New |
|---|---:|---:|
| Precision | 0.858 | 0.857628 |
| Recall | 0.852 | 0.851590 |
| Dice | 0.855 | 0.854598 |
| Pooled water IoU | 0.746 | 0.746113 |
| **Per-scene mean** | **0.527** | **0.590528** |
| **95% CI** | **[0.441, 0.612]** | **[0.508065, 0.669592]** |

Pooled values are unchanged to three decimals, confirming that only the
per-scene column was ever wrong.

### Paired comparisons against OPERA, threshold 0.30

| Comparison | Wins | Losses | Ties | Median Δ | Mean Δ | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| S1+AEF(t) vs OPERA | **49/53** | 2 | 2 | +0.1228 | +0.1555 | 1.9 × 10⁻⁹ |
| S1+AEF(t−1) swap vs OPERA | 49/53 | 3 | 1 | +0.1020 | +0.1415 | 1.6 × 10⁻⁸ |
| S1+AEF(t−1) trained vs OPERA | 48/53 | 4 | 1 | +0.0878 | +0.1451 | 1.6 × 10⁻⁸ |
| S1-only vs OPERA | 21/53 | 31 | 1 | −0.0290 | −0.1128 | 0.0048 |

**Manuscript effect.** §III-E, the Discussion, Table I's footnote and the Fig. 3
caption change from **51/53 to 49/53**, median **+0.14 → +0.123**, and OPERA's
per-scene mean from **0.53 (0.44–0.61) → 0.59 (0.51–0.67)**.

**One narrative consequence.** At 0.527 the OPERA comparator sat close to the
S1-only control (0.478). At 0.591 it is clearly above it — OPERA now beats the
S1-only control on 31 of 53 scenes. Nothing in the paper claims otherwise, but
Fig. 3 now shows OPERA sitting distinctly between the control and the fused
configurations, and the Discussion's point about OPERA's hand-specified static
layers becomes sharper: those priors already recover much of what SAR alone
misses, and the annual embedding recovers more without any thresholded
categorical mask.

### Matched valid-mask check

Table I as published lets each row use its own no-data mask, so the rows do not
cover identical pixel sets: S1-only 55,127,752 pixels, S1+AEF 54,375,145, OPERA
52,651,510. Scoring every method on the intersection instead (51,897,488 pixels,
93.4% of the full grid) moves nothing materially:

| Method | Pooled, native | Pooled, matched | Per-scene, native | Per-scene, matched |
|---|---:|---:|---:|---:|
| S1-only | 0.6021 | 0.6112 | 0.4777 | 0.4852 |
| S1+AEF(t) | 0.8488 | 0.8510 | 0.7460 | 0.7479 |
| S1+AEF(t−1) swap | 0.8400 | 0.8423 | 0.7320 | 0.7338 |
| S1+AEF(t−1) trained | 0.8397 | 0.8419 | 0.7357 | 0.7375 |
| OPERA DSWx-S1 | 0.7461 | 0.7468 | 0.5905 | 0.5912 |

On the matched mask S1+AEF(t) beats OPERA on 50/53 scenes (median +0.1228,
p = 1.3 × 10⁻⁹) rather than 49/53. Every value shifts by at most 0.01, so the
differing native grids do not drive any conclusion. This is worth one sentence
in §II-D — it answers the "products have differing native grids" limitation the
Discussion currently concedes without evidence.

Reproduce with `python scripts/complete_paper_analyses.py matched-mask --out ...`.

---

## Remaining before submission

All computation is complete. What follows is writing.

1. Propagate the OPERA recomputation: 51/53 → 49/53, median +0.14 → +0.123,
   per-scene 0.53 (0.44–0.61) → 0.59 (0.51–0.67), in §III-E, the Discussion,
   Table I's footnote and the Fig. 3 caption.
2. Rewrite the §II-A pairing claim against the timing audit — median 17.0 h,
   max 112.4 h, 71.8% within 24 h for the legacy set; all 600 supplement pairs
   within 24 h. The Fig. 1 caption repeats the claim.
3. Rewrite §III-C for the non-monotone width curve — k=1 now reaches 0.710 of
   0.744 at k=16 — and drop the Table II footnote claiming panels (a) and (b)
   are independent run sets; they now resolve to identical runs.
4. Describe the supplement's second sampling frame in §II-A and soften the
   Discussion's "centered on SWORD nodes along mapped river centerlines".
5. Correct the dataset count: 4,678 / 3,742 / 936 legacy, 5,278 / 4,222 / 1,056
   supplemented. The manuscript's 4,679 / 3,743 is off by one.
6. Add configuration (4) to §II-C and the trained-t−1 paragraph to §III-D;
   delete the "may include train/test shift" hedge.
7. Funding statement is still `[TODO: funding]` in `main.tex`.
