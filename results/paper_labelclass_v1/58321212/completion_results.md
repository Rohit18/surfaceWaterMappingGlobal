# Manuscript numbers after the label-class reruns

Generated 2026-09-21. Replaces the numbers in
`results/paper_retrain_openwater_v1/57725872/completion_results.md` (2026-09-09), which came from runs
trained on mixed labels (about 65% `dw_binary` "binary50", 35% Dynamic World label class). All runs below
use Dynamic World label class only, with the same chips, splits, seeds and hyperparameters.

Sources:

- Open-water runs `paper_labelclass_v1/openwater/58321212` (5,278 tiles, 4,222 / 1,056 split; widths
  k0-k16 and training sizes, 3 seeds each)
- Prior-year training runs `paper_labelclass_v1/tminus1/58321217` (k16, seeds 42-44)
- OPERA comparator `S1ML/intercomparison_opera/grouped/sid_permanent_water_per_sample_metrics.csv`
  (unchanged)
- Scripts: `scripts/complete_paper_analyses.py --run-set labelclass_v1` (sweep, proximity, matched-mask,
  paired), `scripts/make_figure3_iou_distribution.py --run-set labelclass_v1`,
  `Workspace/code/analyze_s1_vs_s1aef_worldcover_errors.py --threshold 0.30`. `--run-set mixed_v1`
  reproduces the 2026-09-09 values; the new `paired` subcommand reproduces the 09-09 paired table exactly.

---

## 1. Table I (53 scenes, threshold 0.30)

From `paper_metrics.csv` (copied from `S1ML/training_runs/paper_labelclass_v1/openwater/results/58321212/`).

| Method | Precision | Recall | Dice | Pooled IoU | Per-scene mean | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| S1 only | 0.933 (was 0.916) | 0.582 (0.637) | 0.717 (0.752) | **0.558** (0.602) | **0.443** (0.478) | 0.355-0.533 |
| S1+AEF (t-1) | 0.882 (0.909) | 0.946 (0.917) | 0.913 (0.913) | **0.840** (0.840) | **0.741** (0.732) | 0.672-0.803 |
| S1+AEF (t) | 0.892 (0.915) | 0.948 (0.921) | 0.919 (0.918) | **0.851** (0.849) | **0.756** (0.746) | 0.689-0.816 |
| OPERA DSWx-S1 | 0.858 | 0.852 | 0.855 | 0.746 | 0.591 | 0.508-0.670 |

The fused models keep their IoU but move from balanced precision/recall to higher recall and lower
precision. The S1-only control loses 0.044 pooled IoU.

## 2. Threshold sweep (abstract, Section III-A)

| Configuration | Optimal threshold | IoU at optimum | IoU at 0.30 | IoU at 0.50 |
|---|---:|---:|---:|---:|
| S1-only (k0, seed 42) | **0.028** (was 0.0405) | **0.696** (0.684) | 0.558 | 0.490 |
| S1+AEF(t) (k16, seed 42) | 0.58 (0.15) | 0.8545 (0.8500) | 0.8508 | 0.8542 |
| S1+AEF(t-1) swap | 0.58 (0.175) | 0.8450 (0.8410) | 0.8397 | 0.8445 |
| S1+AEF(t-1) trained | 0.58 (0.275) | 0.8460 (0.8398) | 0.8394 | 0.8454 |

S1-only at its optimum: precision 0.838, recall 0.805, Dice 0.821, per-scene mean 0.553.

Manuscript effect:
- Abstract: "0.66 for a S1-only control after threshold optimization" becomes **0.70**.
- Section III-A: "0.663 at its reference-set-optimal threshold of 0.01" becomes **0.696 at 0.028**.
- The fixed 0.30 threshold now costs the fused model 0.0037 IoU against its own optimum (0.8508 vs
  0.8545; was 0.0012). The S1-only control gains 0.138 from tuning (was 0.082). The statement that 0.30 is
  close to optimal for the fused model and favourable to the control still holds.
- New observation: the label-class fused models peak at 0.58, not 0.15-0.28. IoU at 0.50 (0.8542) is
  higher than at 0.30 (0.8508).

## 3. Train/evaluation proximity (Section II-D)

Counts are unchanged because the training chips are unchanged: 5 / 8 / 10 evaluation scenes within
5 / 11 / 20 km, all legacy chips; nearest supplement chip 27.9 km. Scenes within 11 km: SID17, SID21, SID53,
SID64, SID78, SID79, SID98, SID99.

| Configuration | Pooled, 53 | Pooled, 45 | Per-scene, 53 | Per-scene, 45 |
|---|---:|---:|---:|---:|
| S1-only | 0.5585 | 0.5725 | 0.4434 | 0.4493 |
| S1+AEF(t) | 0.8508 | 0.8532 | 0.7558 | 0.7652 |
| S1+AEF(t-1) swap | 0.8397 | 0.8412 | 0.7409 | 0.7496 |
| S1+AEF(t-1) trained | 0.8394 | 0.8411 | 0.7420 | 0.7502 |
| OPERA DSWx-S1 | 0.7461 | 0.7436 | 0.5905 | 0.5947 |

Removing the 8 closest scenes raises every model's score slightly. Proximity does not inflate the results.

## 4. WorldCover error analysis (Section III-B)

Both models at 0.30, k0 and k16 seed 42. Output: `Workspace/reports/worldcover_error_analysis_labelclass_v1/`.

| Quantity | 2026-09-09 | Label class |
|---|---:|---:|
| Scenes improved / unchanged / worse | 42 / 9 / 2 | **44 / 7 / 2** |
| Mean S1-only per-scene IoU | 0.4788 | 0.4440 |
| Mean S1+AEF per-scene IoU | 0.7460 | 0.7558 |
| Mean delta IoU | 0.2672 | **0.3118** |

False negatives corrected by AEF, by WorldCover class: permanent water **89.6%** (was 90.3%), mangroves
4.8% (5.3%), bare/sparse vegetation 1.8% (1.6%), tree cover 1.5% (1.1%).

False positives corrected by AEF: **bare/sparse vegetation 41.5%** (was 34.5%), **grassland 36.2%**
(was 43.7%), tree cover 14.1% (10.9%), cropland 2.8% (5.3%).

Manuscript effect: the order of the two largest false-positive classes is reversed. Bare/sparse vegetation
is now the largest, followed by grassland.

Note: the script previously used each summary's best threshold. For the label-class S1+AEF model the best
of {0.30, 0.50} is 0.50, so `--threshold 0.30` was added to keep the paper's operating point.

## 5. Figure 3

`figures/figure3_iou_distribution.{pdf,png}` and `_values.json`. Per-scene means with bootstrap 95% CI:

| Method | Mean | 95% CI |
|---|---:|---:|
| S1 only | 0.443 | 0.355-0.533 |
| OPERA DSWx-S1 | 0.591 | 0.508-0.670 |
| S1+AEF (t-1), swap | 0.741 | 0.672-0.803 |
| S1+AEF (t-1), trained | 0.742 | 0.673-0.805 |
| S1+AEF (t) | 0.756 | 0.689-0.816 |

## 6. Paired comparisons against OPERA (Section III-E, Discussion, Table I footnote, Fig. 3 caption)

Threshold 0.30, each product on its own valid mask (`paired_vs_opera.json`):

| Comparison | Wins | Losses | Ties | Median delta | Mean delta | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| S1+AEF(t) vs OPERA | **51/53** (was 49/53) | 1 | 1 | **+0.132** (+0.123) | +0.165 | 1.3e-9 |
| S1+AEF(t-1) swap vs OPERA | 48/53 (49/53) | 4 | 1 | +0.123 (+0.102) | +0.150 | 1.6e-8 |
| S1+AEF(t-1) trained vs OPERA | 48/53 (48/53) | 4 | 1 | +0.102 (+0.088) | +0.152 | 1.2e-8 |
| S1-only vs OPERA | 18/53 (21/53) | 33 | 2 | -0.092 (-0.029) | -0.147 | 5.8e-4 |

Matched valid mask (`matched_mask.json`; 51,897,488 common pixels, 93.4%): pooled / per-scene IoU
S1-only 0.567 / 0.449, S1+AEF(t) 0.853 / 0.758, t-1 swap 0.842 / 0.743, t-1 trained 0.842 / 0.744, OPERA
0.747 / 0.591. S1+AEF(t) beats OPERA on 51/53 (median +0.132, p = 1.25e-9). All values are within 0.01 of
the native-mask values.

## 7. Ablations (Section III-C, Table II)

Per-scene mean IoU at 0.50, mean of 3 seeds (`ablation_metrics.csv`):

| AEF width k | 0 | 1 | 2 | 3 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Label class | 0.411 | 0.732 | 0.727 | 0.752 | 0.751 | 0.758 | 0.757 |
| 2026-09-09 | 0.440 | 0.710 | 0.706 | 0.727 | 0.731 | 0.740 | 0.744 |

| Training tiles | 1000 | 2000 | 3000 | 4222 |
|---|---:|---:|---:|---:|
| Label class | 0.592 | 0.751 | 0.752 | 0.757 |
| 2026-09-09 | 0.602 | 0.733 | 0.737 | 0.744 |

The width curve is still non-monotone: k=1 reaches 0.732 of 0.757 at k=16, k=2 is lower than k=1, and the
curve is flat from k=3 (k=8 is 0.0005 above k=16). Performance is flat from 2,000 tiles.

## 8. Manuscript edits that follow

1. Table I: all model rows (Section 1).
2. Abstract and Section III-A: S1-only optimum 0.70 / 0.696 at 0.028 (Section 2).
3. Section III-E, Discussion, Table I footnote, Fig. 3 caption: 51/53, median +0.13 (Section 6).
4. Section III-B: swap bare/sparse vegetation and grassland in the false-positive sentence; 44/7/2 (Section 4).
5. Section III-C and Table II: new width and training-size values (Section 7).
6. Section II-D: proximity exclusion values (Section 3).
7. Methods: the training labels are Dynamic World label class only (all rows).
8. Figure 3: replace with `figures/figure3_iou_distribution.pdf`.

## 9. Sentence-level replacements against `manuscripts/prism_upload_surface_water_2026-08-12/main.tex`

That file is the 12 August version: it predates the 2026-09-09 corrections and quotes an older run set
in Section III-C. Values below are for the label-class runs. Line numbers refer to that file.

| Line | Text now | Replace with |
|---:|---|---|
| 73 (abstract) | S1-only "0.66 ... after threshold optimization" | **0.70** |
| 73 (abstract) | fused 0.85, OPERA 0.75 | unchanged (0.851, 0.746) |
| 73 (abstract) | previous-year embedding "lowers IoU by only 0.024" | **0.011** |
| 158 (II-A) | 4679 triplets, 3743 / 936 | **5278, 4222 / 1056** (supplemented set) |
| 158 (II-A) | "class 0, water -> 1" from the DW image | state that the target is the DW `label` band class water (label class only) |
| 192 (Table I) | S1 only 0.901 / 0.629 / 0.741 / 0.588 / 0.481 [0.394, 0.565] | **0.933 / 0.582 / 0.717 / 0.558 / 0.443 [0.355, 0.533]** |
| 193 (Table I) | t-1 0.898 / 0.913 / 0.905 / 0.827 / 0.726 [0.654, 0.792] | **0.882 / 0.946 / 0.913 / 0.840 / 0.741 [0.672, 0.803]** |
| 194 (Table I) | t 0.909 / 0.931 / 0.920 / 0.851 / 0.753 [0.683, 0.815] | **0.892 / 0.948 / 0.919 / 0.851 / 0.756 [0.689, 0.816]** |
| 196 (Table I) | OPERA per-scene 0.527 [0.441, 0.612] | **0.591 [0.508, 0.670]** |
| 201 (footnote) | t > S1-only 48/53, > OPERA 51/53; t-1 > S1-only 46/53 | **50/53, 51/53; 50/53** (all p < 1e-8) |
| 205 (III-A) | S1-only 0.588 at 0.30, 0.663 at optimal 0.01, recall 0.629 | **0.558 at 0.30, 0.696 at 0.028, recall 0.582** |
| 208 (III-A) | 0.588 -> 0.851, gain +0.263 | **0.558 -> 0.851, +0.292** |
| 208 (III-A) | recall 0.629 -> 0.931; precision "essentially unchanged (0.901 to 0.909)" | **recall 0.582 -> 0.948; precision falls 0.933 -> 0.892**; the "unchanged precision" wording no longer holds |
| 208 (III-A) | 48 of 53 scenes, median +0.24 | **50 of 53, median +0.32** |
| 208 (III-A) | per-scene 0.48 (0.39-0.57) -> 0.75 (0.68-0.82) | **0.44 (0.35-0.53) -> 0.76 (0.69-0.82)** |
| 215 (Fig. 3 caption) | 48/53 and 51/53 | **50/53 and 51/53** |
| 220 (III-C) | k=0 0.389; k=1 0.489; "increases at each tested width" to 0.707 at k=16; k=3 0.665 | **0.411; 0.732; non-monotone (k=2 0.727 < k=1), flat from k=3 0.752; k=16 0.757** |
| 220 (III-C) | recall 0.647 (k=1) -> 0.875 (k=16) | **0.907 -> 0.945** |
| 220 (III-C) | 1000 tiles 0.476; plateau by 3000 (0.699); full 0.701 | **0.592; plateau by 2000 (0.751); full 0.757** |
| 220 (III-C) | seed variance +/-0.088 (S1-only) -> +/-0.004 (k=16) | **+/-0.067 -> +/-0.001** |
| 229-240 (Table II) | all rows | Section 7 values; std: k0 0.067/0.104, k1 0.012/0.013, k2 0.009/0.013, k3 0.005/0.004, k4 0.001/0.002, k8 0.002/0.002, k16 0.001/0.004; tiles 1000 0.036/0.037, 2000 0.006/0.008, 3000 0.003/0.007, 4222 0.001/0.004 (IoU std / recall std) |
| 240 (Table II) | "3743 (full training split)" | **4222** |
| 245 (Table II note) | panels (a) and (b) are independent run sets (0.707, 0.701); Table I model 0.753 | drop the independence claim (same runs, 0.757); Table I model **0.756** |
| 253 (III-D) | pooled 0.851 -> 0.827 (2.8%), retains 91% | **0.851 -> 0.840 (1.3%), retains 96%** |
| 253 (III-D) | per-scene cost mean 0.026, median 0.004, p<0.01 | **mean 0.015, median 0.003, p = 0.02** |
| 253 (III-D) | swap improves on S1-only by mean 0.25 on 46/53 | **mean 0.30 on 50/53** |
| 256 (III-E) | 0.746 -> 0.851 (+0.105) | unchanged (+0.105) |
| 256 (III-E) | 51/53, median +0.14; 0.75 (0.68-0.82) vs 0.53 (0.44-0.61) | **51/53, median +0.13; 0.76 (0.69-0.82) vs 0.59 (0.51-0.67)** |
| 261 (Discussion) | per-scene IoU 0.48 -> 0.75; 51/53; prior-year retains roughly 90% | **0.44 -> 0.76; 51/53; retains about 95%** (per-scene 95%, pooled 96%) |

Per-scene paired statistics for line 253 are computed from the `*_paper030` (t) and `*_tminus1_tta` (swap)
per-sample metrics at threshold 0.30.
