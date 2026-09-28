# Paper values after open-water supplementation

Dataset: 5,278 tiles (4,222 train / 1,056 validation), including 600 new open-water samples.

## Main independent evaluation (53 scenes, threshold 0.30)

| Method | Precision | Recall | Dice | Pooled water IoU | Mean scene IoU | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| S1 only | 0.916 | 0.637 | 0.752 | 0.602 | 0.478 | 0.388–0.566 |
| S1+AEF (t-1) | 0.909 | 0.917 | 0.913 | 0.840 | 0.732 | 0.660–0.798 |
| S1+AEF (t) | 0.915 | 0.921 | 0.918 | 0.849 | 0.746 | 0.676–0.810 |
| OPERA DSWx-S1 | 0.858 | 0.852 | 0.855 | 0.746 | 0.591 | 0.508–0.670 |

See `paper_metrics.csv`, `ablation_metrics.csv`, and `aggregation_audit.json` for exact values and provenance.
