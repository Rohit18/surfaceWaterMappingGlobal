#!/usr/bin/env python3
"""Per-sample CSVs for analyze_s1_vs_s1aef_worldcover_errors.py at 10 m inference (Task B4).

Copies of the paper's inputs (reports/58321212/evaluation/width_k{0,16}_seed42_current_tta_paper030/*_per_sample_metrics.csv)
with prob_path replaced by the 10 m-inference probability resampled to the label grid
(revision_checks_20260929/predictions_10m_on_3m/openwater/<tag>/probabilities/<date>/<SID>_prob.tif). The analysis
script reads only sample_id, date, group, label_path and prob_path; the metric columns are left as they were (3 m) and
are not used.
"""
from pathlib import Path
import pandas as pd

from paths import env_root  # noqa: E402  (environment variables; see README.md)

E = env_root("SWM_S1ML") / "training_runs/paper_labelclass_v1/openwater/reports/58321212/evaluation"
R3 = env_root("SWM_EVAL10M") / "predictions_10m_on_3m/openwater"
OUT = env_root("SWM_ABLATION10M") / "worldcover_10m/inputs"
for tag in ("width_k0_seed42_current_tta", "width_k16_seed42_current_tta"):
    d = pd.read_csv(E / f"{tag}_paper030" / f"{tag}_paper030_per_sample_metrics.csv")
    d["prob_path"] = [str(R3 / tag / "probabilities" / r.date / f"{r.sample_id}_prob.tif") for r in d.itertuples()]
    assert all(Path(p).exists() for p in d.prob_path)
    d.to_csv(OUT / f"{tag}_10m_per_sample_inputs.csv", index=False)
    print(tag, len(d))
