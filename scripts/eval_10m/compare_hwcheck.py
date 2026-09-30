#!/usr/bin/env python3
"""Hardware control: 10 m probabilities of k0 and k16 seed 42 on A100-80GB (this run) vs A100-40GB (29 Sep, job 59050165).

Compares the native 10 m rasters pixel by pixel (max |difference|, identical fraction) and the pooled / per-scene IoU
on the common mask at 0.30 and 0.50 after the same bilinear resampling (score_10m.resample_to_label).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoring as S  # noqa: E402
from paths import env_root  # noqa: E402  (environment variables; see README.md)
import score_10m as T  # noqa: E402
import score_ablations as SA  # noqa: E402

OUT = env_root("SWM_ABLATION10M")
PREV = env_root("SWM_EVAL10M")
TAGS = ["width_k0_seed42_current_tta", "width_k16_seed42_current_tta"]


def main() -> int:
    res = {}
    for tag in TAGS:
        a_root = PREV / "predictions_10m/openwater" / tag / "probabilities"
        b_root = OUT / "predictions_10m_hwcheck/openwater" / tag / "probabilities"
        maxdiff, n_ident, n_tot = 0.0, 0, 0
        cnt = {t: {"40GB": np.zeros(4, np.int64), "80GB": np.zeros(4, np.int64)} for t in (0.30, 0.50)}
        scene = {t: {"40GB": [], "80GB": []} for t in (0.30, 0.50)}
        for fa in sorted(a_root.glob("*/*_prob.tif")):
            sid, date = fa.name.replace("_prob.tif", ""), fa.parent.name
            fb = b_root / date / fa.name
            a, b = rasterio.open(fa).read(1), rasterio.open(fb).read(1)
            va, vb = a != T.NODATA, b != T.NODATA
            assert np.array_equal(va, vb), sid
            d = np.abs(a[va].astype(np.float64) - b[va])
            maxdiff = max(maxdiff, float(d.max()))
            n_ident += int((d == 0).sum()); n_tot += int(va.sum())
            label, lvalid = S.read_label(sid)
            common = SA.common_mask(sid, date, lvalid)
            pa = S.read_prob(str(PREV / "predictions_10m_on_3m/openwater" / tag / "probabilities" / date / fa.name))[0]
            pb = T.resample_to_label(fb, sid, OUT / "predictions_10m_hwcheck_on_3m/openwater" / tag / "probabilities" / date / fa.name)
            for t in cnt:
                for k, p in (("40GB", pa), ("80GB", pb)):
                    c = S.counts(p[common] >= t, (label == 1)[common])
                    cnt[t][k] += c; scene[t][k].append(S.iou(c))
        r = {"native_10m_max_abs_diff": maxdiff, "native_10m_identical_fraction": n_ident / n_tot}
        for t in cnt:
            r[f"thr{t:.2f}"] = {k: {"pooled_iou": S.iou(cnt[t][k]), "per_scene_mean": float(np.mean(scene[t][k]))}
                                for k in cnt[t]}
            r[f"thr{t:.2f}"]["max_scene_abs_diff"] = float(np.max(np.abs(np.array(scene[t]["40GB"]) - np.array(scene[t]["80GB"]))))
        res[tag] = r
    (OUT / "ablations_10m/hardware_control.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
