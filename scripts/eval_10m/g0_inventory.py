#!/usr/bin/env python3
"""G0: native 30 m OPERA B02_BWTR products of the 53 GSWD scenes, their grids relative to the label grid, and a
rebuild check of the paper's 3 m OPERA masks.

Products per scene: the `opera_files` column of intercomparison_opera/grouped/sid_permanent_water_per_sample_metrics.csv
(the file the paper's OPERA row was computed from). For each product: CRS, transform, whether the CRS equals the
label CRS, and the offset of the product's grid origin from the label grid origin modulo 30 m (x: product left -
label left; y: label top - product top; both taken modulo 30 m and also modulo the block size 10 x label pixel).
Rebuild check: evaluate_intercomparison_opera_binary.merged_binary_opera (unchanged copy in scripts/orig/; nearest
WarpedVRT onto the label grid, valid = {0,1}, union over products) must equal binary_masks/<SID>_opera_bwtr_binary.tif.
Output: gswd/opera_grid_offsets.csv, gswd/opera_products_per_scene.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import rasterio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # scoring, score_10m and evaluate_intercomparison_opera_binary
import evaluate_intercomparison_opera_binary as EOB  # noqa: E402
import scoring as S  # noqa: E402
from paths import env_root  # noqa: E402  (environment variables; see README.md)

OUT = env_root("SWM_RESOLUTION") / "gswd"
GROUPED = env_root("SWM_S1ML") / "intercomparison_opera/grouped/sid_permanent_water_per_sample_metrics.csv"


def fmod(a: float, m: float) -> float:
    r = a % m
    return 0.0 if min(r, m - r) < 1e-6 else r


def main() -> int:
    g = pd.read_csv(GROUPED)
    rows, prod_rows = [], []
    for r in g.sort_values("sample_id", key=lambda s: s.str[3:].astype(int)).itertuples():
        sid = r.sample_id
        with rasterio.open(S.LABELS.format(sid)) as lab:
            lcrs, lt, lw, lh = lab.crs, lab.transform, lab.width, lab.height
        px = lt.a
        block, block_y = 10 * lt.a, 10 * abs(lt.e)  # SID21: 3.0293 x 3.0117 m pixels
        paths = [Path(p) for p in str(r.opera_files).split(";")]
        grids = set()
        for p in paths:
            with rasterio.open(p) as src:
                t, crs = src.transform, src.crs
                dx, dy = t.c - lt.c, lt.f - t.f
                same = crs == lcrs
                rec = {"sample_id": sid, "product": p.name, "path": str(p), "opera_crs": crs.to_string(),
                       "opera_transform": [t.a, t.b, t.c, t.d, t.e, t.f], "opera_width": src.width,
                       "opera_height": src.height, "label_crs": lcrs.to_string(),
                       "label_transform": [lt.a, lt.b, lt.c, lt.d, lt.e, lt.f], "label_pixel_m": px, "label_pixel_y_m": abs(lt.e),
                       "same_crs": bool(same),
                       "offset_x_mod30_m": fmod(dx, 30.0) if same else None,
                       "offset_y_mod30_m": fmod(dy, 30.0) if same else None,
                       "offset_x_mod_block_m": fmod(dx, block) if same else None,
                       "offset_y_mod_block_m": fmod(dy, block_y) if same else None}
                prod_rows.append(rec)
                grids.add((crs.to_string(), round(t.a, 6), round(t.c % 30, 6), round(t.f % 30, 6)))
        label = SimpleNamespace(crs=lcrs, transform=lt, width=lw, height=lh)
        rebuilt = EOB.merged_binary_opera(label, [SimpleNamespace(path=p) for p in paths], "nearest", 0.5)
        with rasterio.open(EOB.Path(S.OPERA_MASK.format(sid))) as src:
            existing = src.read(1)
        same_mask = bool(np.array_equal(rebuilt, existing))
        rows.append({"sample_id": sid, "n_products": len(paths), "products": ";".join(p.name for p in paths),
                     "all_same_crs_as_label": all(x["same_crs"] for x in prod_rows if x["sample_id"] == sid),
                     "single_30m_grid": len(grids) == 1, "label_pixel_m": px,
                     "offset_x_mod30_m": ";".join(str(x["offset_x_mod30_m"]) for x in prod_rows if x["sample_id"] == sid),
                     "offset_y_mod30_m": ";".join(str(x["offset_y_mod30_m"]) for x in prod_rows if x["sample_id"] == sid),
                     "rebuilt_3m_mask_equals_binary_masks": same_mask})
        print(sid, len(paths), rows[-1]["all_same_crs_as_label"], rows[-1]["single_30m_grid"],
              rows[-1]["offset_x_mod30_m"], rows[-1]["offset_y_mod30_m"], same_mask, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "opera_grid_offsets.csv", index=False)
    pd.DataFrame(prod_rows).to_csv(OUT / "opera_products_per_scene.csv", index=False)
    d = pd.DataFrame(rows)
    print("scenes", len(d), "rebuild equal", int(d.rebuilt_3m_mask_equals_binary_masks.sum()),
          "all same CRS", int(d.all_same_crs_as_label.sum()), "single grid", int(d.single_30m_grid.sum()),
          "products", int(d.n_products.sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
