#!/usr/bin/env python3
"""S1S2-Water (15 scenes): per-scene confusion counts for every scoring variant (Part S). CPU only, no inference.

Unchanged code reused (copies in scripts/orig/): evaluate_intercomparison.py (SCENES, opera_paths, opera_prediction:
nearest warp of each corrected B02_BWTR product onto the frozen grid, covered pixels of later products overwrite
earlier ones, valid = {0,1}); s1s2water_supplement.py (reference aggregation and model probability reading are
repeated here line for line: Resampling.average of valid and mask x valid, reference-valid >= 50% of area, water at
fraction >= 0.5; model = native 10 m probability -> bilinear, valid = finite and in [0, 1]).

Variants (common mask = reference-valid AND OPERA-valid AND all six runs valid, per variant):
  paper            frozen 30 m grid, models bilinear (the paper; must reproduce s1s2water_per_scene_all_methods.csv)
  s1_average       models Resampling.average from native 10 m, otherwise as paper
  opera_union      as paper, OPERA products combined by union (valid in any, water in any) instead of the
                   paper's overwrite order
  s3_pure_paper / s3_pure_average   restricted to reference pixels whose valid-area water fraction is exactly 0 or 1
  refcut025_paper / refcut075_paper / refcut025_average / refcut075_average
                   reference water at fraction >= 0.25 / >= 0.75 (DESIGN.md sensitivity; method thresholds unchanged)
  s2_native@<dx>_<dy> / s2_frozen@<dx>_<dy>   exploratory grid-alignment check (see RESULTS.md): for each class of
                   OPERA products in the reference CRS with grid offset (dx, dy) from the frozen grid, the same
                   products scored (a) on their own grid (reference and models area-averaged, OPERA not resampled) and
                   (b) warped by nearest neighbour to the frozen grid (models averaged); union within the class.
Output: s1s2water/per_scene_counts_long.csv, s1s2water/per_scene_support.csv, s1s2water/opera_grid_offsets.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.windows import Window

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "external_validation/s1s2_water/evaluation"))
import evaluate_intercomparison as EI  # noqa: E402
from paths import env_root  # noqa: E402  (environment variables; see README.md)

ROOT = env_root("SWM_S1S2")
EI.ROOT = ROOT  # the repository copy would otherwise resolve ROOT from its own location
OUT = env_root("SWM_RESOLUTION") / "s1s2water"
SEEDS = (42, 43, 44)
RUNS = {("S1-only", s): f"lc_k0_seed{s}_tta" for s in SEEDS} | {("S1+AEF", s): f"lc_k16_seed{s}_tta" for s in SEEDS}
THRESHOLDS = (0.30, 0.50)


def prob_path(scene, tag):
    paths = list((ROOT / f"work/model/scene_{scene}/inference_inputs_gee10m" / tag / "probabilities").glob("*/*_prob.tif"))
    if len(paths) != 1:
        raise RuntimeError(f"scene {scene} {tag}: {paths}")
    return paths[0]


def probability(path, shape, transform, crs, resampling):
    """s1s2water_supplement.probability with the resampling as a parameter."""
    prob = np.full(shape, -9999.0, dtype="float32")
    with rasterio.open(path) as source:
        reproject(rasterio.band(source, 1), prob, src_transform=source.transform, src_crs=source.crs,
                  src_nodata=source.nodata, dst_transform=transform, dst_crs=crs, dst_nodata=-9999.0,
                  resampling=resampling)
    valid = np.isfinite(prob) & (prob >= 0) & (prob <= 1)
    return prob, valid


def reference(grid, shape, transform, crs):
    """s1s2water_supplement.process_scene reference aggregation, returning the valid fraction and water fraction."""
    valid_fraction = np.zeros(shape, dtype="float32")
    water_valid_fraction = np.zeros(shape, dtype="float32")
    with rasterio.open(grid["reference_valid"]) as source:
        native_valid = source.read(1).astype("float32")
        reproject(native_valid, valid_fraction, src_transform=source.transform, src_crs=source.crs,
                  dst_transform=transform, dst_crs=crs, resampling=Resampling.average)
    with rasterio.open(grid["reference_mask"]) as source:
        reproject(source.read(1).astype("float32") * native_valid, water_valid_fraction,
                  src_transform=source.transform, src_crs=source.crs, dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.average)
    del native_valid
    reference_valid = valid_fraction >= float(grid["reference_valid_fraction_min"])
    reference_fraction = np.divide(water_valid_fraction, valid_fraction, out=np.zeros_like(valid_fraction),
                                   where=valid_fraction > 0)
    return reference_valid, reference_fraction


def counts(pred, ref, common):
    return (int(np.count_nonzero(common & pred & ref)), int(np.count_nonzero(common & pred & ~ref)),
            int(np.count_nonzero(common & ~pred & ref)), int(np.count_nonzero(common & ~pred & ~ref)))


def score(rows, variant, scene, common, ref, probs, opera):
    for thr in THRESHOLDS:
        for (name, seed), p in probs.items():
            tp, fp, fn, tn = counts(p >= thr, ref, common)
            rows.append(dict(variant=variant, method=name, seed=seed, threshold=thr, sample_id=scene, tp=tp, fp=fp, fn=fn, tn=tn))
        tp, fp, fn, tn = counts(opera, ref, common)
        rows.append(dict(variant=variant, method="OPERA", seed="", threshold=thr, sample_id=scene, tp=tp, fp=fp, fn=fn, tn=tn))


def union_on(paths, shape, transform, crs):
    water = np.zeros(shape, bool); valid = np.zeros(shape, bool)
    for path in paths:
        warped = np.full(shape, 255, dtype="uint8")
        with rasterio.open(path) as source:
            reproject(rasterio.band(source, 1), warped, src_transform=source.transform, src_crs=source.crs,
                      src_nodata=source.nodata, dst_transform=transform, dst_crs=crs, dst_nodata=255,
                      resampling=Resampling.nearest)
        this = np.isin(warped, (0, 1))
        valid |= this; water |= (warped == 1) & this
    return water, valid


def scene_job(args):
    scene, grid = args
    rows, sup, offs = [], {"scene_id": scene}, []
    shape = (int(grid["height"]), int(grid["width"]))
    transform = from_origin(float(grid["left"]), float(grid["top"]), 30.0, 30.0)
    crs = grid["crs"]
    rv, frac = reference(grid, shape, transform, crs)
    ref = frac >= 0.5
    opera, opera_valid, _ = EI.opera_prediction(scene, shape, transform, crs)
    ou, ouv = union_on(EI.opera_paths(scene), shape, transform, crs)
    both = opera_valid & ouv
    sup.update(reference_valid=int(rv.sum()), opera_valid=int(opera_valid.sum()), opera_union_valid=int(ouv.sum()),
               opera_union_vs_paper_valid_differs=int((opera_valid != ouv).sum()),
               opera_union_vs_paper_water_differs_where_both_valid=int((both & (opera != ou)).sum()))
    pb, vb, pa, va = {}, {}, {}, {}
    for k, tag in RUNS.items():
        path = prob_path(scene, tag)
        pb[k], vb[k] = probability(path, shape, transform, crs, Resampling.bilinear)
        pa[k], va[k] = probability(path, shape, transform, crs, Resampling.average)
    allb = np.logical_and.reduce(list(vb.values())); alla = np.logical_and.reduce(list(va.values()))
    pure = rv & ((frac == 0) | (frac == 1))
    cp = rv & opera_valid & allb
    ca = rv & opera_valid & alla
    cu = rv & ouv & allb
    sup.update(paper_common=int(cp.sum()), paper_common_water=int((cp & ref).sum()), average_common=int(ca.sum()),
               average_common_water=int((ca & ref).sum()), union_common=int(cu.sum()),
               pure_paper_common=int((cp & pure).sum()), pure_paper_common_water=int((cp & pure & ref).sum()),
               pure_average_common=int((ca & pure).sum()), reference_pure=int(pure.sum()),
               reference_water_frac_ge_0_5=int((rv & ref).sum()))
    score(rows, "paper", scene, cp, ref, pb, opera)
    score(rows, "s1_average", scene, ca, ref, pa, opera)
    score(rows, "opera_union", scene, cu, ref, pb, ou)
    score(rows, "s3_pure_paper", scene, cp & pure, ref, pb, opera)
    score(rows, "s3_pure_average", scene, ca & pure, ref, pa, opera)
    for cut in (0.25, 0.75):
        tag = f"{int(cut * 100):03d}"
        score(rows, f"refcut{tag}_paper", scene, cp, frac >= cut, pb, opera)
        score(rows, f"refcut{tag}_average", scene, ca, frac >= cut, pa, opera)

    # ---- S2: grid offsets and exploratory per-offset-class comparison
    left, top = float(grid["left"]), float(grid["top"])
    classes = {}
    for path in EI.opera_paths(scene):
        with rasterio.open(path) as src:
            t, same = src.transform, src.crs.to_string() == crs
        dx = (t.c - left) % 30 if same else None
        dy = (top - t.f) % 30 if same else None
        offs.append({"scene_id": scene, "product": path.name, "tile": path.name.split("_")[3], "crs": src.crs.to_string(),
                     "frozen_crs": crs, "same_crs": same, "transform": list(t)[:6], "offset_x_mod30_m": dx,
                     "offset_y_mod30_m": dy})
        if same:
            classes.setdefault((dx, dy), []).append(path)
    for (dx, dy), paths in sorted(classes.items()):
        # own grid: lattice with origin offset (dx, dy), cells entirely inside the frozen grid extent
        ncol = shape[1] - (1 if dx else 0); nrow = shape[0] - (1 if dy else 0)
        gt = Affine(30.0, 0.0, left + dx, 0.0, -30.0, top - dy)
        gshape = (nrow, ncol)
        grv, gfrac = reference(grid, gshape, gt, crs)
        gref = gfrac >= 0.5
        gw = np.zeros(gshape, bool); gv = np.zeros(gshape, bool)
        for path in paths:
            with rasterio.open(path) as src:
                assert src.transform.a == 30.0 and abs(((src.transform.c - gt.c) % 30)) < 1e-6
                col = round((gt.c - src.transform.c) / 30.0); row = round((src.transform.f - gt.f) / 30.0)
                arr = src.read(1, window=Window(col, row, ncol, nrow), boundless=True, fill_value=255)
            this = np.isin(arr, (0, 1)); gv |= this; gw |= (arr == 1) & this
        gp, gva = {}, {}
        for k, tag in RUNS.items():
            gp[k], gva[k] = probability(prob_path(scene, tag), gshape, gt, crs, Resampling.average)
        gc = grv & gv & np.logical_and.reduce(list(gva.values()))
        fw, fv = union_on(paths, shape, transform, crs)
        fc = rv & fv & alla
        key = f"{int(dx)}_{int(dy)}"
        sup[f"s2_native@{key}_common"] = int(gc.sum()); sup[f"s2_frozen@{key}_common"] = int(fc.sum())
        score(rows, f"s2_native@{key}", scene, gc, gref, gp, gw)
        score(rows, f"s2_frozen@{key}", scene, fc, ref, pa, fw)
    print(scene, {k: v for k, v in sup.items() if not k.startswith("s2")}, flush=True)
    return rows, sup, offs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--only", nargs="*", type=int)
    a = ap.parse_args()
    with (ROOT / "evaluation/grids_30m.csv").open(newline="") as source:
        grids = {int(row["scene_id"]): row for row in csv.DictReader(source)}
    scenes = [s for s in EI.SCENES if not a.only or s in a.only]
    with Pool(a.procs) as pool:
        res = pool.map(scene_job, [(s, grids[s]) for s in scenes], chunksize=1)
    sfx = "_subset" if a.only else ""
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r for rows, _, _ in res for r in rows]).to_csv(OUT / f"per_scene_counts_long{sfx}.csv", index=False)
    pd.DataFrame([s for _, s, _ in res]).to_csv(OUT / f"per_scene_support{sfx}.csv", index=False)
    pd.DataFrame([o for _, _, offs in res for o in offs]).to_csv(OUT / f"opera_grid_offsets{sfx}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
