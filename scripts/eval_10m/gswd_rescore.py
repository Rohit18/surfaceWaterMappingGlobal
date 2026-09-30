#!/usr/bin/env python3
"""GSWD (53 scenes): per-scene confusion counts for every scoring variant (Part G). CPU only, no inference.

Inputs (read only):
  models: native 10 m probabilities revision_checks_20260929/predictions_10m/openwater/width_k{0,16}_seed{42,43,44}_current_tta
  labels: Workspace/data/S1_Intercomparison_All_Scenes_subset/labels/<SID>.tif (values {0,1} scored)
  OPERA:  native 30 m B02_BWTR products (gswd/opera_products_per_scene.csv, from G0) and the paper's 3 m masks
          intercomparison_opera/binary_masks/<SID>_opera_bwtr_binary.tif
Unchanged code reused: scripts/orig/scoring.py (label/OPERA reading, counts), scripts/orig/score_10m.py (resampling
constants), scripts/orig/evaluate_intercomparison_opera_binary.py (merged_binary_opera: nearest warp, {0,1} valid,
union over products).

Variants (mask = common valid mask of the variant; "6 runs" = S1-only and S1+AEF(t), seeds 42/43/44):
  paper       3 m; model 10 m -> bilinear to label grid (as score_10m.resample_to_label, in memory; checked equal to
              the 29 Sep rasters) -> threshold; OPERA = paper 3 m mask. Mask = the paper's common mask (label, OPERA,
              the four 3 m paper rasters, the four 10 m seed-42 rasters valid; 51,897,488 px). The variant-rule mask
              (label, OPERA, 6 runs) is counted too.
  g1_average  30 m block grid (10 x 10 label pixels, trimmed to complete blocks). Reference: block valid if >= 50% of
              its area is label-valid; water if >= 50% of the valid area is water. Model: 10 m -> block grid,
              Resampling.average -> threshold. OPERA: each native product -> block grid, nearest; union.
  g1_bilinear as g1_average with Resampling.bilinear for the models.
  g2          3 m, trimmed area: model as g1_average, thresholded, each block expanded to its 10 x 10 label pixels;
              reference = 3 m label; OPERA = paper 3 m mask.
  g3_3m       paper scoring restricted to label pixels in pure blocks (all 100 label pixels valid, and all water or all
              non-water). Mask = paper common mask AND pure block.
  g3_30m      g1_average restricted to pure blocks.
  g4          scenes whose OPERA products all share the label CRS and one 30 m grid: OPERA's own grid, cells entirely
              inside the label extent. Reference: Resampling.average of the 3 m valid and water x valid layers, same
              50% rules. Model: 10 m -> Resampling.average. OPERA: products read on their own grid (union), no
              resampling.
Output: gswd/per_scene_counts_long.csv (variant, method, seed, threshold, sample_id, tp, fp, fn, tn) and
gswd/per_scene_support.csv (pixel counts per scene and variant).
"""
from __future__ import annotations

import argparse
import math
import sys
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # scoring, score_10m and evaluate_intercomparison_opera_binary
import evaluate_intercomparison_opera_binary as EOB  # noqa: E402
import scoring as S  # noqa: E402
from paths import env_root  # noqa: E402  (environment variables; see README.md)
import score_10m as T  # noqa: E402

OUT = env_root("SWM_RESOLUTION")
PREV = env_root("SWM_EVAL10M")
P10 = PREV / "predictions_10m/openwater"
R3 = PREV / "predictions_10m_on_3m"
RUNS = {("S1-only", s): f"width_k0_seed{s}_current_tta" for s in (42, 43, 44)} | \
       {("S1+AEF(t)", s): f"width_k16_seed{s}_current_tta" for s in (42, 43, 44)}
PAPER10 = ["openwater/width_k0_seed42_current_tta", "openwater/width_k16_seed42_current_tta",
           "openwater/width_k16_seed42_tminus1_tta", "tminus1/width_k16_seed42_tminus1_tta"]
THRESHOLDS = (0.30, 0.50)
NODATA = T.NODATA
B = 10  # label pixels per block side


def to_grid(src_path: Path, crs, transform, shape, resampling) -> tuple[np.ndarray, np.ndarray]:
    out = np.full(shape, NODATA, np.float32)
    with rasterio.open(src_path) as src:
        assert src.nodata == NODATA
        reproject(source=rasterio.band(src, 1), destination=out, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=NODATA, dst_transform=transform, dst_crs=crs, dst_nodata=NODATA, resampling=resampling)
    return out, np.isfinite(out) & (out != NODATA)


def blocks(a: np.ndarray, nb: int) -> np.ndarray:
    return a[:nb * B, :nb * B].reshape(nb, B, nb, B)


def expand(a: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(a, B, axis=0), B, axis=1)


def add(rows, variant, sid, thr, name, seed, pred, truth):
    c = S.counts(pred, truth)
    rows.append({"variant": variant, "method": name, "seed": seed, "threshold": thr, "sample_id": sid,
                 "tp": int(c[0]), "fp": int(c[1]), "fn": int(c[2]), "tn": int(c[3])})


def score(rows, variant, sid, mask, truth, probs, opera):
    """probs: {(name, seed): prob array}; opera: bool water array; all on the same grid; mask = common mask."""
    tv = truth[mask]
    for thr in THRESHOLDS:
        for (name, seed), p in probs.items():
            add(rows, variant, sid, thr, name, seed, p[mask] >= thr, tv)
        add(rows, variant, sid, thr, "OPERA", "", opera[mask], tv)


def scene(args) -> tuple[list, dict]:
    sid, products, g4_ok = args
    rows, sup = [], {"sample_id": sid}
    label, lvalid = S.read_label(sid)
    truth = label == 1
    opera3, ovalid3 = S.read_opera(sid)
    with rasterio.open(S.LABELS.format(sid)) as lab:
        crs, lt, h, w = lab.crs, lab.transform, lab.height, lab.width
    date = next(P10.glob(f"{RUNS[('S1-only', 42)]}/probabilities/*/{sid}_prob.tif")).parent.name
    native = {k: P10 / tag / "probabilities" / date / f"{sid}_prob.tif" for k, tag in RUNS.items()}

    # ---- paper (3 m, bilinear)
    p3, v3 = {}, {}
    for k, path in native.items():
        p3[k], v3[k] = to_grid(path, crs, lt, (h, w), Resampling.bilinear)
        ref = S.read_prob(str(R3 / "openwater" / RUNS[k] / "probabilities" / date / f"{sid}_prob.tif"))[0]
        assert np.array_equal(p3[k], ref), (sid, k, "in-memory bilinear differs from the 29 Sep 3 m raster")
    paper_common = lvalid & ovalid3
    for root in S.PAPER_3M.values():
        paper_common &= S.read_prob(S.prob_index(root)[sid])[1]
    for g in PAPER10:
        paper_common &= S.read_prob(str(R3 / g / "probabilities" / date / f"{sid}_prob.tif"))[1]
    rule_common = lvalid & ovalid3 & np.logical_and.reduce(list(v3.values()))
    sup.update(label_valid_3m=int(lvalid.sum()), label_water_3m=int((truth & lvalid).sum()),
               paper_common=int(paper_common.sum()), paper_rule_common=int(rule_common.sum()),
               paper_common_water=int((truth & paper_common).sum()))
    score(rows, "paper", sid, paper_common, truth, p3, opera3 == 1)

    # ---- block grid
    nb = h // B
    assert w // B == nb
    bt = Affine(lt.a * B, 0.0, lt.c, 0.0, lt.e * B, lt.f)
    vfrac = blocks(lvalid, nb).mean(axis=(1, 3))
    wv = blocks(truth & lvalid, nb).sum(axis=(1, 3))
    nv = blocks(lvalid, nb).sum(axis=(1, 3))
    frac = np.divide(wv, nv, out=np.zeros(vfrac.shape), where=nv > 0)
    ref_valid = vfrac >= 0.5
    ref_water = frac >= 0.5
    pure = (nv == B * B) & ((wv == 0) | (wv == B * B))
    trim = np.zeros((h, w), bool); trim[:nb * B, :nb * B] = True
    sup.update(blocks=nb * nb, trimmed_label_pixels=int(h * w - (nb * B) ** 2),
               trimmed_label_valid=int((lvalid & ~trim).sum()), ref_valid_blocks=int(ref_valid.sum()),
               ref_water_blocks=int((ref_water & ref_valid).sum()), pure_blocks=int(pure.sum()),
               pure_water_blocks=int((pure & (wv == B * B)).sum()))
    ob = EOB.merged_binary_opera(SimpleNamespace(crs=crs, transform=bt, width=nb, height=nb),
                                 [SimpleNamespace(path=Path(p)) for p in products], "nearest", 0.5)
    ob_valid, ob_water = ob != EOB.OPERA_NODATA, ob == 1
    for variant, rs in (("g1_average", Resampling.average), ("g1_bilinear", Resampling.bilinear)):
        pb, vb = {}, {}
        for k, path in native.items():
            pb[k], vb[k] = to_grid(path, crs, bt, (nb, nb), rs)
        common = ref_valid & ob_valid & np.logical_and.reduce(list(vb.values()))
        sup[f"{variant}_common"] = int(common.sum())
        sup[f"{variant}_common_water"] = int((common & ref_water).sum())
        score(rows, variant, sid, common, ref_water, pb, ob_water)
        if variant == "g1_average":
            pure_c = common & pure
            sup["g3_30m_common"] = int(pure_c.sum())
            sup["g3_30m_common_water"] = int((pure_c & ref_water).sum())
            score(rows, "g3_30m", sid, pure_c, ref_water, pb, ob_water)
            # g2: expand thresholded blocks to 3 m (expanding the probability and thresholding is identical)
            pe = {k: np.full((h, w), NODATA, np.float32) for k in pb}
            ve = {}
            for k in pb:
                pe[k][:nb * B, :nb * B] = expand(pb[k])
                ve[k] = np.zeros((h, w), bool); ve[k][:nb * B, :nb * B] = expand(vb[k])
            common2 = lvalid & ovalid3 & trim & np.logical_and.reduce(list(ve.values()))
            sup["g2_common"] = int(common2.sum())
            sup["g2_common_water"] = int((common2 & truth).sum())
            score(rows, "g2", sid, common2, truth, pe, opera3 == 1)
    # ---- g3_3m: paper scoring on pure blocks
    pure3 = np.zeros((h, w), bool); pure3[:nb * B, :nb * B] = expand(pure)
    m3 = paper_common & pure3
    sup["g3_3m_common"] = int(m3.sum()); sup["g3_3m_common_water"] = int((m3 & truth).sum())
    score(rows, "g3_3m", sid, m3, truth, p3, opera3 == 1)

    # ---- g4: OPERA's own grid
    if g4_ok:
        with rasterio.open(products[0]) as src:
            ot = src.transform
        left, top = lt.c, lt.f
        right, bottom = lt.c + w * lt.a, lt.f + h * lt.e
        c0 = math.ceil((left - ot.c) / ot.a - 1e-9); c1 = math.floor((right - ot.c) / ot.a + 1e-9)
        r0 = math.ceil((ot.f - top) / -ot.e - 1e-9); r1 = math.floor((ot.f - bottom) / -ot.e + 1e-9)
        gt = Affine(ot.a, 0.0, ot.c + c0 * ot.a, 0.0, ot.e, ot.f + r0 * ot.e)
        shape = (r1 - r0, c1 - c0)
        vf = np.full(shape, np.nan, np.float32); wf = np.full(shape, np.nan, np.float32)
        for src_arr, dst in ((lvalid.astype(np.float32), vf), ((truth & lvalid).astype(np.float32), wf)):
            reproject(source=src_arr, destination=dst, src_transform=lt, src_crs=crs, src_nodata=None,
                      dst_transform=gt, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.average)
        rv = np.isfinite(vf) & (vf >= 0.5)
        fr = np.divide(wf, vf, out=np.zeros(shape, np.float32), where=np.isfinite(vf) & (vf > 0))
        rw = fr >= 0.5
        valid_o = np.zeros(shape, bool); water_o = np.zeros(shape, bool)
        for p in products:
            with rasterio.open(p) as src:
                assert src.transform.a == ot.a and (src.transform.c - gt.c) % ot.a == 0 and (src.transform.f - gt.f) % ot.a == 0
                col = round((gt.c - src.transform.c) / ot.a); row = round((src.transform.f - gt.f) / ot.a)
                arr = src.read(1, window=Window(col, row, shape[1], shape[0]), boundless=True, fill_value=255)
            this = np.isin(arr, EOB.OPERA_VALID_VALUES)
            valid_o |= this; water_o |= np.isin(arr, EOB.OPERA_WATER_VALUES) & this
        pg, vg = {}, {}
        for k, path in native.items():
            pg[k], vg[k] = to_grid(path, crs, gt, shape, Resampling.average)
        common4 = rv & valid_o & np.logical_and.reduce(list(vg.values()))
        sup.update(g4_cells=int(shape[0] * shape[1]), g4_ref_valid=int(rv.sum()), g4_common=int(common4.sum()),
                   g4_common_water=int((common4 & rw).sum()), g4_grid_origin_x=gt.c, g4_grid_origin_y=gt.f)
        score(rows, "g4", sid, common4, rw, pg, water_o)
    print(sid, {k: v for k, v in sup.items() if k.endswith("common")}, flush=True)
    return rows, sup


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    inv = pd.read_csv(OUT / "gswd/opera_grid_offsets.csv")
    prods = pd.read_csv(OUT / "gswd/opera_products_per_scene.csv")
    jobs = []
    for r in inv.itertuples():
        if a.only and r.sample_id not in a.only:
            continue
        paths = prods[prods.sample_id == r.sample_id].path.tolist()
        jobs.append((r.sample_id, paths, bool(r.all_same_crs_as_label and r.single_30m_grid)))
    with Pool(a.procs) as pool:
        res = pool.map(scene, jobs)
    suffix = "" if not a.only else "_subset"
    pd.DataFrame([r for rows, _ in res for r in rows]).to_csv(OUT / f"gswd/per_scene_counts_long{suffix}.csv", index=False)
    pd.DataFrame([s for _, s in res]).to_csv(OUT / f"gswd/per_scene_support{suffix}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
