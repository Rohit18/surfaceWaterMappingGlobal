#!/usr/bin/env python3
"""Verify the 10 m inputs: grid, band count, valid fraction, S1 product ID, and agreement with the existing 3 m inputs.

Agreement check: for every 10 m pixel whose centre lies inside the label footprint, read the existing 3 m raster
(inference manifest path) at that centre and compare with the 10 m value. Both are nearest-neighbour samples of
the same 10 m source, so values should agree except where the 3 m pixel centre and the 10 m pixel centre fall into
different source pixels.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio

from paths import env_root  # noqa: E402  (environment variables; see README.md)

MAN = env_root("SWM_S1ML") / "intercomparison_s1aef/manifests"


def sample_at_centres(src_path: str, g: dict, fw: dict) -> tuple[np.ndarray, np.ndarray]:
    """Values of an existing (3 m) raster at the 10 m pixel centres of the footprint window, and in-bounds mask."""
    x0, y0, px = g["transform"][2], g["transform"][5], g["transform"][0]
    cols = np.arange(fw["col_off"], fw["col_off"] + fw["width"])
    rows = np.arange(fw["row_off"], fw["row_off"] + fw["height"])
    xs = x0 + (cols + 0.5) * px
    ys = y0 - (rows + 0.5) * px
    with rasterio.open(src_path) as src:
        inv = ~src.transform
        cc, _ = inv * (xs, np.full_like(xs, ys[0]))
        _, rr = inv * (np.full_like(ys, xs[0]), ys)
        ci = np.floor(cc).astype(int); ri = np.floor(rr).astype(int)
        okc = (ci >= 0) & (ci < src.width); okr = (ri >= 0) & (ri < src.height)
        data = src.read()
        out = data[:, np.clip(ri, 0, src.height - 1)[:, None], np.clip(ci, 0, src.width - 1)[None, :]]
        inb = okr[:, None] & okc[None, :]
    return out, inb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-root", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--products", nargs="+", default=["aef_t", "aef_tminus1", "s1"])
    args = ap.parse_args()
    prim = {r["sample_id"]: r for r in csv.DictReader((MAN / "inference_inputs_primary_protocol_permanent.csv").open())}
    tm1 = {r["sample_id"]: r for r in csv.DictReader((MAN / "inference_inputs_nearest_SID_tminus1.csv").open())}
    old = {"aef_t": {k: v["aef_path"] for k, v in prim.items()}, "aef_tminus1": {k: v["aef_path"] for k, v in tm1.items()},
           "s1": {k: v["s1_path"] for k, v in prim.items()}}
    rows = []
    for sid in sorted(prim):
        g = json.loads((args.inputs_root / sid / "grid.json").read_text())
        fw = g["footprint_window"]
        row = {"sample_id": sid}
        for key in args.products:
            new_path = g["paths"][key]
            if not Path(new_path).exists():
                row[f"{key}_exists"] = False
                continue
            c = g.get("checks", {}).get(key, {})
            row[f"{key}_exists"] = True
            row[f"{key}_grid_ok"] = c.get("grid_ok")
            row[f"{key}_bands_ok"] = c.get("band_count_ok")
            row[f"{key}_valid_all"] = c.get("valid_fraction_all_bands")
            row[f"{key}_valid_footprint"] = c.get("valid_fraction_footprint")
            with rasterio.open(new_path) as src:
                w = ((fw["row_off"], fw["row_off"] + fw["height"]), (fw["col_off"], fw["col_off"] + fw["width"]))
                new = src.read(window=w)
                nod = src.nodata
            ref, inb = sample_at_centres(old[key][sid], g, fw)
            if new.dtype.kind == "f":
                both = np.isfinite(new).all(0) & np.isfinite(ref).all(0) & inb
                if nod is not None and np.isfinite(nod):
                    both &= ~(new == nod).any(0)
                same = np.all(np.isclose(new, ref, rtol=0, atol=1e-6), axis=0) & both
            else:
                both = ~(new == nod).any(0) & ~(ref == -128).any(0) & inb
                same = np.all(new == ref, axis=0) & both
            row[f"{key}_compared_px"] = int(both.sum())
            row[f"{key}_identical_fraction"] = float(same.sum() / max(both.sum(), 1))
            if key == "s1":
                row["s1_product_id_ok"] = c.get("product_id_ok")
                d = (new - ref)[:, both]
                row["s1_median_abs_diff_vv"] = float(np.median(np.abs(d[0]))) if d.size else ""
                row["s1_median_abs_diff_vh"] = float(np.median(np.abs(d[1]))) if d.size else ""
                row["s1_median_abs_diff_angle"] = float(np.median(np.abs(d[2]))) if d.size else ""
        rows.append(row)
        print(json.dumps(row), flush=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "sample_id", k))
    with args.out_csv.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields); wr.writeheader(); wr.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
