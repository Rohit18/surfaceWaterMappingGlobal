#!/usr/bin/env python3
"""Validate S1/AEF/Dynamic World triplets against the paper data contract."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio as rio


def names(root: Path) -> set[str]:
    return {path.name for path in root.glob("*.tif")}


def same_grid(left: rio.DatasetReader, right: rio.DatasetReader) -> bool:
    return (
        left.crs == right.crs
        and left.transform == right.transform
        and left.width == right.width
        and left.height == right.height
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-dir", type=Path, required=True)
    parser.add_argument("--aef-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--expected-index", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    s1_dir = args.s1_dir.expanduser().resolve()
    aef_dir = args.aef_dir.expanduser().resolve()
    label_dir = args.label_dir.expanduser().resolve()
    s1_names, aef_names, label_names = names(s1_dir), names(aef_dir), names(label_dir)
    common = sorted(s1_names & aef_names & label_names)
    errors = []
    for kind, available in (("AEF", aef_names), ("label", label_names)):
        missing = sorted(s1_names - available)
        if missing:
            errors.append("{} missing {} S1 filename(s), e.g. {}".format(kind, len(missing), missing[:3]))

    if args.expected_index is not None:
        with args.expected_index.open(newline="") as handle:
            expected = {row["tile_name"] for row in csv.DictReader(handle)}
        missing = sorted(expected - set(common))
        extra = sorted(set(common) - expected)
        if missing:
            errors.append("{} indexed sample(s) are absent, e.g. {}".format(len(missing), missing[:3]))
        if extra:
            errors.append("{} unindexed sample(s) are present, e.g. {}".format(len(extra), extra[:3]))

    selected = common if args.limit is None else common[: args.limit]
    for name in selected:
        try:
            with rio.open(s1_dir / name) as s1, rio.open(aef_dir / name) as aef, rio.open(label_dir / name) as label:
                if s1.count != 3:
                    errors.append("{}: expected 3 S1 bands, got {}".format(name, s1.count))
                if aef.count != 64:
                    errors.append("{}: expected 64 AEF bands, got {}".format(name, aef.count))
                if label.count != 1:
                    errors.append("{}: expected 1 label band, got {}".format(name, label.count))
                if not same_grid(s1, aef) or not same_grid(s1, label):
                    errors.append("{}: S1, AEF, and label grids differ".format(name))
                values = set(np.unique(label.read(1)).tolist())
                if not values.issubset({0, 1}):
                    errors.append("{}: label values are not binary: {}".format(name, sorted(values)[:10]))
        except Exception as exc:
            errors.append("{}: {}: {}".format(name, type(exc).__name__, exc))

    if errors:
        raise SystemExit("Dataset validation failed:\n- " + "\n- ".join(errors[:50]))
    print({"triplets": len(common), "validated": len(selected), "status": "ok"})


if __name__ == "__main__":
    main()
