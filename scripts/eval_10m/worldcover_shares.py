#!/usr/bin/env python3
"""Class shares of the four changed-error categories (all scenes) from analyze_s1_vs_s1aef_worldcover_errors.py output.

Writes <dir>/class_shares.json: scene groups, pixel totals and every WorldCover class's pixel count and share
(validation_group = all, chip_group = all). --expect-3m checks the manuscript values (44/7/2; FN fixed: permanent water
89.6%, mangroves 4.8%; FP fixed: bare/sparse 41.5%, grassland 36.2%, tree cover 14.1%) at one decimal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CATEGORIES = {"aef_fixes_s1_fn": "corrected false negatives", "aef_fixes_s1_fp": "corrected false positives",
              "aef_adds_fn": "added false negatives", "aef_adds_fp": "added false positives"}
EXPECTED_3M = {"chip_groups": {"aef_helps": 44, "neutral": 7, "aef_hurts": 2},
               "aef_fixes_s1_fn": {"Permanent water": 89.6, "Mangroves": 4.8},
               "aef_fixes_s1_fp": {"Bare / sparse vegetation": 41.5, "Grassland": 36.2, "Tree cover": 14.1}}


def shares(d: Path) -> dict:
    summary = json.loads((d / "summary.json").read_text())
    cls = pd.read_csv(d / "worldcover_error_classes.csv")
    cls = cls[(cls.validation_group == "all") & (cls.chip_group == "all")]
    out = {"dir": str(d), "thresholds": summary["thresholds"], "sample_count": summary["sample_count"],
           "chip_groups": summary["chip_groups"], "chip_iou": summary["chip_iou"],
           "changed_error_totals": summary["changed_error_totals"], "categories": {}}
    for cat, name in CATEGORIES.items():
        sub = cls[cls.category == cat].sort_values("pixel_count", ascending=False)
        total = int(sub.pixel_count.sum())
        assert total == summary["changed_error_totals"][cat], (cat, total)
        out["categories"][cat] = {"name": name, "total_pixels": total, "classes": [
            {"class": r.worldcover_class, "code": int(r.worldcover_code), "pixels": int(r.pixel_count),
             "share_pct": 100 * r.pixel_count / total, "scenes": int(r.chip_count)} for r in sub.itertuples()]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--expect-3m", action="store_true")
    a = ap.parse_args()
    res = shares(a.dir)
    if a.expect_3m:
        checks = {"chip_groups": {k: [res["chip_groups"].get(k, 0), v] for k, v in EXPECTED_3M["chip_groups"].items()}}
        ok = all(g == e for g, e in checks["chip_groups"].values())
        for cat in ("aef_fixes_s1_fn", "aef_fixes_s1_fp"):
            got = {c["class"]: round(c["share_pct"], 1) for c in res["categories"][cat]["classes"]}
            checks[cat] = {k: [got.get(k), v] for k, v in EXPECTED_3M[cat].items()}
            ok &= all(g == e for g, e in checks[cat].values())
        res["check_vs_manuscript"] = {"computed_vs_expected": checks, "passed": bool(ok)}
    (a.dir / "class_shares.json").write_text(json.dumps(res, indent=2) + "\n")
    print("groups", res["chip_groups"])
    for cat, c in res["categories"].items():
        top = ", ".join(f"{x['class']} {x['share_pct']:.1f}% ({x['pixels']:,})" for x in c["classes"][:5])
        print(f"{c['name']} (n={c['total_pixels']:,}): {top}")
    if a.expect_3m:
        print("check passed:", res["check_vs_manuscript"]["passed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
