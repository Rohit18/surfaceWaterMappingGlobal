# Frozen evaluation design

Frozen on 2026-09-14, before inspecting any OPERA or released-model result on
S1S2-Water.

## Reference and grid

- The reference label is `sentinel12_s1_<scene>_msk.tif`; source validity is
  `sentinel12_s1_<scene>_valid.tif`.
- Each scene uses its reference raster CRS. The primary 30 m grid is the smallest
  axis-aligned grid covering the reference bounds whose edges are integer
  multiples of 30 m in that CRS. This rule is independent of every method.
- Reference aggregation is area weighted. For each 30 m pixel, first aggregate
  `valid` and `water * valid`; a target pixel is reference-valid when at least
  50% of its area is source-valid. Its water fraction is
  `sum(water * valid) / sum(valid)`, and its binary label is water at a fraction
  greater than or equal to 0.5.
- Sensitivity results will repeat reference aggregation at water fractions 0.25
  and 0.75, without changing method thresholds.
- Categorical method products and validity masks are warped with nearest-neighbor
  resampling. Continuous probabilities are warped with bilinear resampling before
  applying the single released threshold.

## Method mapping and validity

- OPERA uses the DSWx-S1 B02 BWTR layer. Value 1 is water and value 0 is
  non-water. The BWTR construction includes detected inundated vegetation as
  water. Values 250 (HAND exclusion), 251 (layover/shadow), 254 (ocean mask),
  255 (nodata), and pixels outside product coverage are invalid.
- The released S1-only and S1+AlphaEarth models use their release-supplied global
  decision threshold. The threshold and checkpoint hash must be recorded before
  reading S1S2-Water labels. No scene-specific thresholding is allowed.
- The primary all-method comparison uses the intersection of reference-valid and
  valid coverage from every retained method. Each method is also reported on its
  own intersection with reference validity, together with valid pixel count and
  valid area fraction.
- A method cannot convert missing inputs, nodata, OPERA exclusion classes, or
  missing AlphaEarth embedding pixels to land or water.

## Metrics and summaries

- Per scene: TP, FP, FN, TN, evaluated pixels, valid coverage, water IoU,
  precision, recall, and F1.
- Pooled: confusion counts summed over scenes, followed by metrics computed from
  those pooled counts.
- Distribution: median and interquartile range of scene-level metrics, plus paired
  scene-level IoU differences on identical support.
- Zero-denominator metrics are recorded as missing rather than assigned a perfect
  score.
- Native-resolution model results are secondary and are never compared directly
  to a 30 m method as a resolution-controlled result.

All primary choices above remain fixed after results are available. Any additional
analysis is labeled as sensitivity or exploratory analysis.
