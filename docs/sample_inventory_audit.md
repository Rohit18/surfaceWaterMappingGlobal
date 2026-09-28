# Training sample inventory and timing audit

## Current inventory

The released index contains 4,678 samples from 4,514 unique SWORD v16 nodes. The
sampling frame is therefore explicit: all 4,678 samples are **river-targeted**. This
does not mean every footprint contains only a river. A footprint may also intersect a
lake, reservoir, estuary, or coastline, so water contexts must remain multi-label.

The baseline inventory generated on 2026-08-27 contains:

- 3,742 training and 936 validation samples;
- 3,786 northern-hemisphere and 892 southern-hemisphere samples;
- 325 samples whose SWORD node occurs more than once;
- a median footprint area of 113.103 km2 (range 109.621-132.249 km2);
- strong temporal concentration: 2,767 samples in October, 2,414 in 2022, and 1,600
  in 2020.

The exact WGS84 polygons are in `outputs/audit/sample_footprints.geojson`; the flat
inventory is in `outputs/audit/sample_inventory.csv`. A GSHHG 2.3.7 intermediate
land-ocean-boundary overlay finds 361 footprints that intersect or fall within 5 km of
the coastline. JRC Global Surface Water v1.4 statistics have also been reduced over all
4,678 exact footprints at 30 m. Using a 1% footprint threshold, 3,778 footprints have
observed historical open water, 2,426 have persistent water, and 2,357 have seasonal
water. These classes overlap: 1,654 footprints meet both persistent and seasonal
thresholds.

The official SWOT PLD 2.02 HydroWeb WFS layer intersects 3,085 footprints and contains
31,167 unique PLD lake IDs within them. At the conservative 1% footprint threshold,
1,470 samples have a `lake_or_reservoir` context (2,416 at 0.1%, and 852 at 5%). The
WFS supplies exact polygons, IDs, and reference areas, but not the lake-versus-reservoir
provenance field. Therefore this audit does not infer a reservoir subtype. Add one
GRanD/GeoDAR reservoir overlay only if that distinction is required.

The threshold sensitivity is important. JRC maximum water extent is nonzero in 4,672
footprints, at least 0.1% in 4,581, at least 1% in 3,778, and at least 5% in 2,120.
The inventory retains the raw fractions rather than relying only on the selected 1%
context threshold.

## Representation decision

The existing dataset is sufficiently populated for evaluating the broad water contexts
*within its SWORD river-targeted scope*. It is not sufficient evidence that all global
open-water types were independently sampled. Lake and coastal counts below mean that a
river-targeted footprint happened to overlap the feature, not that the sampling design
selected an independent lake, reservoir, estuary, or shoreline.

| Multi-label context | All | Train | Validation |
| --- | ---: | ---: | ---: |
| River-targeted | 4,678 | 3,742 | 936 |
| Lake or reservoir | 1,470 | 1,189 | 281 |
| Coastal (within 5 km) | 361 | 286 | 75 |
| Persistent water | 2,426 | 1,948 | 478 |
| Seasonal water | 2,357 | 1,904 | 453 |

The exact training labels give a more useful view of exposure than footprint tags
alone. Mean Dynamic World water coverage is 7.24% per tile and the median is 2.04%;
1,613 tiles contain less than 1% labeled water, while 837 contain at least 10%. Coastal
tiles are water-rich (20.71% mean) and lake-or-reservoir tiles average 14.13%. The
important exception is `seasonal_only`: it has 703 tiles, but only 25 contain at least
10% labeled water. More generic seasonal tags would add little; the useful target is
high-water seasonal or ephemeral scenes.

The present gaps are therefore:

- independent non-river lakes;
- confirmed reservoirs, because PLD does not expose reservoir provenance here;
- independently sampled estuaries and coastlines;
- high-water seasonal or ephemeral scenes;
- southern-hemisphere and non-October local hydrologic seasons;
- stable subgroup evaluation for lake-plus-coastal cases, which currently has only 10
  validation samples.

## Mild balancing plan

Do not equalize the categories or discard the current low-water tiles. Add approximately
500--600 unique samples (11--13% of the existing set), using these as overlapping quota
tags rather than mutually exclusive buckets:

| Candidate tag | Target |
| --- | ---: |
| Non-river natural lake | 200 |
| Confirmed reservoir | 125 |
| Coastal or estuarine | 125 |
| High-water seasonal or ephemeral | 150 |

A coastal reservoir in the southern hemisphere can satisfy several targets, so the
quota sum is not a required number of unique samples. First build a candidate table
from PLD, one reservoir inventory, the shoreline layer, and JRC seasonality; no raster
generation is needed at that stage. Exclude the SWORD corridor for the non-river
targets, rank candidates by the missing tags, and only then generate the selected
S1/Dynamic World/AEF triplets.

Use the following controls during selection:

1. Prefer southern and tropical candidates and distribute acquisition dates across
   *local hydrologic seasons*, rather than making global calendar months equal.
2. Within each target, aim for roughly 50% tiles with 10--40% labeled water, 25% above
   40%, and 25% boundary or hard examples with 1--10% water.
3. Group by lake/reservoir ID, basin, or coastal segment before splitting. Do not put
   neighboring tiles from the same feature into both training and validation.
4. Reserve at least 30 validation samples for each new key stratum. These may overlap
   across strata.
5. Keep existing hard negatives. During training, use water-aware crop sampling and
   modest stratum weights instead of deleting low-water scenes.

After supplementation, report precision, recall, and IoU by context, hemisphere,
water-fraction bin, local season, and S1--Dynamic World time-gap bin. Stop adding a
stratum once those held-out metrics are stable; equal sample counts are not the goal.
The machine-readable assessment is `outputs/audit/sample_balance_summary.json`.

## Completed 600-sample supplement manifest

The centroid and footprint selection has been completed; imagery has not yet been
exported. The result contains 600 unique feature IDs and 600 unique 11 km by 11 km
grids, with the following mutually exclusive primary targets (multi-label
`selection_tags` are also retained):

| Primary target | All | Train | Validation |
| --- | ---: | ---: | ---: |
| Non-river natural lake | 200 | 160 | 40 |
| Confirmed reservoir | 125 | 100 | 25 |
| Coastal or estuarine | 125 | 100 | 25 |
| High-water seasonal or ephemeral | 150 | 120 | 30 |

The source inventories are SWOT PLD 2.02, Global Dam Watch 1.0 reservoir polygons,
GSHHG 2.3.7 coastlines, and JRC Global Surface Water 1.4. Natural-lake candidates are
rejected when the official SWORD reach service reports a reach within the conservative
10 km search box; every new centroid is at least 20 km from the released SWORD sample
centroids. The final supplement also has at least 20 km between any two new centroids.

Every selected footprint has at least 90% valid Sentinel-1 and Dynamic World coverage.
All Dynamic World scene components used in a footprint mosaic are within 24 hours of
the exact Sentinel-1 sensing timestamp: the median absolute gap is 7.928 hours and the
maximum is 19.356 hours. The seasonal primary target additionally requires at least
10% JRC seasonal-water footprint coverage and at least 10% water in the selected
Dynamic World acquisition.

Outputs:

- `outputs/audit/supplement_samples.csv`: centroids, exact scene IDs/timestamps,
  source feature IDs, projected grid definitions, coverage fractions, and splits;
- `outputs/audit/supplement_sample_footprints.geojson`: exact WGS84 grid polygons;
- `outputs/audit/supplement_samples_summary.json`: machine-readable integrity and
  distribution summary;
- `scripts/sample_water_feature_supplement.py`: resumable pool, pairing, and selection
  workflow.

There is no reason to add data merely to equalize S1--Dynamic World timing. The exact
gap bins already contain 1,174 samples at 0--6 hours, 870 at 6--12 hours, 1,313 at
12--24 hours, 1,021 at 24--48 hours, and 300 above 48 hours. First compare held-out
metrics across those bins and only intervene if performance degrades systematically
with lag.

## Lean coverage audit

Use only three additions to the existing SWORD information:

1. **SWOT Prior Lake Database (PLD)** for lake-or-reservoir intersections;
2. **JRC Global Surface Water** for observed water fraction and persistent versus
   seasonal behavior;
3. **one authoritative global shoreline** for coastal proximity.

This is sufficient for a defensible coverage inventory. HydroLAKES would largely
duplicate PLD for this purpose. HydroRIVERS, GRWL, GLWD, and a global land-cover layer
are useful for later ecological or geomorphic analysis but are not necessary to answer
whether the training set covers the major open-water contexts. Reservoir subtype is
the one remaining optional gap; use a dedicated reservoir inventory rather than the
multi-gigabyte PLD scientific auxiliaries.

Do not force one class per sample. Report, for example, `river + reservoir +
persistent`, or `river + coastal + seasonal`. Also distinguish **targeted by the
sampling design** from **incidentally intersected by the footprint**.

## S1 versus Dynamic World timing

Exact timing is now recovered for all 4,678 samples. Of these, 1,242 replacement
samples have scene IDs recorded directly by their generation manifests. For the other
3,436 original rasters, nearby Earth Engine S1 GRD candidates were compared at five
stored output-pixel centers. Exact VV, VH, and incidence-angle agreement identifies the
scene that actually contributed each raster pixel; all 3,436 originals matched. Eight
original rasters contain two matched S1 scenes.

Across the complete verified inventory:

- median absolute S1-Dynamic World gap: 16.968 hours;
- mean absolute gap: 19.388 hours;
- 1,174/4,678 (25.096%) are within 6 hours;
- 2,044/4,678 (43.694%) are within 12 hours;
- 3,357/4,678 (71.761%) are within 24 hours;
- 4,378/4,678 (93.587%) are within 48 hours;
- 2,300/4,678 (49.166%) are on the same UTC date;
- S1 is earlier in 2,315 pairs and later in 2,363 pairs.

The earlier 1,242-row subset was not representative: its median was 6.835 hours and all
were within 24 hours. The 3,436 raster-recovered originals have a median gap of 18.909
hours, and 2,115 (61.554%) are within 24 hours.

The filename date is a search anchor rather than a guaranteed S1 sensing date. The
matched S1 scene falls on the indexed UTC date for 3,709 samples; it falls on another
date for 969 samples. The recovered offsets span approximately -3 to +3 days, matching
the legacy compositing window. Therefore a same-day catalog lookup alone would have
selected no scene or the wrong scene for a material fraction of the dataset.
