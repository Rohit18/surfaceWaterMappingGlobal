# Preparing S1 + AEF + Dynamic World data

This repository intentionally does not distribute imagery, AlphaEarth embeddings, or labels. Use this guide to create compatible GeoTIFF triplets.

For the paper dataset, begin with `metadata/training_samples.csv`. It contains the
SWORD node ID, acquisition date, AEF year, exact grid, centroid, and seed-42 split for
all 4,678 usable triplets. `scripts/reconstruct_sample.py` materializes one row from
Google Earth Engine and Source Cooperative and writes a provenance record.

The source collections are `COPERNICUS/S1_GRD` (IW, dual VV/VH) and
`GOOGLE/DYNAMICWORLD/V1`. Dynamic World is selected within plus or minus one day of
the indexed S1 date, converted from label class 0 to water=1, and accepted only when
at least 90% of the output footprint is valid. AEF uses v1 annual int8 values and
nearest-neighbor reprojection.

The AEF archive starts in 2017. For the 98 paper samples whose S1 dates fall in
2015-2016, the prepared dataset uses the 2017 embedding. The `aef_year` column is
authoritative for reconstruction.

## Audit sample footprints and S1/Dynamic World timing

Create exact WGS84 polygons from the recorded raster grids; centroids are retained
only as convenient locators:

```bash
python scripts/build_sample_footprints.py
```

The timing audit reads acquisition metadata only and does not download imagery.
Dynamic World uses its source Sentinel-2 acquisition time. Positive signed gaps mean
that Sentinel-1 was acquired after Dynamic World/Sentinel-2.

```bash
python scripts/audit_s1_dw_timing.py --ee-project YOUR_EARTH_ENGINE_PROJECT
```

Generation manifests may be supplied to reuse exact recorded image provenance. This
is useful for recovery samples that were mosaicked from more than one source scene:

```bash
python scripts/audit_s1_dw_timing.py \
  --provenance-manifest path/to/dynamicworld_manifest.csv \
  --provenance-manifest path/to/recovery_manifest.csv \
  --ee-project YOUR_EARTH_ENGINE_PROJECT
```

If Earth Engine credentials are unavailable but the generation manifests contain the
exact Dynamic World time, the missing S1 sensing-start metadata can be resolved from
the free public Earth Search Sentinel-1 GRD catalog:

```bash
python scripts/audit_s1_dw_timing.py \
  --provenance-manifest path/to/dynamicworld_manifest.csv \
  --provenance-manifest path/to/recovery_manifest.csv \
  --s1-stac-fallback
```

Outputs are written under `outputs/audit/` and include the full per-sample table plus
median, percentile, threshold, ordering, and multi-scene summaries.

Rows whose S1 and Dynamic World IDs were recorded during generation are marked
`verified`. A catalog lookup based only on the legacy index date is marked
`catalog_inferred`; it is useful for diagnosis but must not be reported as measured
pair timing. Raster-pixel recovery below produces `raster_verified` rows. Use the
`verified_only` block in the JSON summary for scientific claims.

When the original S1 GeoTIFFs still exist, recover their exact source scenes by
fingerprinting stored VV, VH, and incidence-angle pixels against nearby catalog scenes:

```bash
python scripts/recover_s1_raster_provenance.py \
  --s1-raster-dir /path/to/s1_multiband \
  --ee-project YOUR_EARTH_ENGINE_PROJECT \
  --resume
```

This is stronger than choosing the nearest catalog scene by date. The legacy filename
date is only a search anchor, and the actual sensing date can differ by up to three
days. Pass the resulting `outputs/audit/s1_raster_provenance.csv` to
`audit_s1_dw_timing.py` as another provenance manifest. Rows with exact stored-raster
pixel matches are marked `raster_verified`.

Create the baseline sample inventory with:

```bash
python scripts/inventory_training_samples.py
```

It records date, split, hemisphere, tile area, repeated SWORD nodes, and the explicit
fact that every sample was river-targeted. Later PLD, JRC, and coastline annotation
tables can be merged without changing the base inventory:

```bash
python scripts/inventory_training_samples.py \
  --annotation-csv outputs/audit/pld_annotations.csv \
  --annotation-csv outputs/audit/jrc_gsw_annotations.csv \
  --annotation-csv outputs/audit/coastline_annotations.csv
```

Expected optional fields are `pld_overlap_fraction`, `pld_feature_count`,
`pld_reservoir_count` or `pld_reservoir_flag`, `jrc_permanent_fraction`,
`jrc_seasonal_fraction`, and `coastline_distance_km`. Classifications remain
multi-label because rivers, lakes, reservoirs, and coastal waters can overlap.

Retrieve only official PLD polygons near the sample footprints through HydroWeb's
public WFS, rather than downloading the 6--42 GB global PLD products:

```bash
python scripts/annotate_pld_wfs.py
```

The request is batched and cached under `outputs/audit/pld_wfs_cache/`, so interrupted
runs resume without repeating completed network work. The public layer supplies PLD
geometry, lake ID, and reference area, but not reservoir provenance; the output labels
the overlap `lake_or_reservoir` and does not guess the subtype.

For a thresholded coastal audit, use GSHHG L1 and L5 shoreline shapefiles:

```bash
python scripts/annotate_coastline.py \
  --coastline-shapefile data/GSHHG_i_L1.shp \
  --coastline-shapefile data/GSHHG_i_L5.shp \
  --threshold-km 5
```

The distance is measured from the complete sample footprint, not only its centroid.

JRC Global Surface Water persistence and seasonality fractions can be generated without
exporting imagery:

```bash
earthengine authenticate --auth_mode=notebook --force
python scripts/annotate_jrc_gsw_ee.py \
  --ee-project YOUR_EARTH_ENGINE_PROJECT --resume
```

Then rerun `inventory_training_samples.py` with all generated annotation CSVs.
The default multi-label context threshold is 1% of the footprint; raw JRC fractions
and counts at 0%, 0.1%, 1%, and 5% are retained for sensitivity analysis.

If the original and replacement Dynamic World generation manifests are available,
extract the exact positive-pixel exposure (repeat `--manifest` for every manifest):

```bash
python scripts/annotate_dw_label_manifests.py \
  --manifest path/to/original_dynamicworld_manifest.csv \
  --manifest path/to/replacement_manifest.csv

python scripts/inventory_training_samples.py \
  --annotation-csv outputs/audit/pld_annotations.csv \
  --annotation-csv outputs/audit/jrc_gsw_annotations.csv \
  --annotation-csv outputs/audit/coastline_annotations.csv \
  --annotation-csv outputs/audit/dw_label_annotations.csv

python scripts/assess_sample_balance.py
```

The balance assessment treats contexts as overlapping, reports label-pixel exposure,
and emits a modest supplementation plan. It does not recommend equalizing tile counts.

## Required rasters per training tile

| Raster | Bands | Data type | Meaning |
| --- | ---: | --- | --- |
| S1 input | 3 | float32 preferred | `VV`, `VH`, `angle`, in that order |
| AEF input | 64 | raw embedding values | Annual AlphaEarth Foundation embedding bands, in source order |
| DW target | 1 | integer or float | Binary mask: water=`1`, other=`0` |

The training code reads all values as float32, converts non-finite values to zero, and binarizes labels at `> 0.5`. Therefore, encode water as one and non-water as zero. Do not encode Dynamic World class IDs directly as the target: class ID zero means water, but would be interpreted as non-water by the trainer.

## Alignment checklist

For each S1/AEF/label triplet, ensure:

- exactly the same CRS, affine transform, width, and height;
- exactly one pixel grid, resolution, and area of interest;
- S1 is exactly three bands and AEF is exactly 64 bands;
- no duplicate filename exists anywhere under the AEF directory;
- file names match exactly when using directory-based loading.

The command below is a quick manual inspection for one file. Repeat it for its S1, AEF, and label partners and compare the reported grid fields.

```bash
gdalinfo data/s1/tile_0001.tif
gdalinfo data/aef/tile_0001.tif
gdalinfo data/labels/tile_0001.tif
```

## Example directory layout

```text
data/
  s1/
    tile_0001.tif
    tile_0002.tif
  aef/
    tile_0001.tif
    tile_0002.tif
  labels/
    tile_0001.tif
    tile_0002.tif
```

Or create a manifest from `training_manifest.csv` and train only a selected `sample_id`:

```bash
python src/train.py \
  --triplet_manifest examples/training_manifest.csv \
  --triplet_sample_id demo_region \
  --artifact_dir runs/demo_region \
  --n_s1_bands 3 --n_aef_bands 64 --n_proj_bands 16 \
  --loss ce_dice --batch_size 4 --crop_size 512
```

## Preparing inference data

Inference needs no Dynamic World target. Provide a directory of three-band S1 GeoTIFFs named `s1_YYYY-MM-DD.tif` and one 64-band annual AEF GeoTIFF. The inference code reprojects AEF windows to each S1 scene grid, but the S1 files in a single run must share the same CRS, transform, width, and height.

```text
data/inference/
  s1/
    s1_2025-01-15.tif
    s1_2025-01-27.tif
  alphaearth_2025.tif
```
