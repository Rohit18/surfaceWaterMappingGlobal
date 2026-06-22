# Preparing S1 + AEF + Dynamic World data

This repository intentionally does not distribute imagery, AlphaEarth embeddings, or labels. Use this guide to create compatible GeoTIFF triplets.

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
