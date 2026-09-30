#!/usr/bin/env python3
"""Fetch AlphaEarth annual embeddings from Source Cooperative VRTs."""

from __future__ import annotations

import csv
import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import requests

OUTPUT_MODES = ("raw", "dequantized")
COMPRESSIONS = ("deflate", "lzw", "zstd", "none")
DEFAULT_OUTPUT_MODE = "raw"
DEFAULT_COMPRESSION = "deflate"

INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.csv"
SOURCE_PREFIX = "https://data.source.coop/tge-labs/aef/"


@dataclass(frozen=True)
class AefTile:
    url: str
    bbox: Tuple[float, float, float, float]
    crs: str


def download_file(url: str, path: Path, timeout: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp.write(chunk)
    tmp_path.replace(path)


def ensure_index(cache_dir: Path, timeout: int = 120) -> Path:
    index_path = cache_dir / "aef_index.csv"
    if not index_path.exists() or index_path.stat().st_size == 0:
        download_file(INDEX_URL, index_path, timeout=timeout)
    return index_path


def cached_tile_path(cache_dir: Path, url: str) -> Path:
    suffix = Path(url.split("?", 1)[0]).suffix or ".tiff"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / "tiles" / "{}{}".format(digest, suffix)


def ensure_source_tile(url: str, cache_dir: Path, timeout: int = 120) -> Path:
    path = cached_tile_path(cache_dir, url)
    if not path.exists() or path.stat().st_size == 0:
        download_file(url, path, timeout=timeout)
    return path


def _first(row: dict, names: Sequence[str]) -> Optional[str]:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return None


def _overlaps(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _to_https_tiff(value: str) -> str:
    path = value.strip()
    if path.startswith("s3://us-west-2.opendata.source.coop/tge-labs/aef/"):
        path = path.replace("s3://us-west-2.opendata.source.coop/tge-labs/aef/", SOURCE_PREFIX, 1)
    elif path.startswith("s3://"):
        raise ValueError("Unsupported AlphaEarth S3 path: {}".format(path))
    elif path.startswith("v1/"):
        path = SOURCE_PREFIX + path
    elif not path.startswith("http"):
        path = SOURCE_PREFIX + path.lstrip("/")
    if path.endswith(".vrt"):
        path = path[:-4] + ".tiff"
    return path


def find_tiles(
    bbox_wgs84: Tuple[float, float, float, float],
    year: int,
    cache_dir: Path,
    crs: Optional[str] = None,
    timeout: int = 120,
) -> List[AefTile]:
    index_path = ensure_index(cache_dir, timeout=timeout)
    tiles: List[AefTile] = []
    with index_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("year", "")).strip() != str(year):
                continue
            row_crs = _first(row, ("crs", "proj:epsg", "epsg"))
            if row_crs and row_crs.isdigit():
                row_crs = "EPSG:{}".format(row_crs)
            if crs and row_crs and row_crs.upper() != crs.upper():
                continue
            try:
                tile_bbox = (
                    float(_first(row, ("wgs84_west", "west", "xmin")) or "nan"),
                    float(_first(row, ("wgs84_south", "south", "ymin")) or "nan"),
                    float(_first(row, ("wgs84_east", "east", "xmax")) or "nan"),
                    float(_first(row, ("wgs84_north", "north", "ymax")) or "nan"),
                )
            except ValueError:
                continue
            if not _overlaps(tile_bbox, bbox_wgs84):
                continue
            location = _first(row, ("location", "path", "href", "url", "filename"))
            if not location:
                continue
            tiles.append(AefTile(url=_to_https_tiff(location), bbox=tile_bbox, crs=row_crs or ""))
    return tiles


def write_embedding(
    bbox_wgs84: Tuple[float, float, float, float],
    output_path: Path,
    year: int,
    dst_crs: str,
    dst_transform,
    width: int,
    height: int,
    cache_dir: Path,
    output_mode: str = DEFAULT_OUTPUT_MODE,
    compression: str = DEFAULT_COMPRESSION,
    timeout: int = 120,
    cache_source_tiles: bool = False,
) -> List[str]:
    if output_mode not in OUTPUT_MODES:
        raise ValueError("Unknown output mode: {}".format(output_mode))

    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    tiles = find_tiles(bbox_wgs84, year, cache_dir, crs=dst_crs, timeout=timeout)
    if not tiles:
        tiles = find_tiles(bbox_wgs84, year, cache_dir, crs=None, timeout=timeout)
    if not tiles:
        raise RuntimeError("No AlphaEarth tiles overlap bbox for {}".format(year))

    dtype = "int8" if output_mode == "raw" else "float32"
    nodata = -128 if output_mode == "raw" else None
    data = np.full((64, height, width), -128, dtype=np.int8)

    for tile in tiles:
        source = str(ensure_source_tile(tile.url, cache_dir, timeout=timeout)) if cache_source_tiles else tile.url
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff,.tif"):
            with rasterio.open(source) as src:
                for band in range(1, min(src.count, 64) + 1):
                    reproject(
                        rasterio.band(src, band),
                        data[band - 1],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        src_nodata=-128,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        dst_nodata=-128,
                        resampling=Resampling.nearest,
                        init_dest_nodata=False,
                    )

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 64,
        "crs": dst_crs,
        "transform": dst_transform,
        "dtype": dtype,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
    }
    if nodata is not None:
        profile["nodata"] = nodata
    if compression != "none":
        profile["compress"] = compression

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        if output_mode == "dequantized":
            values = data.astype("float32")
            mask = values == -128
            values = ((values / 127.5) ** 2) * np.sign(values)
            values[mask] = np.nan
            dst.write(values)
        else:
            dst.write(data)
        dst.descriptions = tuple("A{:02d}".format(idx) for idx in range(64))
    return [tile.url for tile in tiles]
