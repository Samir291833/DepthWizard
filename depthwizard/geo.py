"""Georeferencing and raster I/O helpers.

Isolated from :mod:`depthwizard.loader` because the coordinate helpers here are
also what the interactive point-query needs later: a click maps back through
exactly these functions.

Two environment facts drive the implementation:

* ``pyproj`` is **not** installed in this environment, so CRS transforms go
  through :func:`rasterio.warp.transform`, which returns *lists*.
* Windows holds a hard lock on open GDAL datasets, and an unclosed writer leaves
  a silently truncated file on disk. Every access here is therefore wrapped in a
  context manager.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

from . import config

WGS84 = "EPSG:4326"


def is_georeferenced(ds: "rasterio.DatasetReader") -> bool:
    """True when ``ds`` carries a usable CRS *and* a real affine transform.

    A CRS check alone is not enough: a plain PNG/JPEG opened through GDAL, and
    the Potsdam label rasters, report ``crs=None`` with an identity transform.
    Degenerate zero-scale transforms are rejected too, since they would make
    every pixel map to the same coordinate.
    """
    if ds.crs is None:
        return False
    t = ds.transform
    if t is None or t.is_identity:
        return False
    return t.a != 0 and t.e != 0


def read_rgb_hwc(ds: "rasterio.DatasetReader") -> np.ndarray:
    """Read ``ds`` as an ``(H, W, 3)`` uint8 RGB array.

    Always slices exactly three bands. A bare ``ds.read()`` on a 4-band raster
    yields a shape the HF image processor cannot interpret ("Unable to infer
    channel dimension format"), and a 1-band read trips its mean/std check.

    Non-uint8 inputs are stretched to 0..255 with a 2nd/98th percentile clip.
    That is lossy and changes what the model sees, so it is reported by the
    caller rather than done silently. Potsdam RGB is already uint8, so this path
    is not exercised by the current dataset.
    """
    count = ds.count
    if count >= 3:
        arr = ds.read([1, 2, 3])
    elif count == 1:
        # Replicate a single band to three channels; the processor requires 3.
        arr = np.repeat(ds.read([1]), 3, axis=0)
    else:  # pragma: no cover - a 2-band raster is not a meaningful RGB source
        raise ValueError(f"cannot build RGB from a {count}-band raster: {ds.name}")

    hwc = np.transpose(arr, (1, 2, 0))  # rasterio yields CHW
    return to_uint8_rgb(hwc)


def to_uint8_rgb(hwc: np.ndarray) -> np.ndarray:
    """Coerce an ``(H, W, 3)`` array to contiguous uint8 in 0..255.

    uint8 input is returned untouched. Anything else gets a percentile stretch,
    because the image processor rejects uint16 outright and *silently divides
    float inputs by 255 twice*.
    """
    if hwc.dtype == np.uint8:
        return np.ascontiguousarray(hwc)

    finite = hwc[np.isfinite(hwc)]
    if finite.size == 0:  # pragma: no cover - degenerate all-NaN input
        return np.zeros(hwc.shape, dtype=np.uint8)

    lo, hi = np.percentile(finite, (2.0, 98.0))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(hwc.shape, dtype=np.uint8)

    scaled = (np.clip(hwc, lo, hi) - lo) / (hi - lo) * 255.0
    return np.ascontiguousarray(scaled.astype(np.uint8))


def pixel_to_world(ds_transform, row: float, col: float, centre: bool = True):
    """Map a pixel to projected ``(x, y)``.

    ``rasterio``'s ``Affine.__mul__`` takes ``(col, row)`` -- the reverse of the
    ``(row, col)`` convention used everywhere else -- and returns the pixel's
    *upper-left corner*. Getting that wrong offsets every coordinate by half a
    pixel, so the order is made explicit here and ``centre`` is the default.
    """
    offset = 0.5 if centre else 0.0
    x, y = ds_transform * (col + offset, row + offset)
    return float(x), float(y)


def world_to_lonlat(crs, x: float, y: float) -> tuple[float, float]:
    """Convert projected ``(x, y)`` in ``crs`` to ``(lon, lat)`` in WGS84.

    Uses ``rasterio.warp.transform`` because ``pyproj`` is unavailable here. It
    returns a tuple of *lists*, hence the ``[0]`` indexing.
    """
    lons, lats = warp_transform(crs, WGS84, [x], [y])
    return float(lons[0]), float(lats[0])


def write_single_band(
    path: Path,
    array: np.ndarray,
    *,
    crs=None,
    transform=None,
    nodata: float | None = None,
) -> Path:
    """Write a 2-D float32 GeoTIFF, preserving ``crs``/``transform`` when given.

    A profile is built from scratch rather than copied from the RGB source: a
    copied profile carries ``count=3``, ``dtype=uint8`` and 608x608 blocks, all
    of which are wrong for a single-band float output.
    """
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {array.shape}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    profile = dict(config.FLOAT_RASTER_PROFILE)
    profile.update(height=array.shape[0], width=array.shape[1])
    if crs is not None:
        profile["crs"] = crs
    if transform is not None:
        profile["transform"] = transform
    if nodata is not None:
        profile["nodata"] = nodata

    # Context manager is mandatory: an unclosed GDAL writer leaves a shorter,
    # silently readable, truncated file behind.
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32, copy=False), 1)
    return path


def read_single_band(path: Path) -> tuple[np.ndarray, dict]:
    """Read a single-band raster as float32 with nodata converted to NaN."""
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        meta = {
            "crs": ds.crs,
            "transform": ds.transform,
            "nodata": ds.nodata,
            "width": ds.width,
            "height": ds.height,
            "res": ds.res,
        }
    if meta["nodata"] is not None:
        arr = np.where(arr == np.float32(meta["nodata"]), np.nan, arr)
    return arr, meta
