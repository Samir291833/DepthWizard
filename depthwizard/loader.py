"""Input handling: turn a file on disk into a :class:`Scene`.

:class:`Scene` is the single object that flows through every downstream stage
(preprocess -> estimate -> calibrate -> DSM -> mesh -> query), which is what
lets each stage be tested on its own.

Two input families are supported and deliberately kept distinguishable:

* ``.jpg`` / ``.jpeg`` / ``.png`` -- pixels only, no geospatial meaning.
* ``.tif`` / ``.tiff`` -- may or may not carry a CRS and affine transform. The
  file is inspected; a CRS is never assumed.

A GeoTIFF that lacks usable georeferencing is reported as non-georeferenced and
routed down the relative pipeline, rather than being trusted because of its
extension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio

from . import geo

RASTER_SUFFIXES = {".tif", ".tiff"}
PLAIN_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SUPPORTED_SUFFIXES = RASTER_SUFFIXES | PLAIN_IMAGE_SUFFIXES


@dataclass(frozen=True)
class SourceInfo:
    """Everything known about the input file itself."""

    path: Path
    kind: str  # "raster" | "image"
    is_georef: bool
    width: int
    height: int
    band_count: int
    dtype: str
    crs: object | None = None
    transform: object | None = None
    bounds: tuple | None = None
    res: tuple | None = None
    notes: tuple[str, ...] = ()

    @property
    def pixel_size(self) -> float | None:
        """Ground sampling distance in CRS units, when georeferenced."""
        if not self.is_georef or self.res is None:
            return None
        return float(abs(self.res[0]))

    def describe(self) -> str:
        lines = [
            f"path       : {self.path}",
            f"kind       : {self.kind}",
            f"size       : {self.width} x {self.height} ({self.band_count} band(s), {self.dtype})",
            f"georef     : {self.is_georef}",
        ]
        if self.is_georef:
            lines += [
                f"crs        : {self.crs}",
                f"resolution : {self.pixel_size} CRS units/px",
                f"bounds     : {self.bounds}",
            ]
        for note in self.notes:
            lines.append(f"note       : {note}")
        return "\n".join(lines)


@dataclass
class Scene:
    """A loaded scene plus whatever later stages have computed for it.

    ``disparity`` is named for what the model actually emits: relative inverse
    depth, larger meaning nearer. It is not depth in metres and not an
    elevation. Calibrated products are added by later stages.
    """

    rgb: np.ndarray  # (H, W, 3) uint8
    source: SourceInfo
    disparity: np.ndarray | None = None  # (H, W) float32, unitless
    meta: dict = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def is_georef(self) -> bool:
        return self.source.is_georef

    def pixel_to_lonlat(self, row: float, col: float) -> tuple[float, float] | None:
        """Pixel -> ``(lon, lat)``, or ``None`` when not georeferenced."""
        if not self.is_georef:
            return None
        x, y = geo.pixel_to_world(self.source.transform, row, col)
        return geo.world_to_lonlat(self.source.crs, x, y)


def load_scene(path: str | Path) -> Scene:
    """Load ``path`` into a :class:`Scene`, inspecting its metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"input does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"unsupported input type {suffix!r}; expected one of "
            f"{sorted(SUPPORTED_SUFFIXES)}"
        )

    if suffix in RASTER_SUFFIXES:
        return _load_raster(path)
    return _load_plain_image(path)


def _load_raster(path: Path) -> Scene:
    notes: list[str] = []
    with rasterio.open(path) as ds:
        georef = geo.is_georeferenced(ds)
        src_dtype = ds.dtypes[0]
        rgb = geo.read_rgb_hwc(ds)

        if ds.count > 3:
            notes.append(f"raster has {ds.count} bands; used bands 1,2,3 as R,G,B")
        elif ds.count == 1:
            notes.append("single-band raster replicated to 3 channels")
        if src_dtype != "uint8":
            notes.append(
                f"source dtype {src_dtype} stretched to uint8 via 2-98 percentile clip "
                "(lossy; affects model input)"
            )
        if not georef:
            notes.append(
                "no usable CRS/transform found -- treated as non-georeferenced, "
                "outputs will be RELATIVE only"
            )

        info = SourceInfo(
            path=path,
            kind="raster",
            is_georef=georef,
            width=ds.width,
            height=ds.height,
            band_count=ds.count,
            dtype=src_dtype,
            crs=ds.crs if georef else None,
            transform=ds.transform if georef else None,
            bounds=tuple(ds.bounds) if georef else None,
            res=ds.res if georef else None,
            notes=tuple(notes),
        )
    return Scene(rgb=rgb, source=info)


def _load_plain_image(path: Path) -> Scene:
    from PIL import Image

    with Image.open(path) as im:
        mode = im.mode
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)

    notes = ["non-georeferenced input -- outputs will be RELATIVE only"]
    if mode != "RGB":
        notes.append(f"converted from PIL mode {mode} to RGB")

    info = SourceInfo(
        path=path,
        kind="image",
        is_georef=False,
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
        band_count=3,
        dtype="uint8",
        notes=tuple(notes),
    )
    return Scene(rgb=np.ascontiguousarray(rgb), source=info)
