"""DepthWizard central configuration.

Deliberately plain Python rather than YAML so that this increment introduces no
new dependencies. All paths resolve from this file's location, so the package
behaves the same regardless of the current working directory.

Paths below were verified against the real repository layout documented in
``context/PROJECT_STATUS.md``. Nothing here writes to ``data/``.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_POTSDAM = DATA_DIR / "Processed" / "Potsdam"
TRAINING_POTSDAM = DATA_DIR / "Training" / "Potsdam"

# Full 6000x6000 source tiles (read-only reference data).
POTSDAM_RGB_TILES = PROCESSED_POTSDAM / "RGB"
POTSDAM_DSM_TILES = PROCESSED_POTSDAM / "DSM"
POTSDAM_NDSM_TILES = PROCESSED_POTSDAM / "nDSM"

# Coarse elevation reference: ~22.86 m/px in EPSG:32633. This is a calibration
# reference only -- it is roughly one pixel per 512-patch and carries no
# building-level structure.
POTSDAM_SRTM_UTM33 = PROCESSED_POTSDAM / "SRTM" / "srtm_potsdam_utm33.tif"

# Derived artifacts. Everything DepthWizard produces goes here, never into data/.
OUTPUTS_DIR = Path(os.environ.get("DEPTHWIZARD_OUTPUTS") or (PROJECT_ROOT / "outputs"))
DEPTH_DIR = OUTPUTS_DIR / "depth"
DIAGNOSTICS_DIR = OUTPUTS_DIR / "diagnostics"
REPORTS_DIR = OUTPUTS_DIR / "reports"

SPLITS = ("train", "val", "test")
MODALITIES = ("RGB", "DSM", "Labels", "nDSM")

# A known-good 512x512 georeferenced patch used as the default smoke-test input.
DEFAULT_DEMO_PATCH = TRAINING_POTSDAM / "train" / "RGB" / "2_10_0_0.tif"


def patch_dir(split: str, modality: str) -> Path:
    """Directory holding 512x512 patches for ``split`` / ``modality``."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")
    return TRAINING_POTSDAM / split / modality


def patch_path(split: str, modality: str, name: str) -> Path:
    """Path to one patch, e.g. ``patch_path("test", "DSM", "7_10_0_0.tif")``.

    Patch filenames are aligned across RGB/DSM/Labels/nDSM, so the same ``name``
    addresses the matching patch in every modality.
    """
    return patch_dir(split, modality) / name


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"

# Verified properties of this checkpoint (transformers 5.16.1):
#   * config.depth_estimation_type == "relative", head is ReLU(...) * max_depth
#   * raw output is non-negative, unbounded above, and unitless
#   * it is INVERSE DEPTH (disparity): larger value == nearer to the camera
#   * scale AND shift are arbitrary and differ per image
# For near-nadir aerial imagery, nearer == higher, so elevation increases
# monotonically with the raw value and needs NO sign flip. This is validated
# empirically against the Potsdam DSM by scripts/inspect_depth.py -- it is not
# assumed here.
DEPTH_IS_INVERSE = True

# The DINOv2-B backbone stores position embeddings for a 37x37 patch grid.
# 37 * 14 == 518, so 518x518 is the native input size and avoids any
# position-embedding interpolation. The processor rounds every side to a
# multiple of VIT_PATCH_SIZE; feeding a non-multiple makes the model silently
# discard the trailing rows/columns.
VIT_PATCH_SIZE = 14
NATIVE_INPUT_SIDE = 518

# Upper bound on the longest side handed to the backbone. The checkpoint default
# keep_aspect_ratio=True *covers* the target box, so a strongly non-square input
# would otherwise scale its long side far past 518 and blow up cost on a 4 GB
# GPU / CPU-only box. preprocess.py falls back to a square resize past this.
MAX_INFER_SIDE = 2 * NATIVE_INPUT_SIDE  # 1036

# "auto" -> CUDA when torch reports it available, else CPU.
DEVICE_PREFERENCE = os.environ.get("DEPTHWIZARD_DEVICE", "auto")


# ---------------------------------------------------------------------------
# GeoTIFF output options
# ---------------------------------------------------------------------------

# DEFLATE + floating-point predictor (3) measured ~1.77x on real Depth Anything
# output, versus 1.44x for predictor=2 and 1.18x for none. Predictor 3 is also
# the standards-blessed choice for float data.
#
# blockxsize/blockysize are pinned to 256 on purpose: the Potsdam source
# profiles carry 608x608 blocks, which GDAL pads out for a 512x512 patch and
# wastes ~41% of the file.
FLOAT_RASTER_PROFILE = {
    "driver": "GTiff",
    "dtype": "float32",
    "count": 1,
    "compress": "deflate",
    "predictor": 3,
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
}


def ensure_output_dirs(*dirs: Path) -> None:
    """Create output directories on demand. Never touches ``data/``."""
    for d in dirs or (OUTPUTS_DIR,):
        d.mkdir(parents=True, exist_ok=True)
