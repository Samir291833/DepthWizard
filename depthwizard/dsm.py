"""Relative and explicitly calibrated surface-model products."""

from __future__ import annotations

import numpy as np

from .calibrate import Calibration, relative_surface


def make_rdsm(disparity: np.ndarray) -> np.ndarray:
    """Create rDSM without claiming physical scale or absolute datum."""
    return relative_surface(disparity)


def make_dsm(disparity: np.ndarray, calibration: Calibration) -> np.ndarray:
    """Apply an explicit calibration; oracle use remains visibly evaluation-only."""
    if calibration.method not in {"oracle", "srtm"}:
        raise ValueError(f"unsupported metric calibration method: {calibration.method}")
    return calibration.apply(disparity)