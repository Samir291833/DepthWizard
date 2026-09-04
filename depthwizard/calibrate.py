"""Reference calibration for relative monocular disparity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Calibration:
    method: str
    slope: float
    intercept: float
    fitted_on: str
    justified_metric: bool
    note: str

    def apply(self, disparity: np.ndarray) -> np.ndarray:
        return (self.slope * disparity + self.intercept).astype(np.float32)


def _affine(disparity: np.ndarray, elevation: np.ndarray) -> tuple[float, float, int]:
    valid = np.isfinite(disparity) & np.isfinite(elevation)
    if valid.sum() < 2:
        raise ValueError("at least two finite calibration samples are required")
    slope, intercept = np.polyfit(
        disparity[valid].astype(np.float64), elevation[valid].astype(np.float64), 1
    )
    return float(slope), float(intercept), int(valid.sum())


def oracle_calibration(disparity: np.ndarray, dsm: np.ndarray) -> Calibration:
    """Fit against the scored DSM; this is an evaluation-only upper bound."""
    slope, intercept, _ = _affine(disparity, dsm)
    return Calibration("oracle", slope, intercept, "same DSM patch being scored", False,
                       "evaluation-only upper bound; never a deployment calibration")


def srtm_calibration(patch_disparity: np.ndarray, srtm_elevation: np.ndarray) -> Calibration:
    """Fit one experimental global affine map from patch summaries to SRTM."""
    slope, intercept, n = _affine(patch_disparity, srtm_elevation)
    if n < 10:
        raise ValueError("SRTM calibration needs at least ten patches")
    return Calibration("srtm", slope, intercept, f"{n} coarse SRTM patch samples", False,
                       "experimental; validate against an independent high-resolution DSM")


def relative_surface(disparity: np.ndarray, *, ground_percentile: float = 5.0) -> np.ndarray:
    """Return a non-negative, unitless relative surface height (rDSM)."""
    finite = np.isfinite(disparity)
    if not finite.any():
        return np.full(disparity.shape, np.nan, dtype=np.float32)
    floor = float(np.nanpercentile(disparity, ground_percentile))
    return np.where(finite, np.maximum(disparity - floor, 0.0), np.nan).astype(np.float32)