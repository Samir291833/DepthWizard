"""DepthWizard -- single-view height estimation and 3D flythrough (SIH 26175).

Pipeline stages are separate modules so each can be tested on its own:

    loader -> preprocess -> estimator -> (calibrate -> dsm -> reconstruct)

Only the stages implemented so far are exported. Terminology used throughout:

``disparity``
    Raw Depth Anything V2 output: relative inverse depth, unitless, larger =
    nearer. Never metres.
``rDSM``
    Relative surface model derived from disparity without an external reference.
``DSM``
    Metric surface model, only produced when a reference calibration justifies it.
"""

from __future__ import annotations

from . import config
from .estimator import DepthEstimator, InferenceStats, resolve_device, resolve_dtype
from .loader import Scene, SourceInfo, load_scene

__all__ = [
    "config",
    "DepthEstimator",
    "InferenceStats",
    "Scene",
    "SourceInfo",
    "load_scene",
    "resolve_device",
    "resolve_dtype",
]
