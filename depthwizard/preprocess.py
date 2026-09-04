"""Image preprocessing for Depth Anything V2.

The checkpoint's own processor config does the real work (518x518,
keep_aspect_ratio, ensure_multiple_of=14, bicubic, /255, ImageNet mean/std).
This module's job is to load the *right* processor class, guard the input
against the failure modes that are silent rather than loud, and bound the cost
for arbitrarily-shaped user images.

Silent failure modes being guarded here:

* A float array in ``[0, 1]`` is divided by 255 **twice** with no warning,
  collapsing the input range. Only uint8 is accepted.
* Forcing a non-multiple-of-14 side makes the patch embedding cover only
  ``(side // 14) * 14`` pixels, silently discarding the trailing rows/columns.
* ``keep_aspect_ratio=True`` *covers* the target box, so a 3:1 panorama scales
  its long side well past 518 and can exhaust memory.
"""

from __future__ import annotations

import math

import numpy as np

from . import config


def load_processor(model_id: str = config.MODEL_ID):
    """Return the image processor for ``model_id``.

    ``AutoImageProcessor`` cannot be used in this environment: transformers
    5.16.1 gates it behind torchvision, which is not installed, so importing it
    raises ``ImportError: AutoImageProcessor requires the Torchvision library``.
    The checkpoint's ``preprocessor_config.json`` names ``DPTImageProcessor``
    (the torchvision implementation); the PIL implementation reads the exact
    same config, so it is imported directly.

    Pinning the PIL class also pins reproducibility: the two backends resample
    differently (PIL bicubic on uint8 vs. antialiased tensor resize), so
    installing torchvision later would otherwise shift every depth value.
    """
    from transformers import DPTImageProcessorPil

    return DPTImageProcessorPil.from_pretrained(model_id)


def _round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def predict_backbone_side(height: int, width: int) -> tuple[int, int]:
    """Estimate the tensor size the processor will produce, before running it.

    The processor scales so the *short* side reaches ``NATIVE_INPUT_SIDE`` and
    the long side follows proportionally, each rounded to a multiple of 14. Used
    only to decide whether to bound the cost; the real shape is asserted after
    preprocessing.
    """
    native = config.NATIVE_INPUT_SIDE
    if height == width:
        return native, native

    short, long = (height, width) if height < width else (width, height)
    scaled_long = _round_to_multiple(native * long / short, config.VIT_PATCH_SIZE)
    return (native, scaled_long) if height < width else (scaled_long, native)


def prepare_inputs(
    rgb: np.ndarray,
    processor,
    *,
    max_side: int = config.MAX_INFER_SIDE,
):
    """Turn an ``(H, W, 3)`` uint8 RGB array into model inputs.

    Returns ``(inputs, info)`` where ``inputs`` is the processor's
    ``BatchFeature`` and ``info`` records what was actually done.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"expected an (H, W, 3) RGB array, got shape {rgb.shape}. "
            "rasterio yields CHW -- transpose before calling."
        )
    if rgb.dtype != np.uint8:
        raise TypeError(
            f"expected uint8 in 0..255, got {rgb.dtype}. Float inputs are "
            "silently rescaled by 1/255 twice by the processor."
        )

    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    predicted = predict_backbone_side(height, width)

    kwargs: dict = {}
    keep_aspect = True
    if max(predicted) > max_side:
        # Bound the cost: fall back to a square resize. This distorts aspect
        # ratio, which is reported so the caller can surface it.
        keep_aspect = False
        kwargs["keep_aspect_ratio"] = False

    inputs = processor(images=rgb, return_tensors="pt", **kwargs)
    pixel_values = inputs["pixel_values"]

    actual = (int(pixel_values.shape[-2]), int(pixel_values.shape[-1]))
    patch = config.VIT_PATCH_SIZE
    if actual[0] % patch or actual[1] % patch:
        raise RuntimeError(
            f"processor produced {actual}, which is not a multiple of {patch}; "
            "the model would silently ignore the trailing rows/columns"
        )

    info = {
        "input_size": (height, width),
        "backbone_size": actual,
        "keep_aspect_ratio": keep_aspect,
        "aspect_distorted": not keep_aspect and height != width,
        "scale_factor": (actual[0] / height, actual[1] / width),
    }
    return inputs, info


def tile_grid(
    height: int,
    width: int,
    tile: int = 512,
    overlap: int = 64,
) -> list[tuple[int, int, int, int]]:
    """Plan overlapping tiles covering ``height`` x ``width``.

    Returns ``(row0, col0, tile_h, tile_w)`` windows, clamped to the image so no
    window runs off the edge. Depth Anything's scale and shift are arbitrary
    *per forward pass*, so tiles inferred separately are not directly
    comparable and need alignment before mosaicking -- this only plans the
    geometry; it does not stitch.
    """
    if tile <= overlap:
        raise ValueError("tile must be larger than overlap")

    def starts(total: int) -> list[int]:
        if total <= tile:
            return [0]
        step = tile - overlap
        pos = list(range(0, max(total - tile, 0) + 1, step))
        if pos[-1] != total - tile:
            pos.append(total - tile)
        return pos

    return [
        (r, c, min(tile, height - r), min(tile, width - c))
        for r in starts(height)
        for c in starts(width)
    ]
