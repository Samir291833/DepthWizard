"""Depth Anything V2 Base inference.

Produces **relative inverse depth** (disparity) at the input image's own
resolution. Larger values mean nearer to the camera. The values are unitless,
non-negative, and their scale *and* shift are arbitrary and differ per image --
so they are never metres and never an elevation. Converting them to elevation is
the calibration stage's job, not this module's.

The public entry point is :class:`DepthEstimator`, which loads the model once and
can then be reused across many patches.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import config, preprocess


@dataclass
class InferenceStats:
    """Timing and memory for a single forward pass."""

    device: str
    dtype: str
    input_size: tuple[int, int]
    backbone_size: tuple[int, int]
    preprocess_s: float
    forward_s: float
    postprocess_s: float
    peak_vram_mb: float | None = None

    @property
    def total_s(self) -> float:
        return self.preprocess_s + self.forward_s + self.postprocess_s


def resolve_device(preference: str = config.DEVICE_PREFERENCE) -> str:
    """Pick a torch device, falling back to CPU when CUDA is unavailable.

    ``"auto"`` prefers CUDA. An explicit ``"cuda"`` that cannot be honoured
    raises, rather than silently running 100x slower on the CPU.
    """
    import torch

    pref = (preference or "auto").lower()
    available = torch.cuda.is_available()

    if pref == "cpu":
        return "cpu"
    if pref in {"cuda", "gpu"}:
        if not available:
            raise RuntimeError(
                "device='cuda' requested but torch reports CUDA unavailable "
                f"(torch {torch.__version__}, torch.version.cuda={torch.version.cuda}). "
                "This environment has a CPU-only torch wheel."
            )
        return "cuda"
    if pref != "auto":
        raise ValueError(f"unknown device preference {preference!r}")
    return "cuda" if available else "cpu"


def resolve_dtype(device: str):
    """float16 on CUDA, float32 on CPU.

    fp16 was verified numerically safe for this checkpoint (max abs deviation
    0.0140 on a ~5.3 output range, i.e. ~0.26%) and roughly halves weight
    memory, which matters on a 4 GB card. CPU fp16 is not broadly supported by
    torch kernels and would be slower, so the CPU path stays fp32.
    """
    import torch

    return torch.float16 if device == "cuda" else torch.float32


class DepthEstimator:
    """Lazily-loaded Depth Anything V2 Base depth estimator."""

    def __init__(
        self,
        model_id: str = config.MODEL_ID,
        device: str = config.DEVICE_PREFERENCE,
        *,
        clamp_to_raw_range: bool = True,
    ):
        self.model_id = model_id
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device)
        self.clamp_to_raw_range = clamp_to_raw_range
        self._processor = None
        self._model = None
        self.load_s: float | None = None

    # -- loading ---------------------------------------------------------

    def load(self) -> None:
        """Load processor and weights. Idempotent."""
        if self._model is not None:
            return

        import torch  # noqa: F401  (import cost belongs to load, not predict)
        from transformers import AutoModelForDepthEstimation

        t0 = time.perf_counter()
        self._processor = preprocess.load_processor(self.model_id)
        # `dtype=` replaces the deprecated `torch_dtype=` in transformers v5.
        model = AutoModelForDepthEstimation.from_pretrained(
            self.model_id, dtype=self.dtype
        )
        self._model = model.to(self.device).eval()
        self.load_s = time.perf_counter() - t0

    @property
    def processor(self):
        self.load()
        return self._processor

    @property
    def model(self):
        self.load()
        return self._model

    def describe(self) -> str:
        import torch

        lines = [
            f"model      : {self.model_id}",
            f"device     : {self.device}",
            f"dtype      : {str(self.dtype).replace('torch.', '')}",
            f"torch      : {torch.__version__} (cuda={torch.version.cuda})",
        ]
        if self.device == "cuda":
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            lines.append(f"gpu        : {name} ({total:.2f} GB)")
        if self.load_s is not None:
            lines.append(f"load time  : {self.load_s:.2f} s")
        return "\n".join(lines)

    # -- inference -------------------------------------------------------

    def predict(self, rgb: np.ndarray) -> tuple[np.ndarray, InferenceStats]:
        """Estimate relative inverse depth for an ``(H, W, 3)`` uint8 array.

        Returns ``(disparity, stats)`` with ``disparity`` float32 ``(H, W)`` at
        the *input* resolution.
        """
        import torch
        import torch.nn.functional as F

        self.load()
        height, width = int(rgb.shape[0]), int(rgb.shape[1])

        t0 = time.perf_counter()
        inputs, prep_info = preprocess.prepare_inputs(rgb, self._processor)
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
        t_prep = time.perf_counter() - t0

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = self._model(pixel_values=pixel_values)
        if self.device == "cuda":
            torch.cuda.synchronize()
        t_forward = time.perf_counter() - t0

        peak_vram = None
        if self.device == "cuda":
            peak_vram = torch.cuda.max_memory_allocated() / 1024**2

        t0 = time.perf_counter()
        # predicted_depth is (B, h, w): batch dim present, channel dim already
        # squeezed by the head. Resize in float32 for numerical headroom.
        raw = outputs.predicted_depth.detach().float()
        raw_min = float(raw.min())
        raw_max = float(raw.max())

        # This is the manual equivalent of
        # processor.post_process_depth_estimation(outputs, target_sizes=[(h, w)]),
        # verified bit-identical. Preferred because the library helper ends in an
        # unconditional .squeeze(), which drops a genuine leading dimension for
        # any tile whose height or width is 1 (edge strips of a tiled mosaic).
        resized = F.interpolate(
            raw.unsqueeze(1),  # (B, 1, h, w)
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)  # (B, H, W)

        disparity = resized[0]
        if self.clamp_to_raw_range:
            # Bicubic overshoots: a raw minimum of exactly 0.0 can resample to a
            # small negative value. Clamping keeps the output inside the range
            # the model actually produced.
            disparity = disparity.clamp(min=raw_min, max=raw_max)

        disparity_np = disparity.cpu().numpy().astype(np.float32)
        t_post = time.perf_counter() - t0

        stats = InferenceStats(
            device=self.device,
            dtype=str(self.dtype).replace("torch.", ""),
            input_size=(height, width),
            backbone_size=prep_info["backbone_size"],
            preprocess_s=t_prep,
            forward_s=t_forward,
            postprocess_s=t_post,
            peak_vram_mb=peak_vram,
        )
        return disparity_np, stats

    def predict_scene(self, scene) -> InferenceStats:
        """Run :meth:`predict` on a :class:`~depthwizard.loader.Scene` in place."""
        disparity, stats = self.predict(scene.rgb)
        scene.disparity = disparity
        scene.meta["inference"] = stats
        scene.meta["depth_semantics"] = (
            "relative inverse depth (disparity); larger = nearer to camera; "
            "unitless, arbitrary per-image scale and shift"
        )
        return stats
