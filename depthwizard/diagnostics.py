"""Diagnostic figures and reference comparison for depth inspection.

This is the Step 3 toolkit: it exists to answer "is the model output actually
related to the scene's elevation, and in which direction?" empirically, rather
than by assuming a sign convention.

Nothing here claims metric elevation. The correlation statistics are
scale-and-shift invariant on purpose, because the model output is only defined up
to an arbitrary affine transform.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no interactive backend on this box

import matplotlib.pyplot as plt
import numpy as np


def compare_to_reference(
    disparity: np.ndarray,
    reference: np.ndarray,
    *,
    mask_saturated: bool = True,
) -> dict:
    """Relate raw model output to a reference elevation raster.

    ``reference`` is expected in metres (the Potsdam DSM) with nodata already
    converted to NaN. Returns scale-invariant statistics plus the best-fit
    affine coefficients.

    Pearson r and the affine fit assume elevation is linear in disparity, which
    only holds under a small-relief approximation. Spearman rho makes no such
    assumption and is the honest check of *direction*, so both are reported.

    Exactly-zero disparity is excluded by default: the head ends in ReLU, so
    saturated far-field pixels clamp to 0 with no gradient and would bias a fit.
    """
    if disparity.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: disparity {disparity.shape} vs reference {reference.shape}"
        )

    valid = np.isfinite(disparity) & np.isfinite(reference)
    if mask_saturated:
        valid &= disparity > 0.0

    n = int(valid.sum())
    out: dict = {
        "n_valid": n,
        "n_total": int(disparity.size),
        "saturated_frac": float(np.mean(disparity <= 0.0)),
    }
    if n < 100:
        out.update(pearson_r=np.nan, spearman_rho=np.nan, slope=np.nan, intercept=np.nan)
        return out

    d = disparity[valid].astype(np.float64)
    r = reference[valid].astype(np.float64)

    from scipy.stats import pearsonr, spearmanr

    out["pearson_r"] = float(pearsonr(d, r).statistic)
    out["spearman_rho"] = float(spearmanr(d, r).statistic)

    # Least-squares elevation ~= slope * disparity + intercept. Reported so the
    # sign of `slope` can be read directly; a positive slope means larger
    # disparity corresponds to higher ground.
    slope, intercept = np.polyfit(d, r, 1)
    out["slope"] = float(slope)
    out["intercept"] = float(intercept)

    fitted = slope * d + intercept
    residual = r - fitted
    out["fit_mae_m"] = float(np.mean(np.abs(residual)))
    out["fit_rmse_m"] = float(np.sqrt(np.mean(residual**2)))
    out["reference_range_m"] = float(np.nanmax(r) - np.nanmin(r))
    out["disparity_range"] = float(d.max() - d.min())
    return out


def _subsample(a: np.ndarray, b: np.ndarray, limit: int = 20000, seed: int = 0):
    if a.size <= limit:
        return a, b
    rng = np.random.default_rng(seed)
    idx = rng.choice(a.size, size=limit, replace=False)
    return a[idx], b[idx]


def depth_overview(
    rgb: np.ndarray,
    disparity: np.ndarray,
    out_png: Path,
    *,
    title: str = "",
) -> Path:
    """RGB, disparity, and the disparity histogram side by side."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    axes[0].imshow(rgb)
    axes[0].set_title("RGB input")
    axes[0].axis("off")

    im = axes[1].imshow(disparity, cmap="magma")
    axes[1].set_title("Relative inverse depth\n(bright = nearer = higher)")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="unitless")

    axes[2].hist(disparity.ravel(), bins=120, color="#4C78A8")
    axes[2].set_title("Disparity distribution")
    axes[2].set_xlabel("raw model output (unitless)")
    axes[2].set_ylabel("pixels")

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


def orientation_report(
    rgb: np.ndarray,
    disparity: np.ndarray,
    reference: np.ndarray,
    out_png: Path,
    *,
    stats: dict | None = None,
    title: str = "",
) -> Path:
    """Four-panel figure: RGB, disparity, reference DSM, and their scatter.

    The scatter is the decisive panel for the sign convention: an upward trend
    means larger model output corresponds to higher elevation.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if stats is None:
        stats = compare_to_reference(disparity, reference)

    fig, axes = plt.subplots(1, 4, figsize=(21, 5.2))

    axes[0].imshow(rgb)
    axes[0].set_title("RGB input")
    axes[0].axis("off")

    im1 = axes[1].imshow(disparity, cmap="magma")
    axes[1].set_title("Predicted (relative, unitless)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(reference, cmap="terrain")
    axes[2].set_title("Reference DSM (m)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, label="m")

    valid = np.isfinite(disparity) & np.isfinite(reference) & (disparity > 0)
    d, r = _subsample(disparity[valid], reference[valid])
    axes[3].scatter(d, r, s=2, alpha=0.15, color="#4C78A8", edgecolors="none")
    if np.isfinite(stats.get("slope", np.nan)) and d.size:
        xs = np.linspace(float(d.min()), float(d.max()), 50)
        axes[3].plot(
            xs, stats["slope"] * xs + stats["intercept"], color="#E45756", lw=1.6
        )
    axes[3].set_xlabel("predicted disparity (unitless)")
    axes[3].set_ylabel("reference elevation (m)")
    axes[3].set_title(
        "r = {pearson_r:.3f}   rho = {spearman_rho:.3f}".format(
            pearson_r=stats.get("pearson_r", np.nan),
            spearman_rho=stats.get("spearman_rho", np.nan),
        )
    )

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png
