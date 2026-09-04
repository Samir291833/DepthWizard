"""Inspect Depth Anything V2 output against the Potsdam reference DSM.

This answers the Step 3 questions empirically instead of assuming them:

* Which direction does the raw output run -- does a larger value mean higher or
  lower ground? (Is an inversion needed at all?)
* How strongly is the output related to true elevation on real nadir imagery,
  which is out of the model's training distribution?
* How much of the output is destroyed by ReLU saturation?

It reports Spearman rho (assumption-free, decides direction) alongside Pearson r
and a least-squares affine fit (which assume elevation is linear in disparity --
true only under a small-relief approximation).

Examples
--------
::

    python scripts/inspect_depth.py                      # 8 test patches
    python scripts/inspect_depth.py --split val --n 16
    python scripts/inspect_depth.py --n 4 --figures 4
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from depthwizard import config, load_scene
from depthwizard.diagnostics import compare_to_reference, orientation_report
from depthwizard.estimator import DepthEstimator
from depthwizard.geo import read_single_band

FIELDS = [
    "patch",
    "n_valid",
    "saturated_frac",
    "spearman_rho",
    "pearson_r",
    "slope",
    "intercept",
    "fit_mae_m",
    "fit_rmse_m",
    "reference_range_m",
    "forward_s",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", default="test", choices=list(config.SPLITS))
    p.add_argument("--n", type=int, default=8, help="patches to sample (default 8)")
    p.add_argument("--figures", type=int, default=3, help="figures to save (default 3)")
    p.add_argument("--seed", type=int, default=0, help="sampling seed (default 0)")
    p.add_argument("--device", default=config.DEVICE_PREFERENCE, choices=["auto", "cpu", "cuda"])
    return p.parse_args(argv)


def sample_patches(split: str, n: int, seed: int) -> list[str]:
    """Pick ``n`` patch names that exist in both RGB and DSM for ``split``."""
    rgb_dir = config.patch_dir(split, "RGB")
    dsm_dir = config.patch_dir(split, "DSM")
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"missing patch directory: {rgb_dir}")

    names = sorted(p.name for p in rgb_dir.glob("*.tif") if (dsm_dir / p.name).exists())
    if not names:
        raise FileNotFoundError(f"no aligned RGB/DSM patches found under {rgb_dir}")

    if n >= len(names):
        return names
    rng = np.random.default_rng(seed)
    return [names[i] for i in sorted(rng.choice(len(names), size=n, replace=False))]


def main(argv=None) -> int:
    args = parse_args(argv)

    print("=" * 78)
    print("DepthWizard - depth inspection vs Potsdam reference DSM")
    print("=" * 78)

    names = sample_patches(args.split, args.n, args.seed)
    print(f"\nsplit          : {args.split}")
    print(f"patches        : {len(names)} (seed {args.seed})")

    estimator = DepthEstimator(device=args.device)
    estimator.load()
    print("\n[model]")
    print(estimator.describe())

    print(f"\n{'patch':<18} {'rho':>7} {'r':>7} {'slope':>9} "
          f"{'fitRMSE':>9} {'refRange':>9} {'sat%':>6} {'fwd_s':>6}")
    print("-" * 78)

    rows: list[dict] = []
    for i, name in enumerate(names):
        scene = load_scene(config.patch_path(args.split, "RGB", name))
        reference, _ = read_single_band(config.patch_path(args.split, "DSM", name))

        stats = estimator.predict_scene(scene)
        cmp = compare_to_reference(scene.disparity, reference)

        row = {
            "patch": name,
            "n_valid": cmp["n_valid"],
            "saturated_frac": cmp["saturated_frac"],
            "spearman_rho": cmp.get("spearman_rho", np.nan),
            "pearson_r": cmp.get("pearson_r", np.nan),
            "slope": cmp.get("slope", np.nan),
            "intercept": cmp.get("intercept", np.nan),
            "fit_mae_m": cmp.get("fit_mae_m", np.nan),
            "fit_rmse_m": cmp.get("fit_rmse_m", np.nan),
            "reference_range_m": cmp.get("reference_range_m", np.nan),
            "forward_s": stats.forward_s,
        }
        rows.append(row)

        print(
            f"{name:<18} {row['spearman_rho']:>7.3f} {row['pearson_r']:>7.3f} "
            f"{row['slope']:>9.3f} {row['fit_rmse_m']:>9.3f} "
            f"{row['reference_range_m']:>9.3f} "
            f"{row['saturated_frac'] * 100:>5.1f}% {row['forward_s']:>6.2f}"
        )

        if i < args.figures:
            orientation_report(
                scene.rgb,
                scene.disparity,
                reference,
                config.DIAGNOSTICS_DIR / f"orientation_{args.split}_{Path(name).stem}.png",
                stats=cmp,
                title=f"{name} ({args.split}) - predicted relative depth vs reference DSM",
            )

    rho = np.array([r["spearman_rho"] for r in rows], dtype=float)
    r_p = np.array([r["pearson_r"] for r in rows], dtype=float)
    rmse = np.array([r["fit_rmse_m"] for r in rows], dtype=float)
    rng_m = np.array([r["reference_range_m"] for r in rows], dtype=float)
    sat = np.array([r["saturated_frac"] for r in rows], dtype=float)

    print("\n[aggregate]")
    print(f"Spearman rho   : median {np.nanmedian(rho):+.3f}   "
          f"mean {np.nanmean(rho):+.3f}   min {np.nanmin(rho):+.3f}   max {np.nanmax(rho):+.3f}")
    print(f"Pearson r      : median {np.nanmedian(r_p):+.3f}   mean {np.nanmean(r_p):+.3f}")
    print(f"affine fit RMSE: median {np.nanmedian(rmse):.3f} m "
          f"(reference relief median {np.nanmedian(rng_m):.3f} m)")
    print(f"ReLU saturated : mean {np.nanmean(sat) * 100:.2f}% of pixels")

    positive = int(np.nansum(rho > 0))
    print("\n[verdict]")
    print(f"patches with positive rho: {positive}/{len(rows)}")
    if np.nanmedian(rho) > 0:
        print("Larger raw output corresponds to HIGHER elevation.")
        print("=> No inversion is required to treat the output as a height proxy.")
    else:
        print("Larger raw output corresponds to LOWER elevation.")
        print("=> The output must be negated before use as a height proxy.")
    print(
        "Note: rho measures monotonic agreement only. It says nothing about "
        "metric accuracy, which calibration and evaluation must establish."
    )

    config.ensure_output_dirs(config.REPORTS_DIR)
    csv_path = config.REPORTS_DIR / f"depth_inspection_{args.split}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[written]\n  {csv_path}")
    if args.figures:
        print(f"  {config.DIAGNOSTICS_DIR} (up to {args.figures} orientation figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
