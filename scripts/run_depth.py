"""Run Depth Anything V2 Base on one image and save the relative depth.

Examples
--------
Default Potsdam smoke test::

    python scripts/run_depth.py

Any other input (GeoTIFF, JPG or PNG)::

    python scripts/run_depth.py --input path/to/image.tif
    python scripts/run_depth.py --input photo.jpg --device cpu

Outputs land under ``outputs/`` and never inside ``data/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow "python scripts/run_depth.py" from the project root: sys.path[0] is the
# script's own directory, so the repository root has to be added explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from depthwizard import config, load_scene
from depthwizard.diagnostics import depth_overview
from depthwizard.estimator import DepthEstimator
from depthwizard.geo import write_single_band


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--input",
        type=Path,
        default=config.DEFAULT_DEMO_PATCH,
        help="input image (.tif/.tiff/.jpg/.jpeg/.png). Default: a Potsdam patch.",
    )
    p.add_argument(
        "--device",
        default=config.DEVICE_PREFERENCE,
        choices=["auto", "cpu", "cuda"],
        help="compute device (default: auto -> cuda when available, else cpu)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=config.DEPTH_DIR,
        help=f"where to write rasters (default: {config.DEPTH_DIR})",
    )
    p.add_argument("--no-geotiff", action="store_true", help="skip the GeoTIFF output")
    p.add_argument("--no-figure", action="store_true", help="skip the diagnostic PNG")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    print("=" * 72)
    print("DepthWizard - baseline depth inference")
    print("=" * 72)

    scene = load_scene(args.input)
    print("\n[input]")
    print(scene.source.describe())

    estimator = DepthEstimator(device=args.device)
    estimator.load()
    print("\n[model]")
    print(estimator.describe())

    stats = estimator.predict_scene(scene)
    d = scene.disparity

    print("\n[inference]")
    print(f"input size     : {stats.input_size[0]} x {stats.input_size[1]}")
    print(f"backbone size  : {stats.backbone_size[0]} x {stats.backbone_size[1]}")
    print(f"preprocess     : {stats.preprocess_s * 1000:.0f} ms")
    print(f"forward pass   : {stats.forward_s:.3f} s")
    print(f"postprocess    : {stats.postprocess_s * 1000:.0f} ms")
    print(f"total          : {stats.total_s:.3f} s")
    if stats.peak_vram_mb is not None:
        print(f"peak VRAM      : {stats.peak_vram_mb:.0f} MB")

    print("\n[relative depth] (unitless inverse depth; larger = nearer = higher)")
    print(f"shape          : {d.shape}  dtype {d.dtype}")
    print(f"min / max      : {d.min():.4f} / {d.max():.4f}")
    print(f"mean / std     : {d.mean():.4f} / {d.std():.4f}")
    zero_frac = float(np.mean(d <= 0.0))
    print(f"ReLU-saturated : {zero_frac * 100:.2f}% of pixels at exactly 0")

    stem = args.input.stem
    out_dir = Path(args.out_dir)
    config.ensure_output_dirs(out_dir)

    npy_path = out_dir / f"{stem}_disparity.npy"
    np.save(npy_path, d)
    written = [npy_path]

    if scene.is_georef and not args.no_geotiff:
        tif_path = write_single_band(
            out_dir / f"{stem}_disparity.tif",
            d,
            crs=scene.source.crs,
            transform=scene.source.transform,
        )
        written.append(tif_path)
    elif not args.no_geotiff:
        print(
            "\n[note] input is not georeferenced, so no GeoTIFF was written; "
            "the .npy holds the relative result"
        )

    if not args.no_figure:
        fig_path = depth_overview(
            scene.rgb,
            d,
            config.DIAGNOSTICS_DIR / f"{stem}_depth.png",
            title=f"{stem} - relative inverse depth ({stats.device}, {stats.dtype})",
        )
        written.append(fig_path)

    print("\n[written]")
    for path in written:
        print(f"  {path}")

    print(
        "\nReminder: these values are RELATIVE. Scale and shift are arbitrary and "
        "differ per image, so they are not elevations in metres."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
