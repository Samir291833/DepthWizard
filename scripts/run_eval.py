"""Evaluate all 484 aligned Potsdam test patches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import rasterio

from depthwizard import config, load_scene
from depthwizard.calibrate import oracle_calibration, srtm_calibration
from depthwizard.dsm import make_dsm, make_rdsm
from depthwizard.estimator import DepthEstimator
from depthwizard.evaluate import aggregate, evaluate_patch, flatten_result
from depthwizard.geo import read_single_band


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default=config.DEVICE_PREFERENCE, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out-dir", type=Path, default=config.REPORTS_DIR)
    parser.add_argument("--save-depth", action="store_true")
    return parser.parse_args(argv)


def _names() -> list[str]:
    rgb_dir = config.patch_dir("test", "RGB")
    names = sorted(p.name for p in rgb_dir.glob("*.tif"))
    required = [config.patch_dir("test", modality) for modality in ("DSM", "Labels")]
    names = [name for name in names if all((directory / name).exists() for directory in required)]
    if len(names) != 484:
        raise RuntimeError(f"expected 484 aligned test patches, found {len(names)}")
    return names


def _srtm_at_patch(path: Path, rgb_path: Path) -> float:
    with rasterio.open(rgb_path) as rgb, rasterio.open(path) as srtm:
        x, y = rgb.transform * (rgb.width / 2.0, rgb.height / 2.0)
        row, col = srtm.index(x, y)
        if not (0 <= row < srtm.height and 0 <= col < srtm.width):
            return np.nan
        value = float(srtm.read(1)[row, col])
        return value if srtm.nodata is None or value != srtm.nodata else np.nan


def main(argv=None) -> int:
    args = parse_args(argv)
    names = _names()
    estimator = DepthEstimator(device=args.device)
    estimator.load()
    print(f"evaluating {len(names)} Potsdam test patches on {estimator.device}")

    srtm_values, disparity_summaries = [], []
    disparities, references, labels_by_name = {}, {}, {}
    for index, name in enumerate(names, 1):
        scene = load_scene(config.patch_path("test", "RGB", name))
        estimator.predict_scene(scene)
        disparities[name] = scene.disparity
        references[name], _ = read_single_band(config.patch_path("test", "DSM", name))
        labels_by_name[name], _ = read_single_band(config.patch_path("test", "Labels", name))
        srtm_values.append(_srtm_at_patch(config.POTSDAM_SRTM_UTM33, scene.source.path))
        disparity_summaries.append(float(np.nanmedian(scene.disparity)))
        if args.save_depth:
            config.ensure_output_dirs(config.DEPTH_DIR)
            np.save(config.DEPTH_DIR / f"{Path(name).stem}_disparity.npy", scene.disparity)
        if index % 25 == 0 or index == len(names):
            print(f"  inference {index}/{len(names)}")

    srtm_fit = srtm_calibration(np.array(disparity_summaries), np.array(srtm_values))
    rows = []
    for name in names:
        disparity, dsm = disparities[name], references[name]
        oracle = oracle_calibration(disparity, dsm)
        predictions = {"rdsm": make_rdsm(disparity), "oracle_dsm": make_dsm(disparity, oracle),
                       "srtm_dsm": make_dsm(disparity, srtm_fit)}
        row = flatten_result(evaluate_patch(name, dsm, labels_by_name[name], predictions))
        row["rdsm_mae_m"] = np.nan
        row["rdsm_rmse_m"] = np.nan
        rows.append(row)

    config.ensure_output_dirs(args.out_dir)
    csv_path = args.out_dir / "potsdam_test_evaluation.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    summaries = aggregate(rows, ["rdsm", "oracle_dsm", "srtm_dsm"])
    report = {"patches": len(rows), "srtm_calibration": srtm_fit.__dict__,
              "oracle_warning": "oracle_dsm is evaluation-only and fitted on each scored DSM patch",
              "srtm_verdict": "experimental; usefulness determined by comparison to Potsdam DSM",
              "summary": summaries}
    json_path = args.out_dir / "potsdam_test_evaluation_summary.json"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print(f"written: {csv_path}")
    print(f"written: {json_path}")
    print("SRTM fit: slope={:.6f}, intercept={:.3f}".format(srtm_fit.slope, srtm_fit.intercept))
    for method in ("oracle_dsm", "srtm_dsm"):
        summary = summaries["all"]
        print(f"{method}: RMSE median={summary[method + '_rmse_m_median']:.3f} m, MAE median={summary[method + '_mae_m_median']:.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())