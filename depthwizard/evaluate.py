"""Potsdam metrics with building-content and relief stratification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PatchEvaluation:
    patch: str
    building_share: float
    relief_m: float
    strata: tuple[str, str]
    metrics: dict[str, dict[str, float]]


def _metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(prediction) & np.isfinite(reference)
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "mae_m": np.nan, "rmse_m": np.nan, "pearson_r": np.nan, "spearman_rho": np.nan}
    p = prediction[valid].astype(np.float64)
    r = reference[valid].astype(np.float64)
    error = p - r
    if n < 2 or np.std(p) == 0 or np.std(r) == 0:
        pearson = spearman = np.nan
    else:
        from scipy.stats import pearsonr, spearmanr
        pearson = float(pearsonr(p, r).statistic)
        spearman = float(spearmanr(p, r).statistic)
    return {"n": n, "mae_m": float(np.mean(np.abs(error))), "rmse_m": float(np.sqrt(np.mean(error**2))),
            "pearson_r": pearson, "spearman_rho": spearman}


def building_stratum(share: float) -> str:
    if share < 0.01:
        return "none_<1%"
    if share < 0.25:
        return "low_1-25%"
    if share < 0.50:
        return "medium_25-50%"
    return "high_>=50%"


def relief_stratum(relief_m: float) -> str:
    if relief_m < 5.0:
        return "low_<5m"
    if relief_m < 10.0:
        return "medium_5-10m"
    return "high_>=10m"


def evaluate_patch(name, dsm, labels, predictions) -> PatchEvaluation:
    building_share = float(np.mean(labels == 1))
    valid_dsm = dsm[np.isfinite(dsm)]
    relief = float(np.max(valid_dsm) - np.min(valid_dsm)) if valid_dsm.size else np.nan
    return PatchEvaluation(name, building_share, relief,
                           (building_stratum(building_share), relief_stratum(relief)),
                           {method: _metrics(prediction, dsm) for method, prediction in predictions.items()})


def flatten_result(result: PatchEvaluation) -> dict:
    row = {"patch": result.patch, "building_share": result.building_share,
           "building_stratum": result.strata[0], "relief_m": result.relief_m,
           "relief_stratum": result.strata[1]}
    for method, values in result.metrics.items():
        for key, value in values.items():
            row[f"{method}_{key}"] = value
    return row


def aggregate(rows: list[dict], method_names: list[str]) -> dict[str, dict]:
    """Macro-average patch metrics for all, building, and relief strata."""
    groups = {"all": rows}
    for key in ("building_stratum", "relief_stratum"):
        for row in rows:
            groups.setdefault(row[key], []).append(row)
    output = {}
    for group, group_rows in groups.items():
        summary = {"patches": len(group_rows)}
        for method in method_names:
            for metric in ("mae_m", "rmse_m", "pearson_r", "spearman_rho"):
                values = np.array([row.get(f"{method}_{metric}", np.nan) for row in group_rows])
                summary[f"{method}_{metric}_mean"] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
                summary[f"{method}_{metric}_median"] = float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan
        output[group] = summary
    return output