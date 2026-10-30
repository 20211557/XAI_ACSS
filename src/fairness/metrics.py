"""Group fairness metrics for approval decisions: Disparate Impact ratio and Equalized Odds.

The positive outcome is approval. With y = 1 meaning default:
- approval rate: share of the group that is approved
- TPR: approval rate among applicants who repaid (y = 0); higher is better
- FPR: approval rate among applicants who defaulted (y = 1); lower is better

DI ratio = lowest group approval rate / highest group approval rate.
Equalized Odds gap = largest TPR difference or FPR difference between any two groups.
Pass/fail thresholds come from configs/config.yaml (fairness). Educational use only.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

Z_95 = 1.96  # normal quantile for the 95% interval of each group's approval rate


def _rate(values: np.ndarray) -> float:
    """Mean of a 0/1 array, or NaN when the array is empty."""
    return float(values.mean()) if len(values) else math.nan


def _round(value: float, digits: int = 4) -> float | None:
    """Round for the JSON report; NaN becomes None (JSON null)."""
    return None if math.isnan(value) else round(value, digits)


def _check_inputs(approved, y, groups) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    approved = np.asarray(approved, dtype=int)
    y = np.asarray(y, dtype=int)
    groups = pd.Series(np.asarray(groups, dtype=object))
    if not len(approved) == len(y) == len(groups):
        raise ValueError(
            f"approved, y and groups must have the same length "
            f"({len(approved)}, {len(y)}, {len(groups)})"
        )
    for name, arr in (("approved", approved), ("y", y)):
        if not np.isin(arr, (0, 1)).all():
            raise ValueError(f"{name} must contain only 0 and 1")
    return approved, y, groups


def group_metrics(approved, y, groups) -> dict[str, dict]:
    """Size, default rate, approval rate (with a 95% interval), TPR and FPR for each group.

    Rows whose group is missing (None or NaN) are left out.
    """
    approved, y, groups = _check_inputs(approved, y, groups)
    out = {}
    for g in sorted(groups.dropna().unique(), key=str):
        m = (groups == g).to_numpy()
        n = int(m.sum())
        p = _rate(approved[m])
        half = Z_95 * math.sqrt(p * (1 - p) / n)
        out[str(g)] = {
            "n": n,
            "default_rate": _rate(y[m]),
            "approval_rate": p,
            "approval_ci95": [p - half, p + half],
            "tpr_approval_if_repaid": _rate(approved[m & (y == 0)]),
            "fpr_approval_if_defaulted": _rate(approved[m & (y == 1)]),
        }
    return out


def disparate_impact(approval_rates: list[float]) -> float:
    """Lowest approval rate / highest approval rate (1.0 = equal, 0.8 = the four-fifths rule)."""
    rates = [r for r in approval_rates if not math.isnan(r)]
    if not rates or max(rates) == 0:
        return math.nan
    return min(rates) / max(rates)


def max_gap(values: list[float]) -> float:
    """Largest difference between any two groups, ignoring groups without a value."""
    vals = [v for v in values if not math.isnan(v)]
    return max(vals) - min(vals) if len(vals) >= 2 else math.nan


def fairness_report(approved, y, groups, cfg: dict, attribute: str = "") -> dict:
    """DI ratio and Equalized Odds for one protected attribute, with pass/fail and warnings.

    cfg: the `fairness` section of configs/config.yaml (di_ratio_bounds, equalized_odds_max_diff).
    The output keys match reports/audit/2_fairness.json, so the API can read either file.
    """
    rows = group_metrics(approved, y, groups)
    if len(rows) < 2:
        raise ValueError(f"{attribute or 'attribute'} needs at least 2 groups, got {list(rows)}")
    di = disparate_impact([r["approval_rate"] for r in rows.values()])
    tpr_diff = max_gap([r["tpr_approval_if_repaid"] for r in rows.values()])
    fpr_diff = max_gap([r["fpr_approval_if_defaulted"] for r in rows.values()])
    defined = [d for d in (tpr_diff, fpr_diff) if not math.isnan(d)]
    eo_gap = max(defined) if defined else math.nan

    lo, hi = cfg["di_ratio_bounds"]
    eo_limit = cfg["equalized_odds_max_diff"]
    di_pass = bool(lo <= di <= hi)
    eo_pass = bool(eo_gap <= eo_limit)

    label = attribute or "attribute"
    warnings = []
    if not di_pass:
        warnings.append(f"{label}: DI ratio {di:.3f} is outside [{lo}, {hi}]")
    for name, diff in (("TPR", tpr_diff), ("FPR", fpr_diff)):
        if diff > eo_limit:
            warnings.append(f"{label}: Equalized Odds {name} gap {diff:.3f} exceeds {eo_limit}")
    for w in warnings:
        logger.warning(w)

    return {
        "groups": {
            g: {
                "n": r["n"],
                "default_rate": _round(r["default_rate"]),
                "approval_rate": _round(r["approval_rate"]),
                "approval_ci95": [_round(v) for v in r["approval_ci95"]],
                "tpr_approval_if_repaid": _round(r["tpr_approval_if_repaid"]),
                "fpr_approval_if_defaulted": _round(r["fpr_approval_if_defaulted"]),
            }
            for g, r in rows.items()
        },
        "di_ratio_min_over_max": _round(di),
        "eo_tpr_diff": _round(tpr_diff),
        "eo_fpr_diff": _round(fpr_diff),
        "eo_gap": _round(eo_gap),
        "di_pass": di_pass,
        "eo_pass": eo_pass,
        "warnings": warnings,
    }
