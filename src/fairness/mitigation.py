"""Bias mitigation for one protected attribute (default: age band), two methods.

1. Threshold adjustment (post-processing): keep the shipped model, give each group its own
   approval cutoff. Cutoffs are chosen on the validation set and reported on the test set.
   Rule: among cutoff combinations that meet the Equalized Odds limit and the DI band and do not
   raise the default rate among approved applicants, take the one that approves the most people.
2. Reweighting (pre-processing, Kamiran and Calders 2012): retrain the shipped model's setup on
   train + validation with sample weights
   w(group, outcome) = P(group) P(outcome) / P(group, outcome),
   so group and default are independent in the weighted data. Approval stays at the fixed cutoff.
   The retrained model is only evaluated, never saved over the shipped model.

    python -m src.fairness.mitigation      (about 1 to 2 minutes, most of it the retraining)

Educational use only. Group-specific cutoffs use age directly in the decision, which real lenders
may not be allowed to do.
"""

from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import PROJECT_ROOT, get_settings, load_config
from src.fairness.data import load_splits, protected_groups, score, target_column
from src.fairness.metrics import disparate_impact, fairness_report, max_gap
from src.models.scoring import load_config as load_scoring
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def base_cutoff(scoring_cfg: dict) -> int:
    """Lowest score of an approved grade (475 in configs/scoring.yaml)."""
    approved = scoring_cfg["approval"]["approve_grades"]
    return min(g["min_score"] for g in scoring_cfg["grades"] if g["grade"] in approved)


def performance(approved, y, pd_=None) -> dict:
    """Business side of a decision rule: approvals, default rate among approved, and AUC."""
    approved, y = np.asarray(approved, int), np.asarray(y, int)
    n_approved = int(approved.sum())
    out = {
        "n_approved": n_approved,
        "approval_rate": round(float(approved.mean()), 4),
        "default_rate_approved": round(float(y[approved == 1].mean()), 4) if n_approved else None,
    }
    if pd_ is not None:
        out["auc"] = round(float(roc_auc_score(y, pd_)), 4)
    return out


def evaluate(approved, y, groups: dict[str, pd.Series], cfg: dict, pd_=None) -> dict:
    """Fairness for every protected attribute plus the performance cost."""
    approved = np.asarray(approved, int)
    return {
        "fairness": {a: fairness_report(approved, y, g, cfg, a) for a, g in groups.items()},
        "performance": performance(approved, y, pd_),
    }


# ---------------------------------------------------------------------------
# 1. Threshold adjustment
# ---------------------------------------------------------------------------


def apply_cutoffs(scores, groups, cutoffs: dict[str, float]) -> np.ndarray:
    """Approve (1) when the score reaches the cutoff of the applicant's group."""
    groups = np.asarray(groups, dtype=object)
    missing = set(groups) - set(cutoffs)
    if missing:
        raise ValueError(f"no cutoff for groups {sorted(missing, key=str)}")
    thresholds = np.array([cutoffs[g] for g in groups], dtype=float)
    return (np.asarray(scores, float) >= thresholds).astype(int)


def search_cutoffs(scores, y, groups, cfg: dict, base: float, grid: list[float]) -> dict:
    """Try every combination of group cutoffs from `grid` and keep the best one.

    Feasible = Equalized Odds gap <= cfg limit, DI inside cfg band, and default rate among
    approved applicants <= the rate under the single cutoff `base`.
    Best = most approvals; ties go to the smaller EO gap, then to cutoffs closer to `base`.
    """
    scores, y = np.asarray(scores, float), np.asarray(y, int)
    groups = np.asarray(groups, dtype=object)
    labels = sorted(set(groups), key=str)
    lo, hi = cfg["di_ratio_bounds"]
    eo_limit = cfg["equalized_odds_max_diff"]

    # Per group and cutoff: counts needed for every metric, computed once
    stats = {}
    for g in labels:
        s, yy = scores[groups == g], y[groups == g]
        n_repaid, n_default = int((yy == 0).sum()), int((yy == 1).sum())
        rows = []
        for c in grid:
            app = s >= c
            rows.append(
                {
                    "n": len(s),
                    "approved": int(app.sum()),
                    "approved_default": int((app & (yy == 1)).sum()),
                    "tpr": (app & (yy == 0)).sum() / n_repaid if n_repaid else math.nan,
                    "fpr": (app & (yy == 1)).sum() / n_default if n_default else math.nan,
                }
            )
        stats[g] = rows

    base_approved = scores >= base
    base_default_rate = float(y[base_approved].mean())

    best, best_key, n_feasible = None, None, 0
    for combo in itertools.product(range(len(grid)), repeat=len(labels)):
        rows = [stats[g][i] for g, i in zip(labels, combo, strict=True)]
        n_app = sum(r["approved"] for r in rows)
        if n_app == 0 or sum(r["approved_default"] for r in rows) / n_app > base_default_rate:
            continue
        di = disparate_impact([r["approved"] / r["n"] for r in rows])
        if math.isnan(di) or not lo <= di <= hi:
            continue
        gaps = [max_gap([r[k] for r in rows]) for k in ("tpr", "fpr")]
        gaps = [d for d in gaps if not math.isnan(d)]
        eo = max(gaps) if gaps else math.nan
        if math.isnan(eo) or eo > eo_limit:
            continue
        n_feasible += 1
        cutoffs = [grid[i] for i in combo]
        key = (n_app, -eo, -sum(abs(c - base) for c in cutoffs))
        if best_key is None or key > best_key:
            best_key = key
            best = {"cutoffs": dict(zip(labels, cutoffs, strict=True)), "eo_gap": eo, "di": di}

    if best is None:
        raise ValueError(
            "no cutoff combination meets the EO limit, the DI band and the default-rate rule; "
            "widen fairness.mitigation.cutoff_grid in configs/config.yaml"
        )
    return {
        "cutoffs": {str(g): float(c) for g, c in best["cutoffs"].items()},
        "base_cutoff": float(base),
        "combinations_checked": len(grid) ** len(labels),
        "combinations_feasible": n_feasible,
        "base_default_rate_approved": round(base_default_rate, 4),
    }


# ---------------------------------------------------------------------------
# 2. Reweighting
# ---------------------------------------------------------------------------


def reweighing_weights(groups, y) -> np.ndarray:
    """w(g, y) = P(g) P(y) / P(g, y).

    A (group, outcome) cell that is rarer than independence would predict gets weight > 1.
    In the weighted data every group has the same default rate.
    """
    df = pd.DataFrame({"g": np.asarray(groups, dtype=object), "y": np.asarray(y, int)})
    n = len(df)
    p_g = df["g"].map(df["g"].value_counts() / n)
    p_y = df["y"].map(df["y"].value_counts() / n)
    p_gy = df.groupby(["g", "y"])["y"].transform("size") / n
    return (p_g * p_y / p_gy).to_numpy(float)


def weight_table(groups, y, weights) -> dict:
    """Weight per (group, outcome) cell, for the report."""
    df = pd.DataFrame({"g": np.asarray(groups, dtype=object), "y": np.asarray(y, int)})
    table = df.assign(w=weights).groupby(["g", "y"])["w"].first()
    return {
        str(g): {
            "repaid": round(float(table[(g, 0)]), 4),
            "defaulted": round(float(table[(g, 1)]), 4),
        }
        for g in sorted(df["g"].unique(), key=str)
    }


def fit_reweighted(train_val: pd.DataFrame, weights: np.ndarray):
    """Refit the shipped model's setup (algorithm, features, constraints) with sample weights."""
    from src.data.features import feature_sets
    from src.models.train import load_yaml, make_estimator, monotone_vector

    cfg = load_yaml("train.yaml")
    summary = json.loads((PROJECT_ROOT / cfg["outputs"]["final_summary"]).read_text("utf-8"))
    winner = summary["winner"]
    if winner["resampling"] != "none":
        raise NotImplementedError("reweighting expects a shipped model without resampling")
    feats = feature_sets()["all"]
    mono = None
    if winner["model"] in cfg["monotonic"]["models"]:
        mono = monotone_vector(feats, tuple(winner["dropped_constraints"]))
    y = train_val[cfg["target"]].to_numpy()
    est = make_estimator(winner["model"], feats, "none", cfg, y, mono)
    return est.fit(train_val[feats].astype(float), y, model__sample_weight=weights)


# ---------------------------------------------------------------------------
# Run both methods
# ---------------------------------------------------------------------------


def main() -> None:
    settings = get_settings()
    fair_cfg = load_config()["fairness"]
    mcfg = fair_cfg["mitigation"]
    attr = mcfg["attribute"]
    target = target_column()
    base = base_cutoff(load_scoring())

    splits = load_splits()
    val, test = splits["validation"], splits["test"]
    y_val, y_test = val[target].to_numpy(), test[target].to_numpy()
    groups_val, groups_test = protected_groups(val), protected_groups(test)

    # Before: shipped model, single cutoff
    dec_test = score(test)
    before = evaluate(dec_test["approved"], y_test, groups_test, fair_cfg, dec_test["pd"])

    # 1. Threshold adjustment: choose on validation, apply once to test
    g = mcfg["cutoff_grid"]
    grid = list(range(g["min"], g["max"] + 1, g["step"]))
    dec_val = score(val)
    search = search_cutoffs(dec_val["score"], y_val, groups_val[attr], fair_cfg, base, grid)
    val_approved = apply_cutoffs(dec_val["score"], groups_val[attr], search["cutoffs"])
    search["validation"] = {
        "before": evaluate(dec_val["approved"], y_val, {attr: groups_val[attr]}, fair_cfg),
        "after": evaluate(val_approved, y_val, {attr: groups_val[attr]}, fair_cfg),
    }
    approved_thr = apply_cutoffs(dec_test["score"], groups_test[attr], search["cutoffs"])
    threshold = {
        **evaluate(approved_thr, y_test, groups_test, fair_cfg, dec_test["pd"]),
        "search": search,
        "decisions_changed": int((approved_thr != dec_test["approved"].to_numpy()).sum()),
    }

    # 2. Reweighting: retrain on train + validation with weights, single cutoff on test
    train_val = pd.concat([splits["train"], val])
    tv_groups = protected_groups(train_val)[attr]
    weights = reweighing_weights(tv_groups, train_val[target])
    logger.info("Retraining with reweighting on %d rows", len(train_val))
    est = fit_reweighted(train_val, weights)
    dec_rw = score(test, est)
    reweighting = {
        **evaluate(dec_rw["approved"], y_test, groups_test, fair_cfg, dec_rw["pd"]),
        "weights": weight_table(tv_groups, train_val[target], weights),
        "decisions_changed": int((dec_rw["approved"] != dec_test["approved"]).sum()),
    }

    report = {
        "attribute": attr,
        "evaluated_on": "test",
        "rows": len(test),
        "rule": {
            "di_band": fair_cfg["di_ratio_bounds"],
            "eo_limit": fair_cfg["equalized_odds_max_diff"],
            "base_cutoff": base,
            "threshold_selection": "validation set; EO <= limit, DI in band, default rate among "
            "approved not above the single-cutoff rate; most approvals wins",
        },
        "before": before,
        "methods": {"threshold_adjustment": threshold, "reweighting": reweighting},
    }
    out = settings.reports_dir / mcfg["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nTest set, attribute {attr}. Cutoffs (threshold adjustment): {search['cutoffs']}")
    print(f"{'':22s} {'DI':>7s} {'EO gap':>7s} {'approve':>8s} {'bad|app':>8s} {'AUC':>7s}")
    for name, res in (("before", before), ("threshold_adjustment", threshold),
                      ("reweighting", reweighting)):
        f, p = res["fairness"][attr], res["performance"]
        print(f"{name:22s} {f['di_ratio_min_over_max']:7.4f} {f['eo_gap']:7.4f} "
              f"{p['approval_rate']:8.4f} {p['default_rate_approved']:8.4f} {p['auc']:7.4f}")
    print(f"Saved {out.relative_to(settings.reports_dir.parent).as_posix()}")


if __name__ == "__main__":
    main()
