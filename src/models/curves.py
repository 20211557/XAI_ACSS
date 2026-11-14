"""ROC and Precision-Recall curve points of the shipped model on the test set (real data).

    python -m src.models.curves

Reads the shipped model's test-set scores (data/processed/final_test_scores.csv, written by
`python -m src.models.train --stage final`) and writes reports/test_curves.json, which GET /metrics
and the dashboard's model performance tab read. The API container cannot see data/, so the points
are stored in reports/. Rerun this after retraining: tests/test_curves.py checks the file against
reports/final_summary.json.

Points: thresholds are taken at log-spaced counts of predicted defaults, so the low-FPR,
high-precision end, where the curves change fastest, gets most points. Every point is an exact
threshold of the full curve, not an interpolation. Positive class = default = rejected.
Educational use only.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import PROJECT_ROOT, get_settings, load_config
from src.models.scoring import load_config as load_scoring
from src.models.scoring import pd_to_score

DIGITS = 5


def approval_cutoff(scoring: dict) -> int:
    """Lowest approved score (score >= cutoff is approved), from configs/scoring.yaml."""
    approve = scoring["approval"]["approve_grades"]
    return min(g["min_score"] for g in scoring["grades"] if g["grade"] in approve)


def curve_points(y, pd_, n_points: int, scoring: dict, rejected=None) -> dict:
    """ROC and PR points at exact thresholds, plus AUC, average precision and the operating point.

    y: 1 = default. pd_: predicted default probability. rejected: the actual decisions
    (True = rejected); None means score below the approval cutoff.
    Each point's `score` is the credit score at that threshold: rejecting everyone at or below
    that score gives the point. The ROC curve starts at (0, 0) with no threshold.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(pd_, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    p_sorted, y_sorted = p[order], y[order]
    tp_all, fp_all = np.cumsum(y_sorted), np.cumsum(1 - y_sorted)
    last = np.r_[np.flatnonzero(np.diff(p_sorted)), len(p) - 1]  # last row of each distinct PD
    tp, fp, thresholds = tp_all[last], fp_all[last], p_sorted[last]
    flagged = tp + fp
    targets = np.unique(np.round(np.geomspace(1, len(p), n_points)))
    idx = np.unique(np.minimum(np.searchsorted(flagged, targets), len(flagged) - 1))
    positives, negatives = int(tp[-1]), int(fp[-1])

    fpr = np.round(fp[idx] / negatives, DIGITS)
    tpr = np.round(tp[idx] / positives, DIGITS)
    precision = np.round(tp[idx] / flagged[idx], DIGITS)
    score = pd_to_score(thresholds[idx], scoring).tolist()

    cutoff = approval_cutoff(scoring)
    if rejected is None:
        rejected = pd_to_score(p, scoring) < cutoff
    rejected = np.asarray(rejected, dtype=bool)
    op_tp = int((rejected & (y == 1)).sum())
    op_fp = int((rejected & (y == 0)).sum())
    return {
        "rows": int(len(p)),
        "defaults": positives,
        "auc": round(float(roc_auc_score(y, p)), DIGITS),
        "average_precision": round(float(average_precision_score(y, p)), DIGITS),
        "operating_point": {
            "cutoff_score": cutoff,
            "rejected": int(rejected.sum()),
            "fpr": round(op_fp / negatives, DIGITS),
            "tpr": round(op_tp / positives, DIGITS),
            "recall": round(op_tp / positives, DIGITS),
            "precision": round(op_tp / max(op_tp + op_fp, 1), DIGITS),
        },
        "roc": {"fpr": [0.0, *fpr.tolist()], "tpr": [0.0, *tpr.tolist()], "score": [None, *score]},
        "pr": {
            "recall": tpr.tolist(),
            "precision": precision.tolist(),
            "score": score,
            "baseline": round(positives / len(p), DIGITS),
        },
    }


def main() -> None:
    with open(PROJECT_ROOT / "configs" / "train.yaml", encoding="utf-8") as f:
        outputs = yaml.safe_load(f)["outputs"]
    summary = json.loads((PROJECT_ROOT / outputs["final_summary"]).read_text(encoding="utf-8"))
    scores = pd.read_csv(PROJECT_ROOT / outputs["test_scores"])
    if len(scores) != summary["test"]["rows"]:
        raise ValueError(
            f"{outputs['test_scores']} has {len(scores)} rows, final_summary.json expects "
            f"{summary['test']['rows']}: rerun the final training stage"
        )
    cfg = load_config()["api"]
    points = curve_points(
        scores["default"],
        scores["pd"],
        cfg["curve_points"],
        load_scoring(),
        rejected=scores["approved"].to_numpy() == 0,
    )
    payload = {
        "model_file": summary["model_file"],
        "evaluated_on": summary["evaluated_on"],
        "source": outputs["test_scores"],
        "generated_by": "python -m src.models.curves",
        "positive_class": "default (rejected)",
        **points,
    }
    path = get_settings().reports_dir / cfg["reports"]["test_curves"]
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    op = points["operating_point"]
    print(f"{points['rows']} test rows, {points['defaults']} defaults, "
          f"{len(points['pr']['recall'])} points")
    print(f"AUC {points['auc']} (final_summary {summary['test']['auc']}), "
          f"average precision {points['average_precision']}")
    print(f"cutoff {op['cutoff_score']}: precision {op['precision']} "
          f"(final_summary {summary['test']['precision']}), recall {op['recall']} "
          f"(final_summary {summary['test']['recall']})")
    print(f"Saved {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
