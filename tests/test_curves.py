"""ROC / PR curve points (src/models/curves.py): exact thresholds on toy data, and the committed
reports/test_curves.json matches the shipped model's results in reports/final_summary.json."""
from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import PROJECT_ROOT, load_config
from src.models.curves import approval_cutoff, curve_points
from src.models.scoring import load_config as load_scoring
from src.models.scoring import pd_to_score

SUMMARY = json.loads((PROJECT_ROOT / "reports" / "final_summary.json").read_text(encoding="utf-8"))
CURVES_FILE = PROJECT_ROOT / "reports" / load_config()["api"]["reports"]["test_curves"]


def _toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.07).astype(int)
    pd_ = 1 / (1 + np.exp(-(rng.normal(size=n) + 2.2 * y - 3)))
    return y, pd_


def test_points_are_exact_thresholds_of_the_full_curve():
    y, pd_ = _toy()
    out = curve_points(y, pd_, 200, load_scoring())
    roc, pr = out["roc"], out["pr"]

    assert out["auc"] == pytest.approx(roc_auc_score(y, pd_), abs=1e-5)
    assert out["average_precision"] == pytest.approx(average_precision_score(y, pd_), abs=1e-5)
    assert (roc["fpr"][0], roc["tpr"][0], roc["score"][0]) == (0.0, 0.0, None)
    assert (roc["fpr"][-1], roc["tpr"][-1]) == (1.0, 1.0)
    assert np.all(np.diff(roc["fpr"]) >= 0) and np.all(np.diff(roc["tpr"]) >= 0)
    assert pr["precision"][-1] == pytest.approx(y.mean(), abs=1e-5)  # everyone flagged
    # a point: rejecting everyone at or below its score gives its recall (scores are rounded
    # to integers, so a few rows share the threshold score)
    i = len(pr["score"]) // 2
    rejected = pd_to_score(pd_, load_scoring()) <= pr["score"][i]
    assert y[rejected].sum() / y.sum() == pytest.approx(pr["recall"][i], abs=0.02)
    # trapezoid area under the selected points stays close to the exact AUC
    assert np.trapezoid(roc["tpr"], roc["fpr"]) == pytest.approx(out["auc"], abs=0.01)


def test_operating_point_uses_the_score_cutoff():
    y, pd_ = _toy()
    scoring = load_scoring()
    op = curve_points(y, pd_, 50, scoring)["operating_point"]
    rejected = pd_to_score(pd_, scoring) < approval_cutoff(scoring)

    assert op["cutoff_score"] == approval_cutoff(scoring)
    assert op["rejected"] == int(rejected.sum())
    assert op["recall"] == pytest.approx(y[rejected].sum() / y.sum(), abs=1e-5)
    assert op["precision"] == pytest.approx(y[rejected].mean(), abs=1e-5)


def test_committed_curves_match_the_shipped_model():
    """Fails after a retrain until `python -m src.models.curves` is rerun."""
    curves = json.loads(CURVES_FILE.read_text(encoding="utf-8"))
    test = SUMMARY["test"]

    assert curves["model_file"] == SUMMARY["model_file"]
    assert curves["evaluated_on"] == SUMMARY["evaluated_on"] == "test"
    assert curves["rows"] == test["rows"]
    assert curves["auc"] == pytest.approx(test["auc"], abs=1e-4)
    assert curves["operating_point"]["recall"] == pytest.approx(test["recall"], abs=1e-4)
    assert curves["operating_point"]["precision"] == pytest.approx(test["precision"], abs=1e-4)
    assert curves["pr"]["baseline"] == pytest.approx(SUMMARY["default_rate_test"], abs=1e-4)
    assert 10 <= len(curves["pr"]["recall"]) <= load_config()["api"]["curve_points"]
