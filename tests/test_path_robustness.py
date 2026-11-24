"""Approval probability of approval paths under retraining (src/models/path_robustness.py)."""

import json

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.models.path_robustness import approval_curve, path_frames, probability_at, score_bins

CFG = load_config()
REPORT = PROJECT_ROOT / "reports" / CFG["api"]["reports"]["recourse_approval"]


def test_curve_never_goes_down_and_stays_in_0_1():
    scores = np.array([495, 496, 497, 500, 505, 510, 530])
    rates = np.array([0.9, 0.7, 0.95, 0.85, 1.0, 0.95, 1.0])  # noisy on purpose
    curve = approval_curve(scores, rates)

    probs = [probability_at(curve, s) for s in range(480, 560)]
    assert all(b >= a for a, b in zip(probs, probs[1:], strict=False))
    assert 0 <= min(probs) and max(probs) <= 1
    assert probability_at(curve, 0) == curve["probability"][0]       # flat below the first point
    assert probability_at(curve, 999) == curve["probability"][-1]    # and above the last


def test_score_bins_count_paths_and_rates():
    scores = np.array([495, 498, 503, 503])
    approved = np.array([[True, False, True, True], [True, True, True, False]])
    bins = score_bins(scores, approved, [495, 500, 510])

    assert [b["paths"] for b in bins] == [2, 2]
    assert bins[0]["approval_rate"] == 0.75 and bins[0]["approved_by_all_models"] == 0.5
    assert bins[1]["min_score"] == 500 and bins[1]["max_score"] == 509


def test_path_frames_apply_only_the_proposed_changes():
    test = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [5.0, 6.0, 7.0]}, index=[10, 11, 12])
    paths = pd.DataFrame(
        {"dice_found": [True, False, True], "new_a": [4.0, np.nan, np.nan],
         "new_b": [np.nan, np.nan, 9.0]},
        index=pd.Index([10, 11, 12], name="row_id"),
    )
    X0, X1, found = path_frames(test, ["a", "b"], paths)

    assert list(found.index) == [10, 12]
    assert X1.loc[10].tolist() == [4.0, 5.0] and X1.loc[12].tolist() == [3.0, 9.0]
    assert X0.loc[10].tolist() == [1.0, 5.0]


def test_committed_table_matches_config_and_the_dice_run():
    """The committed table was built for the dashboard horizon from every DiCE path of that run."""
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    horizon = CFG["api"]["recourse_default_horizon"]
    dice = json.loads((PROJECT_ROOT / "reports" / "recourse" / "summary.json").read_text("utf-8"))
    run = dice["horizons"][str(horizon)]

    assert report["horizon_months"] == horizon
    assert report["models"] == CFG["recourse_approval"]["models"]
    assert report["paths"] == round(run["dice_coverage"] * run["rejected"])
    assert sum(b["paths"] for b in report["bins"]) == report["paths"]
    assert report["curve"]["probability"] == sorted(report["curve"]["probability"])
    overall = report["overall"]
    assert 0 < overall["approved_by_all_models"] <= overall["approval_rate"] <= 1
