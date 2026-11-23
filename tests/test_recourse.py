"""DiCE recourse helpers (src/explainer/recourse.py) on hand-built examples. No data needed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.explainer.recourse import (
    Spec,
    apply_steps,
    corner_bound,
    dice_path,
    link_income_debt,
    load_specs,
    max_steps,
    snap_and_trim,
    steps_needed,
)

UP = Spec("a", 1, 5.0, False, 0.0, 100.0, 1.0, 1.0)
DOWN = Spec("b", -1, 0.05, False, 0.0, np.inf, 3.0, 3.0)
GROW = Spec("c", 1, 0.05, True, 0.0, np.inf, 6.0, 3.0)


class LinearScorer:
    """Score = 400 + sum of weight x (value - start): every feature helps in its allowed direction."""

    cfg = {"score": {"max": 1000}}

    def __init__(self, weights: dict[str, float], start: dict[str, float]) -> None:
        self.w, self.start = weights, start

    def _s(self, X):
        return 400 + sum(w * (X[f].to_numpy(float) - self.start[f]) for f, w in self.w.items())

    def score(self, X, X0=None):
        return np.round(self._s(X)).astype(int)

    def raw_score(self, X, X0=None):
        return self._s(X)


def test_specs_match_actionability_yaml():
    specs = load_specs(["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome", "age",
                        "NumberOfTimes90DaysLate", "spending_consistency", "app_login_frequency"])
    # Only dice_vary features: age (immutable), late payments (cannot decrease), login (not recommended) excluded
    assert list(specs) == ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome", "spending_consistency"]
    assert specs["DebtRatio"].direction == -1 and specs["MonthlyIncome"].multiplicative
    assert specs["MonthlyIncome"].step == pytest.approx(0.05)


def test_max_steps_per_horizon():
    assert [max_steps(DOWN, h) for h in (6, 9, 12, 18, 24)] == [2, 3, 4, 6, 8]
    assert [max_steps(GROW, h) for h in (6, 9, 12, 18, 24)] == [1, 1, 2, 3, 4]


def test_apply_steps_direction_and_bounds():
    assert apply_steps(98.0, UP, 3) == 100.0                    # clipped at the upper bound
    assert apply_steps(0.07, DOWN, 4) == 0.0                    # clipped at 0
    assert apply_steps(150.0, UP, 2) == 150.0                   # above the bound: never moves backwards
    assert apply_steps(1000.0, GROW, 2) == pytest.approx(1102.5)
    assert np.isnan(apply_steps(np.nan, UP, 2))


def test_steps_needed_rounds_up():
    assert steps_needed(50.0, 61.0, UP) == 3                     # 11 points = 2.2 steps -> 3
    assert steps_needed(50.0, 60.0, UP) == 2
    assert steps_needed(0.5, 0.42, DOWN) == 2                    # 0.08 = 1.6 steps -> 2
    assert steps_needed(1000.0, 1060.0, GROW) == 2               # 1.05^1 = 1050 < 1060
    assert steps_needed(50.0, 40.0, UP) == 0                     # wrong direction = no change


def test_link_income_debt_inverse():
    X0 = pd.DataFrame({"MonthlyIncome": [1000.0, np.nan, 0.0], "DebtRatio": [0.5, np.nan, 0.3]})
    X = X0.assign(MonthlyIncome=[1100.0, np.nan, 0.0], DebtRatio=[0.45, np.nan, 0.3])
    out = link_income_debt(X, X0)
    # Debt fixed: ratio scales by old income / new income
    assert out["DebtRatio"].iloc[0] == pytest.approx(0.45 * 1000 / 1100)
    assert np.isnan(out["DebtRatio"].iloc[1]) and out["DebtRatio"].iloc[2] == 0.3


def _four_specs():
    return {n: Spec(n, 1, 1.0, False, 0.0, 100.0, 1.0, 1.0) for n in "wxyz"}


def test_snap_and_trim_limits_variables_and_steps():
    specs = _four_specs()
    x0 = pd.Series({n: 0.0 for n in "wxyz"})
    scorer = LinearScorer({"w": 10, "x": 10, "y": 10, "z": 1}, x0.to_dict())
    # DiCE proposal moves all four variables; the target needs 95 points
    path = snap_and_trim(scorer, x0, {"w": 4.3, "x": 4, "y": 4, "z": 9}, specs, 12, 495, 3)
    assert path is not None and len(path["steps"]) <= 3
    assert "z" not in path["steps"]                              # weakest change is dropped first
    assert path["score"] >= 495
    # Fewest steps: removing any single step falls below the target
    for v, n in path["steps"].items():
        assert scorer.score(pd.DataFrame([{**x0, **{u: m for u, m in path["steps"].items()}, v: n - 1}]))[0] < 495


def test_snap_and_trim_rejects_when_four_variables_needed():
    specs = _four_specs()
    x0 = pd.Series({n: 0.0 for n in "wxyz"})
    scorer = LinearScorer({n: 10 for n in "wxyz"}, x0.to_dict())
    # Each variable can add at most 3 steps x 10 = 30 points within 3 months: 95 points need all four
    assert snap_and_trim(scorer, x0, {n: 3 for n in "wxyz"}, specs, 3, 495, 3) is None


def test_corner_bound_counts_fewest_variables():
    specs = _four_specs()
    X = pd.DataFrame([{n: 0.0 for n in "wxyz"}] * 2)
    scorer = LinearScorer({"w": 10, "x": 10, "y": 0, "z": 0}, {n: 0.0 for n in "wxyz"})
    cb = corner_bound(scorer, X, specs, 6, 495, 3)                # 6 steps x 10 = 60 per variable
    assert cb["corner_feasible"].all() and (cb["corner_min_vars"] == 2).all()
    cb = corner_bound(scorer, X, specs, 4, 495, 3)                # 2 x 40 = 80 < 95
    assert not cb["corner_feasible"].any() and not cb["corner_all_vars"].any()


def test_dice_path_respects_direction_and_horizon():
    pytest.importorskip("dice_ml")
    specs = {"x": Spec("x", 1, 1.0, False, 0.0, 100.0, 1.0, 1.0),
             "y": Spec("y", -1, 1.0, False, 0.0, 100.0, 2.0, 2.0)}
    x0 = pd.Series({"x": 10.0, "y": 50.0})
    scorer = LinearScorer({"x": 10, "y": -10}, x0.to_dict())
    rcfg = {"max_changed_features": 3, "init_batch": 500, "init_max_samples": 5000,
            "dice": {"total_cfs": 3, "maxiterations": 50, "proximity_weight": 0.2, "sparsity_weight": 0.2,
                     "diversity_weight": 5.0}}
    path = dice_path(scorer, x0, specs, 12, 495, rcfg, seed=0)
    assert path is not None and path["score"] >= 495
    assert path["values"].get("x", 10.0) >= 10.0 and path["values"].get("y", 50.0) <= 50.0
    assert path["values"].get("x", 10.0) <= 22.0 and path["values"].get("y", 50.0) >= 44.0   # 12 and 6 steps
    assert path["months"] <= 12
