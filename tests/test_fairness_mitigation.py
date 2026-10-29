"""Bias mitigation (src/fairness/mitigation.py) on small hand-built examples. No data needed."""
from __future__ import annotations

import numpy as np
import pytest

from src.fairness.metrics import fairness_report
from src.fairness.mitigation import (
    apply_cutoffs,
    base_cutoff,
    reweighing_weights,
    search_cutoffs,
)
from src.models.scoring import load_config as load_scoring

CFG = {"di_ratio_bounds": [0.8, 1.25], "equalized_odds_max_diff": 0.10}


def test_base_cutoff_is_lowest_approved_grade():
    assert base_cutoff(load_scoring()) == 475


def test_reweighing_weights_hand_calculation():
    # group a: 3 repaid, 1 default; group b: 1 repaid, 3 default. Overall P(default) = 0.5
    groups = ["a"] * 4 + ["b"] * 4
    y = [0, 0, 0, 1, 0, 1, 1, 1]
    w = reweighing_weights(groups, y)
    # w(a, repaid) = P(a) P(repaid) / P(a, repaid) = 0.5 * 0.5 / (3/8) = 2/3
    assert w[0] == pytest.approx(2 / 3)
    # w(a, default) = 0.5 * 0.5 / (1/8) = 2
    assert w[3] == pytest.approx(2.0)
    assert w[4] == pytest.approx(2.0) and w[5] == pytest.approx(2 / 3)


def test_reweighing_makes_default_rate_equal_across_groups():
    rng = np.random.default_rng(0)
    groups = rng.choice(["young", "mid", "old"], 2000, p=[0.2, 0.4, 0.4])
    p_default = np.select([groups == "young", groups == "mid"], [0.15, 0.08], 0.03)
    y = (rng.random(2000) < p_default).astype(int)
    w = reweighing_weights(groups, y)
    rates = [np.average(y[groups == g], weights=w[groups == g]) for g in ("young", "mid", "old")]
    assert max(rates) - min(rates) < 1e-9
    assert w.sum() == pytest.approx(len(y))  # total weight is unchanged


def test_apply_cutoffs_uses_each_group_cutoff():
    approved = apply_cutoffs([480, 480, 470, 470], ["a", "b", "a", "b"], {"a": 475, "b": 485})
    assert approved.tolist() == [1, 0, 0, 0]
    with pytest.raises(ValueError, match="no cutoff"):
        apply_cutoffs([500], ["c"], {"a": 475})


def _unfair_example():
    """Group 'old' gets higher scores for the same outcome, so its FPR is far higher."""
    rng = np.random.default_rng(1)
    n = 4000
    groups = rng.choice(["young", "old"], n)
    y = (rng.random(n) < 0.1).astype(int)
    scores = np.where(y == 1, 450, 520) + rng.normal(0, 30, n) + np.where(groups == "old", 25, 0)
    return scores.round(), y, groups


def test_search_finds_cutoffs_that_meet_every_rule():
    scores, y, groups = _unfair_example()
    base = 475
    before = fairness_report((scores >= base).astype(int), y, groups, CFG)
    assert not before["eo_pass"]

    grid = list(range(400, 551, 5))
    result = search_cutoffs(scores, y, groups, CFG, base, grid)
    approved = apply_cutoffs(scores, groups, result["cutoffs"])
    after = fairness_report(approved, y, groups, CFG)
    assert after["eo_pass"] and after["di_pass"]
    # default rate among approved does not go up
    assert y[approved == 1].mean() <= y[scores >= base].mean() + 1e-12
    # the advantaged group needs a higher cutoff than the other
    assert result["cutoffs"]["old"] > result["cutoffs"]["young"]
    assert 0 < result["combinations_feasible"] <= result["combinations_checked"] == len(grid) ** 2


def test_search_raises_when_nothing_is_feasible():
    scores, y, groups = _unfair_example()
    with pytest.raises(ValueError, match="no cutoff combination"):
        search_cutoffs(scores, y, groups, CFG, 475, [475])
