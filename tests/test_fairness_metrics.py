"""Fairness metrics (src/fairness/metrics.py): hand-checked examples, plus a check that the module
reproduces the audit figures in reports/audit/2_fairness.json on the real test set."""
from __future__ import annotations

import json
import math

import pytest

from src.fairness.metrics import disparate_impact, fairness_report, group_metrics, max_gap
from tests.helpers import ROOT, load_yaml, require

CFG = {"di_ratio_bounds": [0.8, 1.25], "equalized_odds_max_diff": 0.10}

# Group a: 2 repaid (both approved), 2 defaulted (1 approved) -> approval 0.75, TPR 1.0, FPR 0.5
# Group b: 2 repaid (1 approved),    2 defaulted (0 approved) -> approval 0.25, TPR 0.5, FPR 0.0
Y = [0, 0, 1, 1, 0, 0, 1, 1]
APPROVED = [1, 1, 1, 0, 1, 0, 0, 0]
GROUPS = ["a"] * 4 + ["b"] * 4


def test_group_metrics_match_hand_calculation():
    g = group_metrics(APPROVED, Y, GROUPS)
    assert g["a"]["approval_rate"] == 0.75
    assert g["a"]["tpr_approval_if_repaid"] == 1.0
    assert g["a"]["fpr_approval_if_defaulted"] == 0.5
    assert g["b"]["approval_rate"] == 0.25
    assert g["b"]["tpr_approval_if_repaid"] == 0.5
    assert g["b"]["fpr_approval_if_defaulted"] == 0.0
    assert g["a"]["n"] == g["b"]["n"] == 4
    assert g["a"]["default_rate"] == 0.5


def test_report_flags_unfair_decisions():
    r = fairness_report(APPROVED, Y, GROUPS, CFG, "toy")
    assert r["di_ratio_min_over_max"] == pytest.approx(1 / 3, abs=1e-4)
    assert r["eo_tpr_diff"] == r["eo_fpr_diff"] == r["eo_gap"] == 0.5
    assert not r["di_pass"] and not r["eo_pass"]
    assert len(r["warnings"]) == 3  # DI, TPR gap, FPR gap


def test_identical_groups_pass():
    r = fairness_report(APPROVED[:4] * 2, Y[:4] * 2, GROUPS, CFG)
    assert r["di_ratio_min_over_max"] == 1.0
    assert r["eo_gap"] == 0.0
    assert r["di_pass"] and r["eo_pass"] and r["warnings"] == []


def test_thresholds_are_inclusive():
    # approval 0.8 vs 1.0 -> DI exactly 0.8, which passes
    approved = [1] * 4 + [0] + [1] * 5
    r = fairness_report(approved, [0] * 10, ["a"] * 5 + ["b"] * 5, CFG)
    assert r["di_ratio_min_over_max"] == 0.8 and r["di_pass"]


def test_group_without_defaults_is_skipped_in_fpr_gap():
    # group c has no defaulted applicants, so its FPR is undefined (None in the report)
    r = fairness_report(APPROVED + [1, 1], Y + [0, 0], GROUPS + ["c", "c"], CFG)
    assert r["groups"]["c"]["fpr_approval_if_defaulted"] is None
    assert r["eo_fpr_diff"] == 0.5  # gap between a and b only


def test_helpers_handle_edge_cases():
    assert math.isnan(disparate_impact([0.0, 0.0]))
    assert math.isnan(max_gap([0.3]))
    assert max_gap([0.3, math.nan, 0.1]) == pytest.approx(0.2)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="same length"):
        group_metrics([1, 0], [0], ["a", "b"])
    with pytest.raises(ValueError, match="only 0 and 1"):
        group_metrics([1, 2], [0, 0], ["a", "b"])
    with pytest.raises(ValueError, match="at least 2 groups"):
        fairness_report([1, 0], [0, 1], ["a", "a"], CFG)


def test_reproduces_audit_on_test_set():
    """Same numbers as notebooks/audit.py (reports/audit/2_fairness.json) for the shipped model."""
    require(ROOT / load_yaml("train.yaml")["data"])
    from src.fairness.data import load_split, protected_groups, score, target_column

    audit = json.loads((ROOT / "reports" / "audit" / "2_fairness.json").read_text(encoding="utf-8"))
    test = load_split("test")
    approved = score(test)["approved"]
    groups = protected_groups(test)
    for attribute in ("gender", "age_band"):
        r = fairness_report(approved, test[target_column()], groups[attribute], CFG, attribute)
        a = audit[attribute]
        for key in ("di_ratio_min_over_max", "eo_tpr_diff", "eo_fpr_diff"):
            assert r[key] == pytest.approx(a[key], abs=2e-4), (attribute, key)
        for g, row in a["groups"].items():
            assert r["groups"][g]["n"] == row["n"]
            assert r["groups"][g]["approval_rate"] == pytest.approx(row["approval_rate"], abs=1e-4)
        assert (r["di_pass"], r["eo_pass"]) == (a["di_pass"], a["eo_pass"])
