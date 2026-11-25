"""DiCE approval paths over the API: POST /recourse (live search) and GET /recourse/summary."""

import json
import sys

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import _APPROVE_EXAMPLE, _REJECT_EXAMPLE
from src.config import PROJECT_ROOT, load_config
from tests.helpers import load_yaml

client = TestClient(app)
RCFG = load_config()["explainer"]["recourse"]
TARGET = load_yaml("scoring.yaml")["recourse_target_score"]
FEATURES = load_yaml("actionability.yaml")["features"]
# Late payments (cannot decrease) plus weak alternative data: score 0, no path even in 24 months
HOPELESS = {
    **_REJECT_EXAMPLE,
    "NumberOfTimes90DaysLate": 3,
    "NumberOfTime30-59DaysPastDueNotWorse": 4,
    "RevolvingUtilizationOfUnsecuredLines": 1.0,
    "insurance_paid_months": 5,
    "spending_consistency": 20.0,
    "telecom_payment_rate": 0.3,
    "utility_payment_rate": 0.3,
    "regular_payment_count": 5,
    "app_login_frequency": 3,
}


@pytest.fixture(scope="module")
def dice_ready():
    """Shipped model and dice-ml (Windows XGBoost cannot read the shipped model, see README)."""
    pytest.importorskip("dice_ml")
    from src.models.predictor import get_model

    try:
        return get_model()
    except Exception as exc:
        if sys.platform == "win32":
            pytest.skip(f"Windows 에서는 배포 모델을 불러올 수 없음, dev 컨테이너에서 실행: {exc}")
        raise


def _recourse(payload: dict, horizon: int) -> dict:
    response = client.post("/recourse", params={"horizon": horizon}, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_recourse_path_follows_the_rules_and_reaches_the_target(dice_ready):
    body = _recourse(_REJECT_EXAMPLE, 12)

    assert body["approval_status"] == "REJECTED" and body["status"] == "found"
    path = body["path"]
    assert 1 <= len(path["changes"]) <= RCFG["max_changed_features"]
    assert path["months"] <= 12 and path["score_after"] >= TARGET
    for c in path["changes"]:
        spec = FEATURES[c["feature"]]
        assert spec["dice_vary"], c["feature"]
        moved = c["proposed_value"] - c["current_value"]
        assert moved > 0 if spec["direction"] == "increase" else moved < 0, c["feature"]
        lo, hi = spec["bounds"]
        value = c["proposed_value"]
        assert (lo is None or value >= lo) and (hi is None or value <= hi)
        assert c["months"] == c["steps"] * spec["months_per_step"] <= 12


def test_recourse_score_matches_predict_with_the_proposed_values(dice_ready):
    """Following the path through /predict gives the score the path promised."""
    path = _recourse(_REJECT_EXAMPLE, 12)["path"]
    proposed = {**_REJECT_EXAMPLE, **{c["feature"]: c["proposed_value"] for c in path["changes"]}}
    if path["linked_debt_ratio"] is not None:  # income rose with debt payments fixed
        proposed["DebtRatio"] = path["linked_debt_ratio"]

    assert client.post("/predict", json=proposed).json()["credit_score"] == path["score_after"]


# Words a layperson would not know; the reasons must avoid them
JARGON = ("SHAP", "log-odds", "DiCE", "단조", "변수")


def test_recourse_explains_why_in_plain_words_with_the_numbers(dice_ready):
    """The path says why it was chosen; each change says what it does to the score."""
    body = _recourse(_REJECT_EXAMPLE, 12)
    path = body["path"]
    texts = [path["summary"]] + [c["why"] for c in path["changes"]]

    assert f"{body['credit_score']}점에서 {path['score_after']}점" in path["summary"]
    assert not [w for w in JARGON for text in texts if w in text]
    for c in path["changes"]:
        assert f"{c['score_if_only_this']}점" in c["why"]
        if c["actionability"] == "TIME_ONLY":  # duration only, never phrased as advice
            assert "시간이 지나면" in c["why"] and "하세요" not in c["why"]
    if len(path["changes"]) == 1:
        assert path["changes"][0]["score_if_only_this"] == path["score_after"]


def test_recourse_shows_approval_probability_and_says_it_is_not_guaranteed(dice_ready):
    from src.models.path_robustness import probability_at

    table = json.loads((PROJECT_ROOT / "reports" / "recourse" / "approval_probability.json")
                       .read_text("utf-8"))
    path = _recourse(_REJECT_EXAMPLE, 12)["path"]
    prob = path["approval_probability"]

    assert prob == round(probability_at(table["curve"], path["score_after"]), 4)
    assert 0 < prob < 1 and path["approval_models"] == table["models"]
    assert "보장되지는 않습니다" in path["summary"]
    assert f"약 {prob * 100:.0f}%" in path["summary"]


def test_every_change_in_a_multi_variable_path_is_needed(dice_ready):
    """Trimming keeps only changes the target needs: drop any one and the score falls short."""
    late = {**_REJECT_EXAMPLE, "NumberOfTimes90DaysLate": 10,
            "NumberOfTime30-59DaysPastDueNotWorse": 10, "NumberOfTime60-89DaysPastDueNotWorse": 8}
    path = _recourse(late, max(RCFG["horizons_months"]))["path"]

    assert path is not None and len(path["changes"]) > 1
    for c in path["changes"]:
        assert c["score_without_this"] < TARGET and c["score_if_only_this"] < TARGET


def test_recourse_is_deterministic(dice_ready):
    assert _recourse(_REJECT_EXAMPLE, 12)["path"] == _recourse(_REJECT_EXAMPLE, 12)["path"]


def test_recourse_says_when_no_path_exists(dice_ready):
    body = _recourse(HOPELESS, max(RCFG["horizons_months"]))

    assert body["status"] == "not_feasible" and body["path"] is None
    assert str(RCFG["max_changed_features"]) in body["note"]
    # The note names what blocks the path: risk factors a path may not change (late payments)
    assert body["blockers"] and all(b in body["note"] for b in body["blockers"])


def test_recourse_skips_approved_applicants(dice_ready):
    body = client.post("/recourse", json=_APPROVE_EXAMPLE).json()

    assert body["status"] == "approved" and body["path"] is None
    assert body["horizon_months"] == load_config()["api"]["recourse_default_horizon"]


def test_recourse_rejects_unknown_horizon():
    response = client.post("/recourse", params={"horizon": 7}, json=_REJECT_EXAMPLE)
    assert response.status_code == 422


def test_recourse_summary_matches_the_dice_report():
    body = client.get("/recourse/summary").json()
    saved = json.loads((PROJECT_ROOT / "reports" / "recourse" / "summary.json").read_text("utf-8"))
    goal = load_config()["performance_targets"]["recourse_coverage_min"]

    assert [h["horizon_months"] for h in body["horizons"]] == sorted(RCFG["horizons_months"])
    for h in body["horizons"]:
        r = saved["horizons"][str(h["horizon_months"])]
        assert h["dice_coverage"] == r["dice_coverage"] and h["rejected"] == r["rejected"]
        assert h["meets_target"] == (r["dice_coverage"] >= goal)
        assert all(p["labels"] and p["count"] > 0 for p in h["top_paths"])
    assert body["target_score"] == TARGET and body["coverage_target"] == goal
    assert body["model_match"] is True and not body["is_dummy"]
    approval = body["approval"]
    assert approval["horizon_months"] == body["default_horizon"]
    assert sum(b["paths"] for b in approval["bins"]) == approval["paths"]
