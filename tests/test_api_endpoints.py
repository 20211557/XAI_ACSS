"""과제 명세 API 엔드포인트 7개 (+ 대시보드용 2개): 응답 형식, 실제/더미 표시, 입력 검증."""

import ast
import json
import math
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import _APPROVE_EXAMPLE, _REJECT_EXAMPLE
from src.config import PROJECT_ROOT, load_config

client = TestClient(app)
SUMMARY = json.loads((PROJECT_ROOT / "reports" / "final_summary.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def model_ready():
    """배포 모델. Windows XGBoost 는 macOS 에서 저장한 모델을 읽지 못한다 (README, Windows 주의)."""
    from src.models.predictor import get_model

    try:
        return get_model()
    except Exception as exc:
        if sys.platform == "win32":
            pytest.skip(f"Windows 에서는 배포 모델을 불러올 수 없음, dev 컨테이너에서 실행: {exc}")
        raise


def test_predict_returns_spec_fields_within_2_seconds(model_ready):
    start = time.perf_counter()
    response = client.post("/predict", json=_APPROVE_EXAMPLE)
    elapsed = time.perf_counter() - start

    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["credit_score"] <= 1000
    assert body["approval_status"] in ("APPROVED", "REJECTED")
    assert 0 <= body["default_probability"] <= 1
    assert body["risk_grade"] in list("ABCDE")
    assert elapsed < load_config()["performance_targets"]["predict_latency_sec_max"]


def test_predict_examples_are_approved_and_rejected(model_ready):
    approved = client.post("/predict", json=_APPROVE_EXAMPLE).json()
    rejected = client.post("/predict", json=_REJECT_EXAMPLE).json()

    assert approved["approval_status"] == "APPROVED"
    assert rejected["approval_status"] == "REJECTED"
    assert approved["credit_score"] > rejected["credit_score"]


def test_predict_validates_input():
    too_young = {**_APPROVE_EXAMPLE, "age": 12}
    bad_rate = {**_APPROVE_EXAMPLE, "telecom_payment_rate": 1.5}

    assert client.post("/predict", json=too_young).status_code == 422
    assert client.post("/predict", json=bad_rate).status_code == 422


def test_predict_accepts_missing_values(model_ready):
    """null 은 학습 데이터 중앙값으로 대체되고, 소득 결측은 학습 때처럼 플래그가 된다."""
    payload = {**_APPROVE_EXAMPLE, "MonthlyIncome": None, "telecom_payment_rate": None}
    explained = client.post("/explain", json=payload).json()

    assert explained["feature_values"]["income_missing"] == 1
    assert explained["feature_values"]["DebtRatio"] is None


def test_explain_returns_top5_reasons_with_required_fields(model_ready):
    body = client.post("/explain", json=_REJECT_EXAMPLE).json()
    reasons = body["rejection_reasons"]

    assert body["approval_status"] == "REJECTED"
    assert load_config()["explainer"]["rejection_reasons_min"] <= len(reasons) <= 5
    for r in reasons:
        assert {"feature", "current_value", "shap_contribution", "explanation"} <= set(r)
        assert r["shap_contribution"] < 0 and "기여도" in r["explanation"]
    contributions = [r["shap_contribution"] for r in reasons]
    assert contributions == sorted(contributions)


def test_explain_shap_values_add_up_to_predicted_log_odds(model_ready):
    body = client.post("/explain", json=_REJECT_EXAMPLE).json()
    p = body["default_probability"]

    total = body["base_value"] + sum(body["shap_values"].values())
    assert total == pytest.approx(math.log(p / (1 - p)), abs=1e-3)
    assert len(body["shap_values"]) == 21


def test_fairness_returns_di_and_equalized_odds_from_fairness_module():
    body = client.get("/fairness", params={"attribute": "age_band"}).json()
    metrics = json.loads(
        (PROJECT_ROOT / "reports" / "fairness" / "fairness_metrics.json").read_text(
            encoding="utf-8"
        )
    )

    assert {"di_ratio", "equalized_odds_tpr_diff", "equalized_odds_fpr_diff"} <= set(body)
    assert body["di_ratio"] == metrics["age_band"]["di_ratio_min_over_max"]
    assert set(body["attributes"]) == {"gender", "age_band"}
    assert body["is_dummy"] is False


def test_fairness_warns_when_thresholds_are_violated():
    age = client.get("/fairness", params={"attribute": "age_band"}).json()

    assert age["eo_pass"] is False and age["warnings"]
    assert client.get("/fairness", params={"attribute": "unknown"}).status_code == 422


def test_mitigation_compares_real_before_and_after():
    body = client.get("/fairness/mitigation").json()

    assert body["before_is_dummy"] is False and body["after_is_dummy"] is False
    assert body["method"] == load_config()["fairness"]["mitigation"]["api_method"]
    assert set(body["before"]) == set(body["after"]) == {"gender", "age_band"}
    # the age-band Equalized Odds gap fails before mitigation and passes after
    assert body["before"]["age_band"]["eo_pass"] is False
    assert body["after"]["age_band"]["eo_pass"] is True


def test_model_version_is_consistent_across_endpoints():
    """/health 와 /metrics 가 같은 이름으로 최근 학습 결과(final_summary.json)의 모델을 가리킨다."""
    health = client.get("/health").json()
    metrics = client.get("/metrics").json()
    shipped = Path(SUMMARY["model_file"]).stem

    assert health["model_version"] == metrics["model_version"] == shipped
    assert health["trained_model_version"] == shipped and health["model_version_match"] is True


def test_health_flags_model_version_mismatch(monkeypatch):
    from src.api import main

    monkeypatch.setattr(main.settings, "model_version", "v9.9")
    body = client.get("/health").json()

    assert body["status"] == "healthy" and body["model_version"] == "xgboost_v9.9"
    assert body["model_version_match"] is False


def test_metrics_match_final_summary():
    body = client.get("/metrics").json()
    test = SUMMARY["test"]

    for key in ("auc", "ks", "precision", "recall", "f1"):
        assert body[key] == test[key]
    cm = body["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == test["rows"]
    assert cm["tp"] / (cm["tp"] + cm["fn"]) == pytest.approx(test["recall"], abs=1e-3)
    assert body["roc_curve"]["is_dummy"] is True and body["is_dummy"] is False


def test_anova_returns_both_tests_with_tukey_when_significant():
    body = client.get("/anova").json()

    for key in ("test_1", "test_2"):
        t = body[key]
        assert {"f_statistic", "p_value", "effect_size_eta_squared"} <= set(t)
        assert 0 <= t["effect_size_eta_squared"] <= 1
        assert (t["post_hoc_tukey"] is not None) == t["is_significant"]
    assert body["test_1"]["is_significant"]  # 과제 기준: 최소 1개 검증이 p < 0.05


def test_dashboard_helper_endpoints():
    shap_global = client.get("/explain/global").json()
    features = client.get("/features").json()["features"]

    assert shap_global["features"][0]["rank"] == 1 and not shap_global["is_dummy"]
    assert len(features) == 17
    assert {f["group"] for f in features} == {"credit", "alternative"}


def test_dashboard_never_imports_ml_libraries():
    """MSA: 대시보드는 모델을 직접 로드하지 않는다."""
    banned = {
        "joblib",
        "pickle",
        "sklearn",
        "xgboost",
        "lightgbm",
        "shap",
        "mlflow",
        "src.models",
        "src.explainer",
        "src.api",
    }
    for path in (PROJECT_ROOT / "src" / "dashboard").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported |= {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not {m for m in imported if any(m == b or m.startswith(b + ".") for b in banned)}, (
            path.name
        )
