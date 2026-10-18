"""FastAPI 대출 심사 API 서버 (Swagger UI: /docs).

과제 명세 엔드포인트 7개:
  /health, /predict, /explain, /fairness, /fairness/mitigation, /metrics, /anova
대시보드용 추가: /features (입력 폼 정보), /explain/global (전역 SHAP)

실제 값: 배포 모델(/predict, /explain),
        reports/ 산출물(/metrics, /fairness, /anova, /explain/global).
더미 값: Mitigation 적용 후 지표, ROC/PR 곡선 좌표 → 응답에 is_dummy 로 표시 (src/api/dummy.py).
"""

from __future__ import annotations

import math
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Literal

from fastapi import FastAPI, HTTPException, Query

from src.api import dummy, reports
from src.api.schemas import (
    AnovaResponse,
    CreditApplication,
    ExplainResponse,
    FairnessResponse,
    FeaturesResponse,
    GlobalShapResponse,
    HealthResponse,
    MetricsResponse,
    MitigationResponse,
    PredictResponse,
)
from src.config import get_settings, load_config
from src.utils.logger import get_logger

settings = get_settings()
config = load_config()
logger = get_logger(__name__)

FAIRNESS_NOTE = (
    "배포 모델의 test 세트 점검 결과 (notebooks/audit.py). 성별은 타깃과 독립인 시뮬레이션 값이며, "
    "TPR/FPR 의 양성은 '승인'이다 (TPR = 상환자 중 승인 비율, FPR = 부도자 중 승인 비율)."
)


def _model():
    """배포 모델. 불러올 수 없으면 503."""
    from src.models.predictor import get_model

    try:
        return get_model()
    except Exception as exc:  # 파일 없음, 라이브러리 버전·플랫폼 불일치 등
        logger.error("모델을 불러오지 못했습니다: %s", exc)
        raise HTTPException(
            status_code=503, detail=f"모델을 불러오지 못했습니다 ({settings.model_id}): {exc}"
        ) from exc


def _explainer():
    from src.explainer.shap_explainer import get_explainer

    _model()
    return get_explainer()


def _score(application: CreditApplication):
    model = _model()
    try:
        X = model.prepare([application.model_inputs()])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return model, X, model.predict(X).iloc[0]


def _status(approved) -> str:
    return "APPROVED" if int(approved) == 1 else "REJECTED"


def _now() -> datetime:
    return datetime.now(UTC)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 첫 요청이 느려지지 않도록 모델과 SHAP explainer 를 미리 불러온다
    try:
        _explainer()
        logger.info("모델 로드 완료: %s", settings.model_path.name)
    except Exception:  # 모델이 없어도 지표·공정성·ANOVA 엔드포인트는 동작해야 한다
        logger.warning("모델 없이 시작합니다. /predict, /explain 은 503 을 반환합니다")
    yield


app = FastAPI(
    title="XAI 대안신용평가 API",
    description=(
        "대안 데이터 기반 신용평가, SHAP 설명, 공정성 검증, ANOVA 통계 검증 API\n\n"
        "응답의 `is_dummy` 가 true 인 값은 아직 계산할 수 없어 넣은 예시 값이다.\n\n"
        f"> ⚠️ {config['project']['disclaimer']}"
    ),
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_version=settings.model_id,
        model_available=settings.model_path.exists(),
        timestamp=_now(),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["scoring"],
    summary="신용 점수, 승인 여부, 부도 확률, 리스크 등급",
)
def predict(application: CreditApplication) -> PredictResponse:
    _, _, row = _score(application)
    return PredictResponse(
        customer_id=application.customer_id,
        credit_score=int(row["score"]),
        approval_status=_status(row["approved"]),
        default_probability=round(float(row["pd"]), 6),
        risk_grade=str(row["grade"]),
        is_thin_filer=bool(row["thin_filer"]),
        model_version=settings.model_id,
        timestamp=_now(),
    )


@app.post(
    "/explain",
    response_model=ExplainResponse,
    tags=["xai"],
    summary="SHAP 기반 거절 사유 Top 5 와 변수별 SHAP 값",
)
def explain(application: CreditApplication) -> ExplainResponse:
    from src.explainer.reasons import feature_metadata, rejection_reasons

    model, X, row = _score(application)
    base_value, values = _explainer().explain(X)
    shap_values = {f: round(float(v), 6) for f, v in zip(model.features, values[0], strict=True)}
    feature_values = {f: (None if math.isnan(v) else float(v)) for f, v in X.iloc[0].items()}
    reasons = rejection_reasons(shap_values, feature_values, model.reference_values)
    labels = {
        f: feature_metadata()["features"].get(f, {}).get("label_ko", f) for f in model.features
    }
    return ExplainResponse(
        customer_id=application.customer_id,
        approval_status=_status(row["approved"]),
        credit_score=int(row["score"]),
        risk_grade=str(row["grade"]),
        default_probability=round(float(row["pd"]), 6),
        rejection_reasons=reasons,
        total_rejection_factors=sum(v > 0 for v in shap_values.values()),
        base_value=round(base_value, 6),
        shap_values=shap_values,
        feature_values=feature_values,
        feature_names_kr=labels,
        model_version=settings.model_id,
        timestamp=_now(),
    )


@app.get(
    "/explain/global",
    response_model=GlobalShapResponse,
    tags=["xai"],
    summary="전역 SHAP 중요도 (평균 |SHAP| 상위 변수)",
)
def explain_global() -> GlobalShapResponse:
    from src.explainer.reasons import feature_metadata

    features = feature_metadata()["features"]
    df = reports.read_csv("shap_global")
    rows = [
        {**r, "feature_name_kr": features.get(r["feature"], {}).get("label_ko", r["feature"])}
        for r in df.to_dict(orient="records")
    ]
    return GlobalShapResponse(
        features=rows,
        source=reports.source_of("shap_global"),
        is_dummy=False,
        note=f"모델 보고서의 test 세트 평균 |SHAP| 상위 {len(rows)}개 "
        "(과제 기준 Top 15 는 src/explainer 구현 후 제공)",
    )


@app.get(
    "/features",
    response_model=FeaturesResponse,
    tags=["system"],
    summary="고객 입력 변수 목록 (설명, 범위, 기본값 = 학습 데이터 중앙값)",
)
def features() -> FeaturesResponse:
    from src.api.schemas import CreditApplication as Schema
    from src.explainer.reasons import feature_metadata

    try:
        medians = _model().reference_values
    except HTTPException:
        medians = {}
    meta = feature_metadata()["features"]
    out = []
    for field in Schema.model_fields.values():
        if field.alias is None:  # customer_id
            continue
        column = field.alias
        limits = {type(m).__name__: m for m in field.metadata}
        out.append(
            {
                "feature": column,
                "label_ko": meta[column]["label_ko"],
                "group": "alternative" if meta[column]["source"] == "simulated" else "credit",
                "default": medians.get(column),
                "min": getattr(limits.get("Ge"), "ge", None),
                "max": getattr(limits.get("Le"), "le", None),
                "integer": "int" in str(field.annotation),
                "actionability": meta[column].get("actionability", ""),
            }
        )
    return FeaturesResponse(features=out)


@app.get(
    "/fairness",
    response_model=FairnessResponse,
    tags=["fairness"],
    summary="보호 속성별 DI Ratio 와 Equalized Odds (TPR/FPR 차이)",
)
def fairness(
    attribute: Literal["gender", "age_band"] | None = Query(
        None, description="보호 속성 (기본값: configs/config.yaml 의 fairness.default_attribute)"
    ),
) -> FairnessResponse:
    cfg = config["fairness"]
    audit = reports.read_json("fairness_audit")
    summaries = {a: reports.fairness_summary(a, audit, cfg) for a in ("gender", "age_band")}
    chosen = summaries[attribute or cfg["default_attribute"]]
    return FairnessResponse(
        **chosen,
        attributes=summaries,
        thresholds={
            "di_ratio_bounds": cfg["di_ratio_bounds"],
            "equalized_odds_max_diff": cfg["equalized_odds_max_diff"],
        },
        source=reports.source_of("fairness_audit"),
        is_dummy=False,
        note=FAIRNESS_NOTE,
    )


@app.get(
    "/fairness/mitigation",
    response_model=MitigationResponse,
    tags=["fairness"],
    summary="Bias Mitigation 적용 전/후 공정성 지표 비교",
)
def fairness_mitigation() -> MitigationResponse:
    audit = reports.read_json("fairness_audit")
    before = {
        a: reports.fairness_summary(a, audit, config["fairness"]) for a in ("gender", "age_band")
    }
    return MitigationResponse(
        method="threshold_adjustment",
        before=before,
        after={a: dummy.mitigated(s, config["fairness"]) for a, s in before.items()},
        before_is_dummy=False,
        after_is_dummy=True,
        source=reports.source_of("fairness_audit"),
        note="before 는 실제 점검 결과, after 는 완화 기법(src/fairness) 구현 전까지 쓰는 "
        "예시 값이다 (집단별 승인율·TPR·FPR 을 평균 쪽으로 절반 당긴 값).",
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["model"],
    summary="test 세트 성능 지표 (AUC, KS, Precision, Recall, F1), 혼동행렬, ROC/PR 곡선, 등급표",
)
def metrics() -> MetricsResponse:
    summary = reports.read_json("final_summary")
    t = summary["test"]
    roc, pr = dummy.curves(t["auc"], summary["default_rate_test"], config["api"]["curve_points"])
    curve_note = (
        "test 세트 점수 파일이 없어, 보고된 AUC 와 같은 이항정규 근사 곡선을 그린 예시 값이다"
    )
    grades = reports.read_csv("grade_table")
    return MetricsResponse(
        auc=t["auc"],
        ks=t["ks"],
        precision=t["precision"],
        recall=t["recall"],
        f1=t["f1"],
        auc_ci95=t["auc_ci95"],
        ks_ci95=t["ks_ci95"],
        f1_ci95=t["f1_ci95"],
        psi=t["psi_trainval_vs_test"],
        brier=t["brier"],
        thin_auc=t["thin_auc"],
        thin_auc_gmsc_only=t["thin_auc_gmsc_only"],
        thin_auc_lift=t["thin_auc_lift"],
        approval_rate=summary["approval_rate"],
        default_rate=summary["default_rate_test"],
        rows=t["rows"],
        evaluated_on=summary["evaluated_on"],
        model_version=f"{summary['winner']['model']}_v{summary['model_version']}",
        confusion_matrix={
            **reports.confusion_matrix(summary),
            "note": "반올림된 test 지표(행 수, 부도율, 정밀도, 재현율)로 복원한 근사값 "
            "(칸마다 몇 건 이내 오차)",
        },
        roc_curve={**roc, "is_dummy": True, "note": curve_note},
        pr_curve={**pr, "is_dummy": True, "note": curve_note},
        grade_table=grades.to_dict(orient="records"),
        source=f"{reports.source_of('final_summary')}, {reports.source_of('grade_table')}",
        is_dummy=False,
    )


@app.get(
    "/anova",
    response_model=AnovaResponse,
    tags=["statistics"],
    summary="One-way ANOVA 2종 (피처 그룹별, 알고리즘별): F, p-value, η², Tukey HSD",
)
def anova() -> AnovaResponse:
    from src.statistics.anova import run_tests

    results = run_tests(reports.read_csv("cv_fold_auc"), config["statistics"])
    return AnovaResponse(
        **results,
        source=reports.source_of("cv_fold_auc"),
        is_dummy=False,
        note="train 분할 안의 5 × 5 폴드 AUC (segment = all). 폴드들이 학습 데이터를 공유해 "
        "독립 표본이 아니므로 p-value 는 근사값이다 (reports/anova_handoff_ko.md).",
    )
