"""FastAPI 대출 심사 API 서버 (Swagger UI: /docs).

과제 명세 엔드포인트 7개:
  /health, /predict, /explain, /fairness, /fairness/mitigation, /metrics, /anova
대시보드용 추가: /features (입력 폼 정보), /explain/global (전역 SHAP),
               /explain/global/summary-plot (SHAP Summary Plot 이미지),
               /recourse (DiCE 승인 경로), /recourse/summary (기간별 경로 커버리지)

실제 값: 배포 모델(/predict, /explain, /recourse),
        reports/ 산출물(/metrics, /fairness, /fairness/mitigation, /anova, /explain/global,
        /recourse/summary).
더미 값: reports/test_curves.json 이 없거나 배포 모델과 다를 때의 ROC/PR 곡선만 예시 값으로
        그리고 is_dummy 로 표시한다 (src/api/dummy.py).
"""

from __future__ import annotations

import math
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from src.api import dummy, reports
from src.api import recourse as recourse_service
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
    RecourseResponse,
    RecourseSummaryResponse,
)
from src.config import get_settings, load_config
from src.utils.logger import get_logger

settings = get_settings()
config = load_config()
logger = get_logger(__name__)

FAIRNESS_NOTE = (
    "배포 모델의 test 세트 공정성 검증 결과 (python -m src.fairness). 성별은 타깃과 독립인 "
    "시뮬레이션 값이며, TPR/FPR 의 양성은 '승인'이다 "
    "(TPR = 상환자 중 승인 비율, FPR = 부도자 중 승인 비율)."
)
MITIGATION_LABELS = {
    "reweighting": "가중치 재학습 (Reweighting)",
    "threshold_adjustment": "연령대별 승인 기준점 (Threshold Adjustment)",
}


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
    try:
        recourse_service.warm_up()
    except Exception as exc:  # dice-ml 이 없거나 모델이 없으면 /recourse 만 실패한다
        logger.warning("DiCE 경로 탐색을 준비하지 못했습니다: %s", exc)
    yield


app = FastAPI(
    title="XAI 대안신용평가 API",
    description=(
        "대안 데이터 기반 신용평가, SHAP 설명, 공정성 검증, ANOVA 통계 검증 API\n\n"
        "응답의 `is_dummy` 가 true 인 값은 아직 계산할 수 없어 넣은 예시 값이다.\n\n"
        f"> ⚠️ {config['project']['disclaimer']}"
    ),
    version="0.3.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    trained = reports.trained_model_id()
    return HealthResponse(
        status="healthy",
        model_version=settings.model_id,
        model_available=settings.model_path.exists(),
        trained_model_version=trained,
        model_version_match=None if trained is None else trained == settings.model_id,
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
        note=f"모델 보고서의 test 세트 평균 |SHAP| 상위 {len(rows)}개. 과제 기준 Summary Plot "
        "(상위 15개) 이미지는 GET /explain/global/summary-plot "
        "(reports/figures/13_shap_summary_ko.png, src/explainer/global_shap.py)",
    )


@app.get(
    "/explain/global/summary-plot",
    response_class=FileResponse,
    tags=["xai"],
    summary="SHAP Summary Plot 이미지 (test 세트, 상위 15개 변수, PNG)",
    responses={
        200: {"content": {"image/png": {}}, "description": "PNG 이미지"},
        404: {"description": "그림 파일이 없음 (python -m src.explainer.global_shap 으로 생성)"},
    },
)
def explain_global_summary_plot(
    lang: Literal["ko", "en"] = Query("ko", description="그림 언어"),
) -> FileResponse:
    path = reports.figure_path("shap_summary_plot", lang=lang)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{reports.source_of('shap_summary_plot', lang=lang)} 가 없습니다 "
            "(python -m src.explainer.global_shap 으로 생성)",
        )
    return FileResponse(path, media_type="image/png")


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


@app.post(
    "/recourse",
    response_model=RecourseResponse,
    tags=["xai"],
    summary="DiCE 승인 경로 (거절 고객이 바꿀 수 있는 변수만, 기간 안에)",
)
def recourse(
    application: CreditApplication,
    horizon: int | None = Query(
        None,
        description="경로 기간 (개월). configs/config.yaml 의 explainer.recourse.horizons_months "
        "중 하나. 비우면 api.recourse_default_horizon",
    ),
) -> RecourseResponse:
    rules = recourse_service.rules()
    horizon = horizon or config["api"]["recourse_default_horizon"]
    if horizon not in rules["horizons"]:
        raise HTTPException(
            status_code=422, detail=f"horizon 은 {rules['horizons']} 중 하나여야 합니다"
        )
    model, X, row = _score(application)
    cutoff = recourse_service.approval_cutoff(model.scoring_cfg)
    common = {
        "customer_id": application.customer_id,
        "approval_status": _status(row["approved"]),
        "credit_score": int(row["score"]),
        "horizon_months": horizon,
        "target_score": rules["target"],
        "approval_cutoff": cutoff,
        "max_changed_features": rules["max_vars"],
        "model_version": settings.model_id,
        "timestamp": _now(),
    }
    if int(row["approved"]) == 1:
        return RecourseResponse(
            **common, status="approved", elapsed_seconds=0.0,
            note="이미 승인된 고객이라 경로를 찾지 않습니다.",
        )
    # 이 고객의 SHAP: 경로 변수가 지금 위험을 얼마나 높이는지, 경로가 없을 때 무엇이 막는지 설명
    _, values = _explainer().explain(X)
    shap = {f: float(v) for f, v in zip(model.features, values[0], strict=True)}
    try:
        result = recourse_service.find_path(X, horizon, shap=shap, cutoff=cutoff)
    except ImportError as exc:  # dice-ml 미설치
        raise HTTPException(status_code=503, detail=f"DiCE 를 쓸 수 없습니다: {exc}") from exc
    blockers = ", ".join(result["blockers"])
    notes = {
        "found": f"점수 {rules['target']}점 (승인 {cutoff}점 + 여유) 을 목표로, 바꿀 수 있는 "
        f"변수 {rules['max_vars']}개 이하를 {horizon}개월 안에 바꾸는 경로입니다. 변수는 동시에 "
        "바꾸므로 전체 기간은 가장 오래 걸리는 변수의 기간입니다.",
        "not_feasible": f"{horizon}개월 동안 바꿀 수 있는 항목을 {rules['max_vars']}가지까지 "
        f"최대한 좋게 바꿔도 {rules['target']}점에 닿지 못합니다."
        + (f" 점수를 가장 많이 깎는 항목 ({blockers}) 이 경로에 넣지 않는 것들이기 "
           "때문입니다. 바꿀 수 없거나, 점수만 올리는 꼼수가 될 수 있어 일부러 뺀 항목입니다."
           if blockers else ""),
        "not_found": "목표에 닿는 조합은 있지만 DiCE 가 경로를 찾지 못했습니다.",
    }
    return RecourseResponse(**common, **result, note=notes[result["status"]])


@app.get(
    "/recourse/summary",
    response_model=RecourseSummaryResponse,
    tags=["xai"],
    summary="기간별 승인 경로 커버리지 (test 세트 거절자, 미리 계산)",
)
def recourse_summary() -> RecourseSummaryResponse:
    try:
        body = recourse_service.summary(
            config["api"]["recourse_default_horizon"],
            config["performance_targets"]["recourse_coverage_min"],
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"{reports.source_of('recourse_summary')} 가 없습니다 "
            "(python -m src.explainer.recourse 로 생성)",
        ) from exc
    note = (
        "배포 모델이 test 세트에서 거절한 신청자 전체에 DiCE 를 돌린 결과 "
        "(python -m src.explainer.recourse)."
    )
    if body["model_match"] is False:
        note += " 거절자 수가 지금 배포 모델과 다릅니다. 경로를 다시 계산해야 합니다."
    return RecourseSummaryResponse(
        **body, source=reports.source_of("recourse_summary"), is_dummy=False, note=note
    )


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
    metrics = reports.read_json("fairness_metrics")
    summaries = {a: reports.fairness_summary(a, metrics, cfg) for a in ("gender", "age_band")}
    chosen = summaries[attribute or cfg["default_attribute"]]
    return FairnessResponse(
        **chosen,
        attributes=summaries,
        thresholds={
            "di_ratio_bounds": cfg["di_ratio_bounds"],
            "equalized_odds_max_diff": cfg["equalized_odds_max_diff"],
        },
        source=reports.source_of("fairness_metrics"),
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
    cfg = config["fairness"]
    method = cfg["mitigation"]["api_method"]
    result = reports.read_json("fairness_mitigation")
    attributes = ("gender", "age_band")
    before = {a: reports.fairness_summary(a, result["before"]["fairness"], cfg) for a in attributes}
    chosen = result["methods"][method]
    after = {a: reports.fairness_summary(a, chosen["fairness"], cfg) for a in attributes}
    p0, p1 = result["before"]["performance"], chosen["performance"]
    return MitigationResponse(
        method=method,
        before=before,
        after=after,
        before_is_dummy=False,
        after_is_dummy=False,
        source=reports.source_of("fairness_mitigation"),
        note=(
            f"test 세트, 완화 대상 {result['attribute']}. 적용 후 = {MITIGATION_LABELS[method]}. "
            f"AUC {p0['auc']:.4f} → {p1['auc']:.4f}, 승인율 {p0['approval_rate']:.2%} → "
            f"{p1['approval_rate']:.2%}, 승인자 부도율 {p0['default_rate_approved']:.2%} → "
            f"{p1['default_rate_approved']:.2%}. 다른 방법의 결과도 같은 파일에 있다."
        ),
    )


def _curves(summary: dict) -> tuple[dict, dict]:
    """ROC / PR 곡선: 배포 모델의 test 세트 실제 점수 (reports/test_curves.json).

    파일이 없거나 다른 모델의 곡선이면 보고된 AUC 와 같은 이항정규 근사 곡선(더미)을 돌려준다."""
    try:
        curves = reports.read_json("test_curves")
    except FileNotFoundError:
        curves = None
    if curves is None or curves.get("model_file") != summary["model_file"]:
        auc, prevalence = summary["test"]["auc"], summary["default_rate_test"]
        roc, pr = dummy.curves(auc, prevalence, config["api"]["curve_points"])
        reason = "곡선 파일이 없어" if curves is None else "곡선 파일이 배포 모델과 달라"
        note = (
            f"{reason}, 보고된 AUC 와 같은 이항정규 근사 곡선을 그린 예시 값이다 "
            "(python -m src.models.curves 로 다시 만든다)"
        )
        return {**roc, "is_dummy": True, "note": note}, {**pr, "is_dummy": True, "note": note}
    source = reports.source_of("test_curves")
    op = curves["operating_point"]
    note = (
        f"{curves['evaluated_on']} 세트 {curves['rows']:,}명의 실제 점수 (배포 모델). "
        f"양성 = 부도 = 거절, 점 {len(curves['pr']['recall'])}개. "
        f"◆ = 승인 기준 {op['cutoff_score']}점"
    )
    common = {"operating_point": op, "source": source, "is_dummy": False, "note": note}
    roc = {**curves["roc"], "auc": curves["auc"], **common}
    pr = {**curves["pr"], "average_precision": curves["average_precision"], **common}
    return roc, pr


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["model"],
    summary="test 세트 성능 지표 (AUC, KS, Precision, Recall, F1), 혼동행렬, ROC/PR 곡선, 등급표",
)
def metrics() -> MetricsResponse:
    summary = reports.read_json("final_summary")
    t = summary["test"]
    roc, pr = _curves(summary)
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
        model_version=reports.model_id(summary),
        confusion_matrix={
            **reports.confusion_matrix(summary),
            "note": "반올림된 test 지표(행 수, 부도율, 정밀도, 재현율)로 복원한 근사값 "
            "(칸마다 몇 건 이내 오차)",
        },
        roc_curve=roc,
        pr_curve=pr,
        grade_table=grades.to_dict(orient="records"),
        source=f"{reports.source_of('final_summary')}, {reports.source_of('grade_table')}",
        is_dummy=False,
    )


@app.get(
    "/anova",
    response_model=AnovaResponse,
    tags=["statistics"],
    summary="One-way ANOVA 2종 (피처 그룹별, 알고리즘별): F, p-value, η², Tukey HSD. "
    "보너스 test_3: 세그먼트 × 피처 구성 Two-way ANOVA (교호작용)",
)
def anova() -> AnovaResponse:
    from src.statistics.anova import run_tests

    results = run_tests(reports.read_csv("cv_fold_auc"), config["statistics"])
    return AnovaResponse(
        **results,
        source=reports.source_of("cv_fold_auc"),
        is_dummy=False,
        note="train 분할 안의 5 × 5 폴드 AUC. 검증 1, 2 는 segment = all 이고 "
        "검증 2 는 통합 피처끼리 비교한다. 검증 3 은 XGBoost (리샘플링 없음) 의 "
        "씬파일러·일반 AUC 이며, 같은 폴드의 두 값이 같은 모델에서 나오므로 "
        "폴드별 짝지은 교호작용 검정을 함께 보고한다. 폴드들이 학습 데이터를 공유해 "
        "독립 표본이 아니므로 p-value 는 근사값이다 (reports/anova_report_ko.md).",
    )
