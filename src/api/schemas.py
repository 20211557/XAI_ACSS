"""API Request/Response Pydantic 스키마.

고객 입력 JSON 의 키는 모델 학습 데이터의 컬럼명이다 (예: NumberOfTimes90DaysLate).
설명과 허용 범위는 configs/actionability.yaml, 최소 연령은 configs/data.yaml 에서 읽는다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic.fields import FieldInfo

from src.config import PROJECT_ROOT

with open(PROJECT_ROOT / "configs" / "actionability.yaml", encoding="utf-8") as _f:
    _FEATURES = yaml.safe_load(_f)["features"]
with open(PROJECT_ROOT / "configs" / "data.yaml", encoding="utf-8") as _f:
    _MIN_AGE = yaml.safe_load(_f)["min_age"]

_PAST_DUE_MAX = 98  # 원본 데이터의 특수코드 96/98 도 입력 가능 (정제 규칙이 결측 + 플래그로 바꿈)


def _field(
    column: str,
    *,
    required: bool = False,
    ge: float | None = 0,
    le: float | None = None,
    use_bounds: bool = True,
) -> FieldInfo:
    """컬럼명 alias, 한국어 설명, 허용 범위(actionability.yaml bounds)를 가진 필드."""
    meta = _FEATURES[column]
    bounds = meta.get("bounds") if use_bounds else None
    if bounds:
        ge = bounds[0] if bounds[0] is not None else ge
        le = bounds[1] if bounds[1] is not None else le
    return Field(
        ... if required else None, alias=column, description=meta["label_ko"], ge=ge, le=le
    )


_APPROVE_EXAMPLE = {
    "customer_id": "C001234",
    "RevolvingUtilizationOfUnsecuredLines": 0.12,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.25,
    "MonthlyIncome": 6500,
    "NumberOfOpenCreditLinesAndLoans": 7,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1,
    "telecom_payment_rate": 0.92,
    "utility_payment_rate": 0.92,
    "telecom_tenure_months": 60,
    "insurance_paid_months": 22,
    "spending_consistency": 78.0,
    "regular_payment_count": 16,
    "app_login_frequency": 25,
}
_REJECT_EXAMPLE = {
    "customer_id": "C005678",
    "RevolvingUtilizationOfUnsecuredLines": 0.95,
    "age": 27,
    "NumberOfTime30-59DaysPastDueNotWorse": 2,
    "DebtRatio": 0.6,
    "MonthlyIncome": 2400,
    "NumberOfOpenCreditLinesAndLoans": 2,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 0,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 0,
    "telecom_payment_rate": 0.75,
    "utility_payment_rate": 0.75,
    "telecom_tenure_months": 36,
    "insurance_paid_months": 17,
    "spending_consistency": 68.0,
    "regular_payment_count": 14,
    "app_login_frequency": 22,
}


class CreditApplication(BaseModel):
    """고객 정보. 값이 없으면 null (학습 데이터 중앙값으로 대체)."""

    model_config = ConfigDict(
        populate_by_name=True, json_schema_extra={"examples": [_APPROVE_EXAMPLE, _REJECT_EXAMPLE]}
    )

    customer_id: str | None = Field(None, examples=["C001234"])
    revolving_utilization: float | None = _field(
        "RevolvingUtilizationOfUnsecuredLines", use_bounds=False
    )
    age: int = _field("age", required=True, ge=_MIN_AGE, le=120)
    past_due_30_59: int | None = _field("NumberOfTime30-59DaysPastDueNotWorse", le=_PAST_DUE_MAX)
    debt_ratio: float | None = _field("DebtRatio")
    monthly_income: float | None = _field("MonthlyIncome")
    open_credit_lines: int | None = _field("NumberOfOpenCreditLinesAndLoans")
    times_90_days_late: int | None = _field("NumberOfTimes90DaysLate", le=_PAST_DUE_MAX)
    real_estate_loans: int | None = _field("NumberRealEstateLoansOrLines")
    past_due_60_89: int | None = _field("NumberOfTime60-89DaysPastDueNotWorse", le=_PAST_DUE_MAX)
    dependents: int | None = _field("NumberOfDependents")
    telecom_payment_rate: float | None = _field("telecom_payment_rate")
    utility_payment_rate: float | None = _field("utility_payment_rate")
    telecom_tenure_months: int | None = _field("telecom_tenure_months")
    insurance_paid_months: int | None = _field("insurance_paid_months")
    spending_consistency: float | None = _field("spending_consistency")
    regular_payment_count: int | None = _field("regular_payment_count")
    app_login_frequency: int | None = _field("app_login_frequency")

    def model_inputs(self) -> dict:
        """모델 컬럼명 → 값 (customer_id 제외)."""
        return self.model_dump(by_alias=True, exclude={"customer_id"})


ApprovalStatus = Literal["APPROVED", "REJECTED"]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., examples=["healthy"])
    model_version: str = Field(..., examples=["xgboost_v1.0"])
    model_available: bool = Field(..., description="model_version 에 해당하는 모델 파일 존재 여부")
    trained_model_version: str | None = Field(
        None,
        examples=["xgboost_v1.0"],
        description="최근 학습 결과(reports/final_summary.json)의 배포 모델",
    )
    model_version_match: bool | None = Field(
        None,
        description="model_version 과 trained_model_version 이 같은지 (학습 결과가 없으면 null)",
    )
    timestamp: datetime


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    customer_id: str | None = None
    credit_score: int = Field(..., ge=0, le=1000, examples=[687])
    approval_status: ApprovalStatus
    default_probability: float = Field(..., ge=0, le=1, examples=[0.12])
    risk_grade: str = Field(..., examples=["B"])
    is_thin_filer: bool = Field(
        ..., description="씬파일러 (대리 기준: 대출·신용한도 2개 이하, 부동산 담보대출 없음)"
    )
    model_version: str
    timestamp: datetime


class RejectionReason(BaseModel):
    rank: int
    feature: str
    feature_name_kr: str
    current_value: float | None
    reference_value: float | None = Field(None, description="학습 데이터 중앙값")
    shap_contribution: float = Field(
        ..., description="승인 방향 기준 기여도 (log-odds, 음수 = 거절 쪽)"
    )
    explanation: str
    actionability: str = Field(
        ..., description="변수별 조치 가능성 분류 (configs/actionability.yaml)"
    )
    actionability_label: str


class ExplainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    customer_id: str | None = None
    approval_status: ApprovalStatus
    credit_score: int
    risk_grade: str
    default_probability: float
    rejection_reasons: list[RejectionReason]
    total_rejection_factors: int = Field(..., description="부도 위험을 높인 변수 수 (SHAP > 0)")
    base_value: float = Field(..., description="SHAP 기준값 (부도 log-odds)")
    shap_values: dict[str, float] = Field(
        ...,
        description="변수별 SHAP (부도 log-odds, 양수 = 위험 증가). base_value + 합 = logit(PD)",
    )
    feature_values: dict[str, float | None] = Field(..., description="정제 후 모델 입력값")
    feature_names_kr: dict[str, str]
    model_version: str
    timestamp: datetime


class GroupMetrics(BaseModel):
    n: int
    default_rate: float
    approval_rate: float
    approval_ci95: list[float] | None = None
    tpr: float = Field(..., description="실제 상환자 중 승인 비율")
    fpr: float = Field(..., description="실제 부도자 중 승인 비율")


class FairnessSummary(BaseModel):
    attribute: str
    di_ratio: float = Field(..., description="집단 승인율 최소 / 최대")
    equalized_odds_tpr_diff: float
    equalized_odds_fpr_diff: float
    di_pass: bool
    eo_pass: bool
    groups: dict[str, GroupMetrics]
    warnings: list[str]


class FairnessThresholds(BaseModel):
    di_ratio_bounds: list[float]
    equalized_odds_max_diff: float


class FairnessResponse(BaseModel):
    attribute: str
    di_ratio: float
    equalized_odds_tpr_diff: float
    equalized_odds_fpr_diff: float
    di_pass: bool
    eo_pass: bool
    groups: dict[str, GroupMetrics]
    warnings: list[str]
    attributes: dict[str, FairnessSummary] = Field(
        ..., description="모든 보호 속성 (gender, age_band)"
    )
    thresholds: FairnessThresholds
    source: str
    is_dummy: bool
    note: str


class MitigationResponse(BaseModel):
    method: str
    before: dict[str, FairnessSummary]
    after: dict[str, FairnessSummary]
    before_is_dummy: bool
    after_is_dummy: bool
    source: str
    note: str


class ConfusionMatrix(BaseModel):
    tn: int
    fp: int
    fn: int
    tp: int
    positive_label: str = "부도 (거절)"
    is_dummy: bool = False
    note: str


class RocCurve(BaseModel):
    fpr: list[float]
    tpr: list[float]
    is_dummy: bool
    note: str


class PrCurve(BaseModel):
    recall: list[float]
    precision: list[float]
    baseline: float
    is_dummy: bool
    note: str


class GradeRow(BaseModel):
    grade: str
    rows: int
    share: float
    mean_pd: float
    default_rate: float
    min_score: int
    max_score: int


class MetricsResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    auc: float
    ks: float
    precision: float
    recall: float
    f1: float
    auc_ci95: list[float]
    ks_ci95: list[float]
    f1_ci95: list[float]
    psi: float
    brier: float
    thin_auc: float
    thin_auc_gmsc_only: float
    thin_auc_lift: float
    approval_rate: float
    default_rate: float
    rows: int
    evaluated_on: str
    model_version: str
    confusion_matrix: ConfusionMatrix
    roc_curve: RocCurve
    pr_curve: PrCurve
    grade_table: list[GradeRow]
    source: str
    is_dummy: bool


class TukeyRow(BaseModel):
    comparison: str
    mean_diff: float
    p_adj: float
    ci_lower: float
    ci_upper: float
    significant: bool


class AnovaResult(BaseModel):
    test_name: str
    group_column: str
    groups: list[str]
    group_means: dict[str, float]
    group_sizes: dict[str, int]
    f_statistic: float
    p_value: float
    effect_size_eta_squared: float
    effect_size: str
    effect_size_ko: str
    alpha: float
    is_significant: bool
    post_hoc_tukey: list[TukeyRow] | None
    data_filter: dict[str, str]


class AnovaResponse(BaseModel):
    test_1: AnovaResult
    test_2: AnovaResult
    source: str
    is_dummy: bool
    note: str


class GlobalShapFeature(BaseModel):
    rank: int
    feature: str
    feature_name_kr: str
    mean_abs_shap: float
    actionability: str


class GlobalShapResponse(BaseModel):
    features: list[GlobalShapFeature]
    source: str
    is_dummy: bool
    note: str


class FeatureInfo(BaseModel):
    feature: str
    label_ko: str
    group: Literal["credit", "alternative"]
    default: float | None = Field(None, description="학습 데이터 중앙값 (입력 폼 기본값)")
    min: float | None = None
    max: float | None = None
    integer: bool
    actionability: str


class FeaturesResponse(BaseModel):
    features: list[FeatureInfo]
