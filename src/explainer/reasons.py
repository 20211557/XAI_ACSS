"""SHAP 기반 거절 사유: 부도 위험을 가장 많이 높인 변수 Top K 를 자연어 문장으로 만든다.

shap_contribution 은 과제 예시와 같이 승인 방향 기준이다 (= -SHAP, 음수 = 거절 쪽으로 기여).
비교 기준(reference_value)은 학습 데이터 중앙값이다.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache

import yaml

from src.config import PROJECT_ROOT, load_config

# 단위별 비교 표현 (값이 기준보다 클 때, 작을 때)
_COMPARE = {
    "회": ("많습니다", "적습니다"),
    "건": ("많습니다", "적습니다"),
    "개": ("많습니다", "적습니다"),
    "명": ("많습니다", "적습니다"),
    "개월": ("깁니다", "짧습니다"),
}
_DEFAULT_COMPARE = ("높습니다", "낮습니다")
# 권장 불가 변수는 조언이 아니라 참고로만 표시한다 (README 팀원 연계, 거절 사유)
NOT_RECOMMENDED_NOTE = " [참고(변경 불가)]"
# 숫자를 한국어로 읽었을 때 받침이 있어 '으로' 가 붙는 끝자리 (영·십·백·천, 삼, 육)
_DIGITS_WITH_BATCHIM = set("036")


@lru_cache
def feature_metadata() -> dict:
    with open(PROJECT_ROOT / "configs" / "actionability.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _batchim(ch: str) -> int | None:
    """한글 음절의 받침 번호 (0 = 받침 없음), 한글이 아니면 None."""
    code = ord(ch) - 0xAC00
    return code % 28 if 0 <= code <= 11171 else None


def subject_particle(word: str) -> str:
    """'이' 또는 '가'."""
    b = _batchim(word.strip()[-1])
    return "이" if b else "가"


def ro_particle(text: str) -> str:
    """'로' 또는 '으로' (받침 없음·ㄹ 받침 → '로')."""
    last = text.strip()[-1]
    b = _batchim(last)
    if b is not None:
        return "로" if b in (0, 8) else "으로"
    if last.isdigit():
        return "으로" if last in _DIGITS_WITH_BATCHIM else "로"
    return "로"  # %, 영문 등


def short_label(label: str) -> str:
    """문장용 이름: 끝의 괄호 설명을 뗀다.

    예: '부채비율(월 부채상환+생활비/월소득)' → '부채비율'
    """
    return re.sub(r"\s*\([^()]*\)\s*$", "", label).strip()


def format_value(value: float | None, fmt: dict) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "정보 없음"
    kind, unit = fmt.get("kind", "decimal"), fmt.get("unit", "")
    if kind == "pct":
        return f"{value * 100:.0f}%"
    if kind == "count":
        return f"{value:.0f}{unit}"
    if kind == "money":
        return f"{value:,.0f}{unit}"
    if kind == "flag":
        return "해당" if value >= 0.5 else "해당 없음"
    return f"{value:.1f}{unit}"


def explain_sentence(
    label: str, value: float | None, reference: float | None, contribution: float, fmt: dict
) -> str:
    name = short_label(label)
    tail = f"(기여도: {contribution:.2f})"
    if fmt.get("kind") == "flag":
        state = "해당합니다" if value is not None and value >= 0.5 else "해당하지 않습니다"
        return f"{name}에 {state} {tail}"
    if value is None or math.isnan(value):
        ref = format_value(reference, fmt)
        return f"{name} 정보가 없어 학습 데이터 중앙값({ref})으로 평가했습니다 {tail}"
    cur, ref = format_value(value, fmt), format_value(reference, fmt)
    head = f"{name}{subject_particle(name)} {cur}{ro_particle(cur)}"
    if reference is None or cur == ref:
        return (
            f"{head}, 학습 데이터 중앙값({ref})과 같은 수준이지만 위험을 높이는 요인입니다 {tail}"
        )
    more, less = _COMPARE.get(fmt.get("unit", ""), _DEFAULT_COMPARE)
    return f"{head}, 학습 데이터 중앙값({ref}) 대비 {more if value > reference else less} {tail}"


def rejection_reasons(
    shap_values: dict[str, float],
    values: dict[str, float | None],
    references: dict[str, float],
    top_k: int | None = None,
) -> list[dict]:
    """부도 위험을 높인 변수(SHAP ≥ 최소 기여도)를 기여도 순으로 최대 top_k 개."""
    cfg = load_config()["explainer"]
    top_k = top_k or cfg["rejection_reasons_top_k"]
    meta = feature_metadata()
    features, classes = meta["features"], meta["classes"]
    floor = cfg["rejection_reason_min_contribution"]
    risky = sorted((f for f, v in shap_values.items() if v >= floor), key=lambda f: -shap_values[f])
    reasons = []
    for rank, feature in enumerate(risky[:top_k], start=1):
        info = features.get(feature, {})
        label = info.get("label_ko", feature)
        action = info.get("actionability", "")
        contribution = -float(shap_values[feature])
        value, reference = values.get(feature), references.get(feature)
        text = explain_sentence(
            label, value, reference, contribution, cfg["value_format"].get(feature, {})
        )
        if action == "NOT_RECOMMENDED":
            text += NOT_RECOMMENDED_NOTE
        reasons.append(
            {
                "rank": rank,
                "feature": feature,
                "feature_name_kr": label,
                "current_value": None if value is None or math.isnan(value) else float(value),
                "reference_value": None if reference is None else float(reference),
                "shap_contribution": round(contribution, 4),
                "explanation": text,
                "actionability": action,
                "actionability_label": classes.get(action, {}).get("label_ko", ""),
            }
        )
    return reasons
