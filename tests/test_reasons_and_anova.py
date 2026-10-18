"""거절 사유 문장 (조사, 값 표시)과 ANOVA 계산 (모델·데이터 불필요)."""

import numpy as np
import pandas as pd
import pytest

from src.explainer.reasons import explain_sentence, ro_particle, short_label, subject_particle
from src.statistics.anova import eta_squared, one_way_anova


@pytest.mark.parametrize(
    ("word", "particle"),
    [("연령", "이"), ("앱 로그인 빈도", "가"), ("부채비율", "이"), ("정기결제 건수", "가")],
)
def test_subject_particle(word, particle):
    assert subject_particle(word) == particle


@pytest.mark.parametrize(
    ("text", "particle"),
    [
        ("78%", "로"),
        ("5,400", "으로"),
        ("3회", "로"),
        ("2건", "으로"),
        ("12개월", "로"),
        ("73점", "으로"),
        ("45세", "로"),
        ("1", "로"),
    ],
)
def test_ro_particle(text, particle):
    assert ro_particle(text) == particle


def test_sentence_matches_mission_format():
    text = explain_sentence("통신요금 정시납부율(최근 12개월)", 0.78, 0.92, -0.15, {"kind": "pct"})

    assert short_label("통신요금 정시납부율(최근 12개월)") == "통신요금 정시납부율"
    assert (
        text == "통신요금 정시납부율이 78%로, 학습 데이터 중앙값(92%) 대비 낮습니다 (기여도: -0.15)"
    )


def test_sentence_for_counts_flags_and_missing_values():
    late = explain_sentence("90일 이상 연체 횟수", 3, 0, -0.12, {"kind": "count", "unit": "회"})
    flag = explain_sentence("월 소득 결측 표시", 1, 0, -0.2, {"kind": "flag"})
    missing = explain_sentence("월 소득", float("nan"), 5400, -0.05, {"kind": "money"})

    assert (
        late == "90일 이상 연체 횟수가 3회로, 학습 데이터 중앙값(0회) 대비 많습니다 (기여도: -0.12)"
    )
    assert flag.startswith("월 소득 결측 표시에 해당합니다")
    assert "정보가 없어" in missing and "5,400" in missing


def test_eta_squared_matches_definition():
    groups = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    # 집단 간 제곱합 13.5, 전체 제곱합 17.5
    assert eta_squared(groups) == pytest.approx(13.5 / 17.5)


def test_one_way_anova_runs_tukey_only_when_significant():
    rng = np.random.default_rng(0)
    different = pd.DataFrame(
        {
            "g": np.repeat(["a", "b", "c"], 25),
            "auc": np.r_[
                rng.normal(0.80, 0.01, 25), rng.normal(0.85, 0.01, 25), rng.normal(0.90, 0.01, 25)
            ],
        }
    )
    same = different.assign(auc=rng.normal(0.85, 0.01, 75))

    sig = one_way_anova(different, "g", labels={"a": "A", "b": "B", "c": "C"})
    nonsig = one_way_anova(same, "g")
    assert sig["is_significant"] and len(sig["post_hoc_tukey"]) == 3
    assert sig["groups"] == ["A", "B", "C"] and sig["effect_size"] == "large"
    assert not nonsig["is_significant"] and nonsig["post_hoc_tukey"] is None
