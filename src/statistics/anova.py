"""One-way ANOVA (F, p-value, η²) 와 Tukey HSD 사후 검정.

입력은 5 × 5 폴드 AUC (reports/cv_fold_auc.csv). 검증 정의(집단 열, 필터, 라벨)는
configs/config.yaml 의 statistics.anova 에 있고 reports/anova_handoff_ko.md 를 따른다.
주의: 폴드 AUC 는 학습 데이터를 공유하므로 독립 표본이 아니며 p-value 는 근사값이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

EFFECT_LABELS_KO = {"negligible": "무시할 수준", "small": "작음", "medium": "중간", "large": "큼"}


def eta_squared(groups: list[np.ndarray]) -> float:
    """η² = 집단 간 제곱합 / 전체 제곱합."""
    values = np.concatenate(groups)
    grand = values.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = float(((values - grand) ** 2).sum())
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def effect_size(eta2: float, bounds: dict) -> str:
    if eta2 >= bounds["large"]:
        return "large"
    if eta2 >= bounds["medium"]:
        return "medium"
    if eta2 >= bounds["small"]:
        return "small"
    return "negligible"


def one_way_anova(
    df: pd.DataFrame,
    group_col: str,
    value_col: str = "auc",
    *,
    alpha: float = 0.05,
    eta_bounds: dict | None = None,
    labels: dict | None = None,
    name: str = "",
) -> dict:
    eta_bounds = eta_bounds or {"small": 0.01, "medium": 0.06, "large": 0.14}
    labels = labels or {}
    named = df[group_col].map(lambda g: labels.get(g, g))
    order = [
        labels.get(g, g)
        for g in (labels or sorted(df[group_col].unique()))
        if g in set(df[group_col])
    ]
    groups = [df.loc[named == g, value_col].to_numpy(float) for g in order]
    f_stat, p_value = stats.f_oneway(*groups)
    eta2 = eta_squared(groups)
    significant = bool(p_value < alpha)
    result = {
        "test_name": name or f"{group_col}별 {value_col} 비교 (One-way ANOVA)",
        "group_column": group_col,
        "groups": order,
        "group_means": {g: round(float(v.mean()), 4) for g, v in zip(order, groups, strict=True)},
        "group_sizes": {g: int(len(v)) for g, v in zip(order, groups, strict=True)},
        "f_statistic": round(float(f_stat), 4),
        "p_value": float(p_value),
        "effect_size_eta_squared": round(eta2, 4),
        "effect_size": effect_size(eta2, eta_bounds),
        "effect_size_ko": EFFECT_LABELS_KO[effect_size(eta2, eta_bounds)],
        "alpha": alpha,
        "is_significant": significant,
        "post_hoc_tukey": None,
    }
    if significant:
        tukey = pairwise_tukeyhsd(df[value_col].to_numpy(float), named.to_numpy(), alpha=alpha)
        result["post_hoc_tukey"] = [
            {
                "comparison": f"{g1} vs {g2}",
                "mean_diff": round(float(diff), 4),
                "p_adj": round(float(p_adj), 6),
                "ci_lower": round(float(lo), 4),
                "ci_upper": round(float(hi), 4),
                "significant": bool(reject),
            }
            for g1, g2, diff, p_adj, lo, hi, reject in tukey.summary().data[1:]
        ]
    return result


def run_tests(cv_fold_auc: pd.DataFrame, stats_cfg: dict) -> dict:
    """configs/config.yaml 의 statistics 설정으로 정의된 모든 ANOVA 검증."""
    out = {}
    for key, test in stats_cfg["anova"].items():
        rows = cv_fold_auc
        for col, value in (test.get("filter") or {}).items():
            rows = rows[rows[col] == value]
        out[key] = one_way_anova(
            rows,
            test["group"],
            alpha=stats_cfg["alpha"],
            eta_bounds=stats_cfg["eta_squared_bounds"],
            labels=test.get("labels"),
            name=test["name"],
        )
        out[key]["data_filter"] = test.get("filter") or {}
    return out
