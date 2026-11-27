"""One-way ANOVA (F, p-value, η²) 와 Tukey HSD 사후 검정, 보너스 Two-way ANOVA (교호작용).

입력은 5 × 5 폴드 AUC (reports/cv_fold_auc.csv). 검증 정의(집단 열, 필터, 라벨)는
configs/config.yaml 의 statistics.anova / statistics.two_way_anova 에 있고
reports/anova_handoff_ko.md 를 따른다.
주의: 폴드 AUC 는 학습 데이터를 공유하므로 독립 표본이 아니며 p-value 는 근사값이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm
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


def two_way_anova(
    df: pd.DataFrame,
    factors: list[str],
    levels: dict[str, list[str]],
    value_col: str = "auc",
    *,
    pair_on: list[str],
    alpha: float = 0.05,
    eta_bounds: dict | None = None,
    labels: dict | None = None,
    factor_labels: dict | None = None,
    name: str = "",
) -> dict:
    """2 × 2 이원 ANOVA (주효과 2개 + 교호작용, 부분 η²) 와 짝지은 교호작용 대비 t-검정.

    levels 는 요인별 수준 순서다. 교호작용 대비 = 첫 행 수준의 열 효과 - 둘째 행 수준의 열 효과이고,
    열 효과 = 마지막 열 수준 평균 - 첫 열 수준 평균이다. 짝지은 검정은 pair_on (예: repeat, fold)
    이 같은 행끼리 대비를 계산해 그 평균이 0 인지 본다 (같은 모델에서 나온 값의 상관을 지운다).
    """
    eta_bounds = eta_bounds or {"small": 0.01, "medium": 0.06, "large": 0.14}
    labels = labels or {}
    factor_labels = factor_labels or {}
    row, col = factors

    def lab(value: str) -> str:
        return labels.get(value, value)

    # 칸마다 표본 수가 같은 균형 설계이므로 제곱합 유형(I, II, III)에 따라 결과가 달라지지 않는다
    model = smf.ols(f"{value_col} ~ C({row}) * C({col})", data=df).fit()
    table = anova_lm(model, typ=2)
    ss_resid = float(table.loc["Residual", "sum_sq"])
    terms = {
        f"C({row})": factor_labels.get(row, row),
        f"C({col})": factor_labels.get(col, col),
        f"C({row}):C({col})": "교호작용",
    }
    effects = []
    for term, effect in terms.items():
        ss = float(table.loc[term, "sum_sq"])
        partial = ss / (ss + ss_resid)
        size = effect_size(partial, eta_bounds)
        p_value = float(table.loc[term, "PR(>F)"])
        effects.append(
            {
                "effect": effect,
                "term": term.replace("C(", "").replace(")", ""),
                "sum_sq": round(ss, 6),
                "df": int(table.loc[term, "df"]),
                "f_statistic": round(float(table.loc[term, "F"]), 4),
                "p_value": p_value,
                "partial_eta_squared": round(partial, 4),
                "effect_size": size,
                "effect_size_ko": EFFECT_LABELS_KO[size],
                "is_significant": bool(p_value < alpha),
            }
        )

    grouped = df.groupby([row, col])[value_col]
    cell_means = [
        {
            "row": lab(r),
            "column": lab(c),
            "mean": round(float(grouped.get_group((r, c)).mean()), 4),
            "sd": round(float(grouped.get_group((r, c)).std()), 4),
            "n": int(grouped.get_group((r, c)).size),
        }
        for r in levels[row]
        for c in levels[col]
    ]
    first, last = levels[col][0], levels[col][-1]
    simple_effects = {
        lab(r): round(
            float(grouped.get_group((r, last)).mean() - grouped.get_group((r, first)).mean()), 4
        )
        for r in levels[row]
    }

    wide = df.pivot_table(index=pair_on, columns=[row, col], values=value_col)
    lift = {r: wide[(r, last)] - wide[(r, first)] for r in levels[row]}
    contrast = (lift[levels[row][0]] - lift[levels[row][1]]).dropna()
    t_stat, t_p = stats.ttest_1samp(contrast, 0.0)
    paired_check = {
        "description": (
            f"{'·'.join(pair_on)} 이 같은 표본끼리 "
            f"({lab(levels[row][0])} 향상 - {lab(levels[row][1])} 향상) 의 평균이 0 인지 "
            "보는 단일표본 t-검정"
        ),
        "n_pairs": int(contrast.size),
        "mean_contrast": round(float(contrast.mean()), 4),
        "sd_contrast": round(float(contrast.std()), 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(t_p),
        "positive_pairs": int((contrast > 0).sum()),
        "is_significant": bool(t_p < alpha),
    }
    return {
        "test_name": name or f"{row} × {col}별 {value_col} 비교 (Two-way ANOVA)",
        "factors": [factor_labels.get(f, f) for f in factors],
        "levels": {factor_labels.get(f, f): [lab(v) for v in levels[f]] for f in factors},
        "effects": effects,
        "residual_df": int(table.loc["Residual", "df"]),
        "cell_means": cell_means,
        "simple_effects": simple_effects,
        "alpha": alpha,
        "paired_check": paired_check,
    }


def _filter(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    """filters 의 값이 목록이면 그중 하나, 아니면 그 값과 같은 행."""
    for col, value in (filters or {}).items():
        df = df[df[col].isin(value if isinstance(value, list) else [value])]
    return df


def run_tests(cv_fold_auc: pd.DataFrame, stats_cfg: dict) -> dict:
    """configs/config.yaml 의 statistics 설정으로 정의된 모든 ANOVA 검증."""
    out = {}
    for key, test in stats_cfg["anova"].items():
        rows = _filter(cv_fold_auc, test.get("filter"))
        out[key] = one_way_anova(
            rows,
            test["group"],
            alpha=stats_cfg["alpha"],
            eta_bounds=stats_cfg["eta_squared_bounds"],
            labels=test.get("labels"),
            name=test["name"],
        )
        out[key]["data_filter"] = test.get("filter") or {}
    for key, test in (stats_cfg.get("two_way_anova") or {}).items():
        filters = test["filter"]
        out[key] = two_way_anova(
            _filter(cv_fold_auc, filters),
            test["factors"],
            {f: filters[f] for f in test["factors"]},
            pair_on=test["pair_on"],
            alpha=stats_cfg["alpha"],
            eta_bounds=stats_cfg["eta_squared_bounds"],
            labels=test.get("labels"),
            factor_labels=test.get("factor_labels"),
            name=test["name"],
        )
        out[key]["data_filter"] = filters
    return out
