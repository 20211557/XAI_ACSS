"""더미 데이터: 아직 실제로 계산할 수 없는 값. 응답에는 is_dummy=True 로 표시한다.

- Bias Mitigation 적용 후 공정성 지표: src/fairness 의 완화 기법이 아직 없다 (체민규 담당)
- ROC / PR 곡선 좌표: test 세트 점수 파일(data/processed/final_test_scores.csv)이 레포에 없다
실제 구현이 생기면 이 모듈의 함수를 그 결과로 바꾼다.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def mitigated(before: dict, fairness_cfg: dict) -> dict:
    """Threshold Adjustment 를 가정한 예시 값.

    집단별 승인율·TPR·FPR 을 전체 평균 쪽으로 절반 당긴다.
    """
    groups = before["groups"]
    mean = {
        k: float(np.mean([g[k] for g in groups.values()])) for k in ("approval_rate", "tpr", "fpr")
    }
    after_groups = {
        name: {**g, **{k: round((g[k] + mean[k]) / 2, 4) for k in mean}, "approval_ci95": None}
        for name, g in groups.items()
    }
    rates = [g["approval_rate"] for g in after_groups.values()]
    tprs = [g["tpr"] for g in after_groups.values()]
    fprs = [g["fpr"] for g in after_groups.values()]
    tpr_diff, fpr_diff = round(max(tprs) - min(tprs), 4), round(max(fprs) - min(fprs), 4)
    di = round(min(rates) / max(rates), 4)
    lo, hi = fairness_cfg["di_ratio_bounds"]
    return {
        **before,
        "di_ratio": di,
        "equalized_odds_tpr_diff": tpr_diff,
        "equalized_odds_fpr_diff": fpr_diff,
        "di_pass": lo <= di <= hi,
        "eo_pass": max(tpr_diff, fpr_diff) <= fairness_cfg["equalized_odds_max_diff"],
        "groups": after_groups,
        "warnings": [],
    }


def curves(auc: float, prevalence: float, n_points: int) -> tuple[dict, dict]:
    """보고된 AUC 와 같은 이항정규(binormal) 근사 ROC 곡선과, 부도율로 계산한 PR 곡선."""
    d = np.sqrt(2) * stats.norm.ppf(auc)
    # FPR 을 로그 간격으로 잡아 정밀도가 빠르게 변하는 낮은 FPR 구간도 촘촘하게 그린다
    fpr = np.r_[0.0, np.geomspace(1e-4, 1, n_points - 1)]
    tpr = stats.norm.cdf(stats.norm.ppf(np.clip(fpr, 1e-6, 1 - 1e-6)) + d)
    tpr[0], tpr[-1] = 0.0, 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = prevalence * tpr / (prevalence * tpr + (1 - prevalence) * fpr)
    precision[0] = 1.0
    roc = {"fpr": np.round(fpr, 4).tolist(), "tpr": np.round(tpr, 4).tolist()}
    pr = {
        "recall": np.round(tpr, 4).tolist(),
        "precision": np.round(precision, 4).tolist(),
        "baseline": round(prevalence, 4),
    }
    return roc, pr
