"""더미 데이터: 아직 실제로 계산할 수 없는 값. 응답에는 is_dummy=True 로 표시한다.

- ROC / PR 곡선 좌표: test 세트 점수 파일(data/processed/final_test_scores.csv)이 레포에 없다
실제 구현이 생기면 이 모듈의 함수를 그 결과로 바꾼다.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


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
