"""더미 데이터: 실제 값을 쓸 수 없을 때의 예시 값. 응답에는 is_dummy=True 로 표시한다.

- ROC / PR 곡선 좌표: 평소에는 reports/test_curves.json (python -m src.models.curves, 배포 모델의
  test 세트 실제 점수) 을 쓴다. 이 파일이 없거나 배포 모델과 다를 때만 아래 근사 곡선을 쓴다.
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
