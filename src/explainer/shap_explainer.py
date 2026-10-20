"""SHAP TreeExplainer: 개별 고객의 지역 해석.

값의 단위는 부도 log-odds 이다. base_value + Σ shap = logit(PD), 양수 = 부도 위험을 높이는 방향.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import shap

from src.models.predictor import get_model


class ShapExplainer:
    def __init__(self, pipeline) -> None:
        if not hasattr(pipeline, "steps"):
            raise TypeError("전처리 단계와 트리 모델로 이루어진 학습된 Pipeline 이 필요합니다")
        self.preprocess = pipeline[:-1]
        self.explainer = shap.TreeExplainer(pipeline[-1])

    def explain(self, X: pd.DataFrame) -> tuple[float, np.ndarray]:
        """(base_value, shap 값 행렬 [행 × 변수])."""
        values = self.explainer(self.preprocess.transform(X))
        return float(np.atleast_1d(values.base_values)[0]), np.asarray(values.values)


@lru_cache
def get_explainer() -> ShapExplainer:
    return ShapExplainer(get_model().pipeline)
