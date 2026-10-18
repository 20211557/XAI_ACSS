"""배포 모델 서빙: 원시 고객 정보 → 모델 변수 21개 → PD → 점수·등급·승인.

학습-서빙 일치: 원시 입력에 학습 때와 같은 정제 규칙(src.data.preprocessor.clean)을 적용한다.
특수코드(96/98), 소득 결측·0, 사용률 이상치가 학습 데이터와 같은 결측값과 플래그 열이 된다.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.config import PROJECT_ROOT, get_settings
from src.data.features import feature_sets, is_thin_filer_proxy
from src.data.preprocessor import clean
from src.data.preprocessor import load_config as load_data_config
from src.models.scoring import load_config as load_scoring
from src.models.scoring import score_frame

# 정제 규칙이 만드는 플래그 열 (원시 입력에는 없음)
DERIVED_FEATURES = ("pastdue_special_code", "income_missing", "income_zero", "util_outlier")


class CreditModel:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.pipeline = joblib.load(path)
        self.features = feature_sets()["all"]
        self.raw_features = [f for f in self.features if f not in DERIVED_FEATURES]
        self.data_cfg = load_data_config()
        self.scoring_cfg = load_scoring()
        with open(PROJECT_ROOT / "configs" / "simulator.yaml", encoding="utf-8") as f:
            self.thin_rule = yaml.safe_load(f)["thin_filer"]["rule"]
        trained_on = list(getattr(self.pipeline, "feature_names_in_", self.features))
        if trained_on != self.features:
            raise ValueError(f"{path.name} 의 입력 변수가 configs 의 변수 목록과 다릅니다")

    def prepare(self, applications: list[dict]) -> pd.DataFrame:
        """원시 입력(컬럼명 → 값, None = 결측)을 모델 입력 행렬로 바꾼다."""
        raw = pd.DataFrame(applications, columns=self.raw_features, dtype=float)
        target = self.data_cfg["target"]
        # clean() 은 정제 요약에 부도율을 계산하므로 타깃 열이 필요하다 (값은 쓰이지 않음)
        cleaned, _ = clean(raw.assign(**{target: 0}), self.data_cfg)
        if len(cleaned) != len(raw):
            raise ValueError(f"연령은 {self.data_cfg['min_age']}세 이상이어야 합니다")
        return cleaned[self.features].astype(float)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """pd, score, grade, approved (1/0), thin_filer (1/0) 열을 가진 DataFrame."""
        out = score_frame(self.pipeline.predict_proba(X)[:, 1], self.scoring_cfg)
        out["thin_filer"] = is_thin_filer_proxy(X, self.thin_rule).to_numpy()
        return out

    @property
    def reference_values(self) -> dict[str, float]:
        """학습 데이터(train + validation) 중앙값: 파이프라인 imputer 가 학습한 값."""
        imputer = getattr(self.pipeline, "named_steps", {}).get("impute")
        if imputer is None:
            return {}
        return dict(zip(self.features, np.asarray(imputer.statistics_, float), strict=True))


@lru_cache
def get_model() -> CreditModel:
    return CreditModel(get_settings().model_path)
