"""프로젝트 전역 설정.

- 환경마다 달라지는 값(서비스 주소, 경로, 모델 버전)은 환경 변수 / .env 에서 읽는다 → ``Settings``
- 모델·비즈니스 임계값은 ``configs/config.yaml`` 에서 읽는다 → ``load_config()``
  (데이터·시뮬레이터·학습·점수 세부 설정은 ``configs/*.yaml`` 각 파일, 해당 모듈이 직접 읽는다)

상대 경로는 모두 PROJECT_ROOT 기준으로 해석하므로 코드에 절대 경로를 쓰지 않는다.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
        protected_namespaces=(),
    )

    app_env: str = "local"
    log_level: str = "INFO"

    api_url: str = "http://localhost:8000"
    mlflow_tracking_uri: str = "http://localhost:5000"

    model_name: str = "xgboost"
    model_version: str = "v1.0"

    model_dir: Path = Path("models")
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    config_path: Path = Path("configs/config.yaml")

    @field_validator("model_dir", "data_dir", "reports_dir", "config_path")
    @classmethod
    def _resolve_from_project_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @property
    def model_id(self) -> str:
        """버전 정보가 포함된 모델 식별자 (예: ``xgboost_v1.0``)."""
        return f"{self.model_name}_{self.model_version}"

    @property
    def model_path(self) -> Path:
        """src/models/train.py 가 저장하는 파일 (예: ``models/xgboost_v1.0.joblib``)."""
        return self.model_dir / f"{self.model_id}.joblib"


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or get_settings().config_path
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
