"""API Request/Response Pydantic 스키마."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., examples=["healthy"])
    model_version: str = Field(..., examples=["xgboost_v1.0"])
    model_available: bool = Field(..., description="model_version 에 해당하는 모델 파일 존재 여부")
    timestamp: datetime
