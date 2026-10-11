"""FastAPI 대출 심사 API 서버 (Swagger UI: /docs).

현재는 /health 만 제공한다. /predict, /explain, /fairness, /fairness/mitigation,
/metrics, /anova 는 권장 구현 순서 21~22 단계에서 추가한다.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI

from src.api.schemas import HealthResponse
from src.config import get_settings, load_config

settings = get_settings()
config = load_config()

app = FastAPI(
    title="XAI 대안신용평가 API",
    description=(
        "대안 데이터 기반 신용평가, SHAP 설명, 공정성 검증, ANOVA 통계 검증 API\n\n"
        f"> ⚠️ {config['project']['disclaimer']}"
    ),
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_version=settings.model_id,
        model_available=settings.model_path.exists(),
        timestamp=datetime.now(timezone.utc),
    )
