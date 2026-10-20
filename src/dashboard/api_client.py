"""대시보드 → FastAPI HTTP 클라이언트.

대시보드는 모델 파일을 직접 로드하지 않는다 (MSA). 모든 추론/지표 조회는 이 클라이언트를 통한다.
"""

from __future__ import annotations

from typing import Any

import requests

from src.config import get_settings


class ApiError(RuntimeError):
    """API 오류 응답 (상태 코드와 detail 메시지)."""


class ApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 10.0) -> None:
        self.base_url = (base_url or get_settings().api_url).rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs) -> Any:
        response = requests.request(
            method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs
        )
        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(f"{method} {path} → {response.status_code}: {detail}")
        return response.json()

    def _get(self, path: str, **params) -> Any:
        return self._request("GET", path, params={k: v for k, v in params.items() if v is not None})

    def _post(self, path: str, payload: dict) -> Any:
        return self._request("POST", path, json=payload)

    def health(self) -> dict:
        return self._get("/health")

    def features(self) -> dict:
        return self._get("/features")

    def examples(self) -> list[dict]:
        """/predict 입력 예시 (OpenAPI 스키마의 CreditApplication examples)."""
        schema = self._get("/openapi.json")
        return schema["components"]["schemas"]["CreditApplication"].get("examples", [])

    def predict(self, payload: dict) -> dict:
        return self._post("/predict", payload)

    def explain(self, payload: dict) -> dict:
        return self._post("/explain", payload)

    def explain_global(self) -> dict:
        return self._get("/explain/global")

    def fairness(self, attribute: str | None = None) -> dict:
        return self._get("/fairness", attribute=attribute)

    def mitigation(self) -> dict:
        return self._get("/fairness/mitigation")

    def metrics(self) -> dict:
        return self._get("/metrics")

    def anova(self) -> dict:
        return self._get("/anova")
