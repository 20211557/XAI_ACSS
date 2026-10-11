"""대시보드 → FastAPI HTTP 클라이언트.

대시보드는 모델 파일을 직접 로드하지 않는다 (MSA). 모든 추론/지표 조회는 이 클라이언트를 통한다.
"""

from __future__ import annotations

from typing import Any

import requests

from src.config import get_settings


class ApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 5.0) -> None:
        self.base_url = (base_url or get_settings().api_url).rstrip("/")
        self.timeout = timeout

    def _get(self, path: str) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}{path}", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._get("/health")
