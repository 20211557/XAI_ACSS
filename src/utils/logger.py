"""공통 로거. 데이터 로드 요약, 공정성 WARNING 등 모든 모듈이 이 로거를 사용한다."""

from __future__ import annotations

import logging
import sys

from src.config import get_settings

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(get_settings().log_level.upper())
        logger.propagate = False
    return logger
