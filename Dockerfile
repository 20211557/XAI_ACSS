# syntax=docker/dockerfile:1
# =============================================================================
# XAI 대안신용평가 시스템 - 멀티 스테이지 Dockerfile
#
#   target      용도                                  포트
#   ---------   -----------------------------------   -----
#   mlflow      MLflow Tracking Server + Registry      5000
#   api         FastAPI 신용평가/설명 API              8000
#   dashboard   Streamlit 심사역 대시보드 (모델 없음)  8501
#   dev         개발 환경 (JupyterLab, pytest, 학습)   8888
#
# 각 서비스는 docker-compose.yml 에서 target 으로 선택해 빌드한다.
# =============================================================================
ARG PYTHON_VERSION=3.10

# -----------------------------------------------------------------------------
# base: 모든 이미지 공통 (Python 런타임, 비루트 사용자)
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN groupadd --gid 1000 app \
 && useradd --uid 1000 --gid app --create-home --shell /bin/bash app

# -----------------------------------------------------------------------------
# mlflow: 클라이언트와 서버 버전이 어긋나지 않도록 requirements.txt 의 mlflow 버전을 그대로 사용
# -----------------------------------------------------------------------------
FROM base AS mlflow

COPY requirements.txt /tmp/requirements.txt
RUN pip install "$(grep -E '^mlflow==' /tmp/requirements.txt)" \
 && mkdir -p /mlflow \
 && chown app:app /mlflow

USER app
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--serve-artifacts", "--artifacts-destination", "/mlflow/artifacts"]

# -----------------------------------------------------------------------------
# ml: 학습/추론 공통 런타임 (XGBoost/LightGBM 은 OpenMP 런타임 libgomp 필요)
# -----------------------------------------------------------------------------
FROM base AS ml

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# -----------------------------------------------------------------------------
# api: FastAPI 서버. models/, reports/ 는 docker-compose 에서 볼륨으로 마운트
# -----------------------------------------------------------------------------
FROM ml AS api

COPY configs/ configs/
COPY src/ src/

USER app
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# -----------------------------------------------------------------------------
# dashboard: ML 라이브러리를 설치하지 않아 모델 파일을 직접 로드할 수 없다 (MSA 강제).
#            모든 추론/지표 조회는 API_URL 로 HTTP 호출
# -----------------------------------------------------------------------------
FROM base AS dashboard

COPY requirements-dashboard.txt .
RUN pip install -r requirements-dashboard.txt

COPY configs/ configs/
COPY src/ src/

USER app
EXPOSE 8501
CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]

# -----------------------------------------------------------------------------
# dev: EDA·학습·테스트용. 소스는 이미지에 넣지 않고 레포 전체를 /app 에 마운트
# -----------------------------------------------------------------------------
FROM ml AS dev

COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

USER app
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir=/app"]
