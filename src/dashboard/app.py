"""심사역용 Streamlit 대시보드.

실행: streamlit run src/dashboard/app.py
(PYTHONPATH 에 프로젝트 루트가 있어야 한다. Docker 이미지는 자동 설정)
각 탭의 실제 기능은 권장 구현 순서 23~24 단계에서 구현한다.
"""

from __future__ import annotations

import requests
import streamlit as st

from src.config import load_config
from src.dashboard.api_client import ApiClient

config = load_config()
client = ApiClient()

st.set_page_config(page_title="XAI 대안신용평가", page_icon="💳", layout="wide")
st.title("XAI 대안신용평가 · 심사역 대시보드")
st.caption(f"⚠️ {config['project']['disclaimer']}")

with st.sidebar:
    st.subheader("API 서버 상태")
    try:
        health = client.health()
        st.success(f"연결됨 · 모델 `{health['model_version']}`")
        if not health.get("model_available"):
            st.warning("학습된 모델 파일이 아직 없습니다.")
    except requests.RequestException:
        st.error(f"API 서버에 연결할 수 없습니다: {client.base_url}")

TABS = {
    "고객 심사": "입력 폼(나이, 소득, 부채비율 등) → 승인/거절, 신용 점수, 리스크 등급",
    "XAI 분석": "SHAP Summary Plot, 개별 고객 Waterfall Plot, 거절 사유 Top 5",
    "공정성": "집단별 승인율 막대 차트, DI Ratio, Mitigation 전/후 비교",
    "모델 성능": "ROC Curve, Precision-Recall Curve, 혼동행렬 히트맵",
    "통계 검증": "ANOVA 결과 테이블 (F, p, η²), Tukey HSD 결과",
}

for tab, (name, description) in zip(st.tabs(list(TABS)), TABS.items(), strict=True):
    with tab:
        st.subheader(name)
        st.info(f"구현 예정: {description}")
