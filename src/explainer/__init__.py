"""설명가능성(XAI) 모듈.

- shap_explainer.py : SHAP TreeExplainer (개별 고객의 SHAP 값)
- global_shap.py    : 전역 해석 그림, Summary Plot (상위 15개) / Dependence Plot (변수 4개)
- waterfall.py      : 지역 해석 그림, 개별 고객의 Waterfall Plot (예시 고객: 거절 / 승인)
- reasons.py        : SHAP 기반 거절 사유 Top 5
                      (feature, current_value, shap_contribution, explanation)
"""
