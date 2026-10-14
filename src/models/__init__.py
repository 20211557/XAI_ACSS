"""신용평가 모델.

- train.py    : LR / XGBoost / LightGBM × 피처 그룹 × 불균형 처리 비교 (5 × 5겹 Stratified CV),
                단조 제약, 최종 모델 저장 (models/{name}_v{version}.joblib), MLflow 로깅
                실행: python -m src.models.train --stage all
- evaluate.py : AUC-ROC, KS, PSI, Precision/Recall/F1, 씬파일러 AUC uplift, 부트스트랩 신뢰구간
- scoring.py  : 부도확률 → 신용 점수(0-1000) → 리스크 등급(A-E) → 승인 여부 (configs/scoring.yaml)
- german.py   : German Credit 보조 모델 (실제 성별 공정성 점검용, python -m src.models.german)

예정: registry.py (MLflow Model Registry 등록)
"""
