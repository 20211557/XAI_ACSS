"""신용평가 모델.

- train.py    : LR / XGBoost / LightGBM 학습, 5-Fold Stratified CV, 불균형 처리, MLflow 로깅
- evaluate.py : AUC-ROC, KS, PSI, Precision/Recall/F1, 씬파일러 AUC uplift
- registry.py : 버전별 모델 저장/로드 (models/{name}_{version}.pkl), MLflow Model Registry 등록
- scoring.py  : 부도확률 → 신용 점수(0-1000), 승인 여부, 리스크 등급(A-E) 변환
"""
