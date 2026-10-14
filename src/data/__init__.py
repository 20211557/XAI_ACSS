"""데이터 파이프라인.

- loader.py              : Give Me Some Credit / German Credit 로드, 컬럼·타입·결측치 비율 출력
- preprocessor.py        : 정제 규칙, 폴드 안 결측치 대체·스케일링·인코딩, Stratified 70:15:15 분할
- simulator.py           : 대안 데이터 시뮬레이터 (generate_alternative_data,
                           thin_filer_ratio, bias_ratio)
- features.py            : 씬파일러 판정 (과제 규칙 / 대리 규칙), 모델 변수 목록
- actionability_table.py : configs/actionability.yaml → 변수별 조치 가능성 Excel 생성

실행: python -m src.data (정제) → python -m src.data.simulator (대안 데이터)
"""
