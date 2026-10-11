"""데이터 파이프라인.

- loader.py       : Give Me Some Credit / German Credit 로드, 컬럼·타입·결측치 비율 로깅
- simulator.py    : 대안 데이터 시뮬레이터 (5개 대안 변수, thin_filer_ratio, bias_ratio)
- thin_filer.py   : 씬파일러 판정 (연체 컬럼 2개 이상 결측 또는 카드 이력 12개월 미만)
- preprocessor.py : 결측치 처리, 스케일링, 인코딩, Stratified Train/Valid/Test 분할
"""
