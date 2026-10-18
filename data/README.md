# 데이터

> 본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다.

원본 데이터는 라이선스(Kaggle 대회 규정) 때문에 레포에 포함하지 않는다. 각자 아래 방법으로 내려받는다.

```
data/
├── raw/         # 원본 (git 미추적): cs-training.csv, german.data
└── processed/   # 정제·시뮬레이션·분할 결과 (git 미추적, 파이프라인이 생성)
```

## Give Me Some Credit (기본 데이터셋, 150,000건)

1. https://www.kaggle.com/c/GiveMeSomeCredit/rules 에서 대회 규정에 동의한다 (동의하지 않으면 403 에러).
2. https://www.kaggle.com/settings → API → Create New Token 으로 발급한 값을 `.env` 의 `KAGGLE_USERNAME`, `KAGGLE_KEY` 에 넣는다.
3. dev 컨테이너에서 다운로드:

```bash
docker compose --profile dev run --rm dev kaggle competitions download -c GiveMeSomeCredit -p data/raw
docker compose --profile dev run --rm dev python -m zipfile -e data/raw/GiveMeSomeCredit.zip data/raw/
```

학습에는 `data/raw/cs-training.csv` 를 사용한다 (`configs/data.yaml` 의 `raw_path`).

## German Credit (보조 데이터셋, 1,000건)

실제 성별 공정성 점검용 보조 모델(`python -m src.models.german`)이 `data/raw/german.data` 를 읽는다
(`configs/german.yaml` 의 `raw_path`). [UCI 저장소](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)에서 내려받거나 아래처럼 받는다.

```bash
docker compose --profile dev run --rm dev python -c "import urllib.request; urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data', 'data/raw/german.data')"
```

출처: Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77 (CC BY 4.0)

## 생성되는 파일 (data/processed/)

| 파일 | 만드는 명령 | 내용 |
| --- | --- | --- |
| `gmsc_clean.csv` | `python -m src.data` | 정제한 GMSC (특수코드·이상치 → 결측 + 플래그 열) |
| `gmsc_sim.csv` | `python -m src.data.simulator` | 대안 변수 8개 + 시뮬레이션 성별 + 씬파일러 플래그 (학습용) |
| `gmsc_sim_target_only.csv` | `python -m src.data.simulator` | 신용 연결 없이(b = 0) 만든 비교용 버전 (시뮬레이터 보고서용) |
| `split.csv` | `python -m src.models.train` | `row_id`, `split` (train / validation / test, 70:15:15, stratify=y). 팀 전체가 이 분할을 사용한다 |
| `final_test_scores.csv` | `python -m src.models.train` | 배포 모델의 test 세트 PD, 점수, 등급, 승인 여부 |
