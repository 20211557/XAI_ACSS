# 데이터

원본 데이터는 라이선스(Kaggle 대회 규정) 때문에 레포에 포함하지 않는다. 각자 아래 방법으로 내려받는다.

```
data/
├── raw/         # 원본 (git 미추적)
└── processed/   # 전처리·분할 결과 (git 미추적)
```

## Give Me Some Credit (기본 데이터셋, 150,000건)

1. https://www.kaggle.com/c/GiveMeSomeCredit/rules 에서 대회 규정에 동의한다 (동의하지 않으면 403 에러).
2. https://www.kaggle.com/settings → API → Create New Token 으로 발급한 값을 `.env` 의 `KAGGLE_USERNAME`, `KAGGLE_KEY` 에 넣는다.
3. dev 컨테이너에서 다운로드:

```bash
docker compose --profile dev run --rm dev kaggle competitions download -c GiveMeSomeCredit -p data/raw
docker compose --profile dev run --rm dev python -m zipfile -e data/raw/GiveMeSomeCredit.zip data/raw/
```

학습에는 `data/raw/cs-training.csv` 를 사용한다 (`configs/config.yaml` 의 `data.raw_file`).

## German Credit (대안 데이터셋, 1,000건)

별도 다운로드 없이 코드에서 불러온다.

```python
from ucimlrepo import fetch_ucirepo

german_credit = fetch_ucirepo(id=144)
```
