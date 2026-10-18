#!/usr/bin/env bash
# Local setup without Docker (macOS / Linux). Run from the project root:  bash setup.sh
# With Docker, use the dev container instead (see README).
set -e
PY=${PYTHON:-python3}
echo "Python: $($PY --version)"
$PY -c 'import sys; assert sys.version_info >= (3,12), "Python 3.12+ required (xgboost 3.4, trained model)"'
if [[ "$(uname)" == "Darwin" ]] && command -v brew >/dev/null; then
  brew list libomp >/dev/null 2>&1 || { echo "Installing libomp (needed by LightGBM/XGBoost on macOS)"; brew install libomp; }
fi
$PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements-dev.txt -q
python -c "import pandas, scipy, sklearn, lightgbm, xgboost, imblearn, mlflow, skops, yaml, fastapi, streamlit; print('All packages OK')"
[ -f data/raw/cs-training.csv ] && echo "GMSC found" || echo "MISSING: put cs-training.csv in data/raw/ (see data/README.md)"
echo "Done. Activate with: source .venv/bin/activate"
