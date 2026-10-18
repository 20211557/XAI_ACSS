@echo off
REM Local setup without Docker (Windows). Run from the project root: setup.bat
REM With Docker, use the dev container instead (see README).
python --version
python -c "import sys; assert sys.version_info >= (3,12), 'Python 3.12+ required (xgboost 3.4, trained model)'" || exit /b 1
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip -q
pip install -r requirements-dev.txt -q
python -c "import pandas, scipy, sklearn, lightgbm, xgboost, imblearn, mlflow, skops, yaml, fastapi, streamlit; print('All packages OK')"
if exist data\raw\cs-training.csv (echo GMSC found) else (echo MISSING: put cs-training.csv in data\raw\ ^(see data\README.md^))
echo NOTE: native Windows XGBoost cannot load the shipped model (trained on macOS). Train and test in the dev container (see README).
echo Done. Activate with: .venv\Scripts\activate
