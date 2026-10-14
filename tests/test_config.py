import json

import pytest
import yaml

from src.config import PROJECT_ROOT, Settings, load_config


def _yaml(name: str) -> dict:
    with open(PROJECT_ROOT / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_relative_paths_resolve_under_project_root():
    settings = Settings(model_dir="models", config_path="configs/config.yaml")

    assert settings.model_dir == PROJECT_ROOT / "models"
    assert settings.config_path.is_file()


def test_model_path_contains_version():
    settings = Settings(model_name="xgboost", model_version="v1.0")

    assert settings.model_path.name == "xgboost_v1.0.joblib"


def test_default_model_is_the_shipped_model():
    """API 기본 모델(MODEL_NAME, MODEL_VERSION)이 학습 결과의 배포 모델 파일과 같아야 한다."""
    fields = Settings.model_fields
    settings = Settings(model_name=fields["model_name"].default,
                        model_version=fields["model_version"].default, model_dir="models")
    summary_path = PROJECT_ROOT / "reports" / "final_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert settings.model_path == PROJECT_ROOT / summary["model_file"]
    assert settings.model_path.is_file()


def test_split_ratio_matches_spec():
    split = _yaml("train.yaml")["split"]
    ratio = (split["train"], split["validation"], split["test"])

    assert sum(ratio) == pytest.approx(1.0)
    assert ratio in [(0.70, 0.15, 0.15), (0.80, 0.10, 0.10)]


def test_fairness_thresholds_match_spec():
    fairness = load_config()["fairness"]

    assert fairness["di_ratio_bounds"] == [0.8, 1.25]
    assert fairness["equalized_odds_max_diff"] == pytest.approx(0.10)


def test_fairness_age_groups_match_simulator_bands():
    age_groups = load_config()["fairness"]["protected_attributes"]["age_group"]

    assert age_groups == _yaml("simulator.yaml")["age_bands"]["labels"]
