import pytest

from src.config import PROJECT_ROOT, Settings, load_config


def test_relative_paths_resolve_under_project_root():
    settings = Settings(model_dir="models", config_path="configs/config.yaml")

    assert settings.model_dir == PROJECT_ROOT / "models"
    assert settings.config_path.is_file()


def test_model_path_contains_version():
    settings = Settings(model_name="xgboost", model_version="v1.0")

    assert settings.model_path.name == "xgboost_v1.0.pkl"


def test_split_ratio_matches_spec():
    split = load_config()["data"]["split"]
    ratio = (split["train"], split["valid"], split["test"])

    assert sum(ratio) == pytest.approx(1.0)
    assert ratio in [(0.70, 0.15, 0.15), (0.80, 0.10, 0.10)]
    assert split["stratify"] is True


def test_fairness_thresholds_match_spec():
    fairness = load_config()["fairness"]

    assert fairness["di_ratio_bounds"] == [0.8, 1.25]
    assert fairness["equalized_odds_max_diff"] == pytest.approx(0.10)
