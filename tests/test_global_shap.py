"""Global SHAP (src/explainer/global_shap.py): config, labels, plots on toy data, and consistency
with reports/shap_global_top10.csv on the real test set."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.features import feature_sets
from src.explainer.global_shap import importance, labels, plot_dependence, plot_summary
from tests.helpers import ROOT, load_yaml, require

FEATURES = feature_sets()["all"]


def test_dependence_features_are_model_features():
    cfg = load_config()["explainer"]
    deps = cfg["dependence_features"]
    assert set(deps) <= set(FEATURES)
    assert len(deps) >= cfg["dependence_plot_min_features"]
    assert cfg["summary_top_features"] <= len(FEATURES)


@pytest.mark.parametrize("lang", ["ko", "en"])
def test_every_model_feature_has_a_short_label(lang):
    names = labels(lang)
    assert all(names.get(f) for f in FEATURES)
    assert not any(names[f].endswith(")") for f in FEATURES)  # trailing notes removed


def test_plots_run_on_toy_data():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    cols = ["age", "DebtRatio", "NumberOfTimes90DaysLate"]
    X = pd.DataFrame(rng.random((300, 3)) * [60, 2, 3], columns=cols)
    X.iloc[:5, 1] = np.nan  # missing values are drawn gray / excluded
    shap_values = pd.DataFrame(rng.normal(size=(300, 3)) * [0.1, 0.5, 0.3], columns=cols)

    assert list(importance(shap_values).index) == ["DebtRatio", "NumberOfTimes90DaysLate", "age"]
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plot_summary(shap_values, X, top=2, lang="en")
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == [
        "Times 90+ days late", "Debt ratio"  # bottom to top: most important at the top
    ]
    plt.close(fig)
    fig = plot_dependence(shap_values, X, cols, lang="en")
    assert sum(ax.get_visible() for ax in fig.axes) == 3
    plt.close(fig)


def test_importance_matches_published_top10():
    """TreeExplainer values on the test set give the same ranking as the model report's top 10."""
    require(ROOT / load_yaml("train.yaml")["data"])
    from src.explainer.global_shap import compute
    from src.fairness.data import load_split

    X = load_split("test")[FEATURES].astype(float)
    _, shap_values = compute(X)
    published = pd.read_csv(ROOT / "reports" / "shap_global_top10.csv")
    imp = importance(shap_values)
    assert list(imp.index[:10]) == published["feature"].tolist()
    assert imp.head(10).to_numpy() == pytest.approx(published["mean_abs_shap"].to_numpy(), abs=1e-3)
