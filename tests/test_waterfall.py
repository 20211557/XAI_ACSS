"""Local SHAP waterfall plot (src/explainer/waterfall.py): config, the plot on toy values, and
the committed example images for one rejected and one approved applicant."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import PROJECT_ROOT, load_config
from src.data.features import feature_sets
from src.explainer.global_shap import labels
from src.explainer.waterfall import EXAMPLES, localize_rest_label, plot_waterfall

FEATURES = feature_sets()["all"]


def test_waterfall_setting_fits_the_model():
    assert 3 <= load_config()["explainer"]["waterfall_max_display"] < len(FEATURES)


def test_rest_row_is_translated_for_korean_figures():
    assert localize_rest_label("12 other features", "ko") == "기타 변수 12개"
    assert localize_rest_label("12 other features", "en") == "12 other features"
    assert localize_rest_label("Age", "ko") == "Age"


def test_waterfall_plot_runs_on_toy_values():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    feats = FEATURES[:8]
    shap_values = pd.Series(rng.normal(size=8) * 0.3, index=feats)
    values = pd.Series(rng.random(8), index=feats)
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plot_waterfall(-2.5, shap_values, values, "en", max_display=4, title="t", note="n")

    ticks = " ".join(t.get_text() for t in fig.axes[0].get_yticklabels())
    assert labels("en")[shap_values.abs().idxmax()] in ticks  # largest effect is drawn
    assert "5 other features" in ticks  # 8 features, 3 drawn + 1 grouped row
    assert "−" not in ticks  # minus signs replaced for fonts without U+2212
    plt.close(fig)


@pytest.mark.parametrize("example", EXAMPLES)
@pytest.mark.parametrize("lang", ["ko", "en"])
def test_example_waterfall_images_exist(example, lang):
    path = PROJECT_ROOT / "reports" / "figures" / f"15_shap_waterfall_{example}_{lang}.png"
    assert path.read_bytes().startswith(b"\x89PNG")
