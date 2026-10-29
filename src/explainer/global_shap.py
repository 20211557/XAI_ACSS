"""Global SHAP explanation of the shipped model on the test set: Summary Plot and Dependence Plots.

    python -m src.explainer.global_shap

Writes, in Korean and English:
- reports/figures/13_shap_summary_{ko,en}.png     beeswarm of the top features (mean |SHAP|)
- reports/figures/14_shap_dependence_{ko,en}.png  SHAP value against feature value
SHAP values are in default log-odds: positive = pushes toward default (rejection).
Settings: `explainer` in configs/config.yaml. Educational use only.

Korean figures need a Korean font. The dev container has none, so pass one in, for example:
    docker compose --profile dev run --rm \
      -v C:/Windows/Fonts/NanumGothic.ttf:/fonts/ko.ttf:ro -e KO_FONT_PATH=/fonts/ko.ttf \
      dev python -m src.explainer.global_shap
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # no display in the container

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.explainer.reasons import feature_metadata, short_label  # noqa: E402

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
KO_FONTS = ("AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans KR", "Noto Sans CJK KR")
# Colors shared with notebooks/eda.py (reference palette)
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"

TEXT = {
    "ko": {
        "summary_title": "전역 SHAP: 변수별 영향 (test 세트, 상위 {n}개)",
        "summary_x": "SHAP 값 (부도 log-odds, 양수 = 부도 위험 증가)",
        "high": "높음", "low": "낮음", "value": "변수 값",
        "dep_title": "SHAP Dependence Plot: 변수 값에 따른 영향 (test 세트)",
        "dep_y": "SHAP 값",
        "clipped": "상위 1% 값은 표시 범위 밖",
        "missing": "결측 {n}건 제외",
    },
    "en": {
        "summary_title": "Global SHAP: feature effects (test set, top {n})",
        "summary_x": "SHAP value (default log-odds, positive = higher default risk)",
        "high": "High", "low": "Low", "value": "Feature value",
        "dep_title": "SHAP dependence plots: effect by feature value (test set)",
        "dep_y": "SHAP value",
        "clipped": "top 1% of values outside the range",
        "missing": "{n} missing excluded",
    },
}


def labels(lang: str) -> dict[str, str]:
    """Short display name per feature, from configs/actionability.yaml."""
    feats = feature_metadata()["features"]
    return {f: short_label(m[f"label_{lang}"]) for f, m in feats.items()}


def korean_font() -> str:
    """Name of an installed Korean font; KO_FONT_PATH registers a font file first."""
    path = os.environ.get("KO_FONT_PATH")
    if path:
        font_manager.fontManager.addfont(path)
        return font_manager.FontProperties(fname=path).get_name()
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in KO_FONTS:
        if name in installed:
            return name
    raise RuntimeError(
        "no Korean font found; set KO_FONT_PATH to a .ttf file (see the module docstring)"
    )


def set_style(lang: str) -> None:
    plt.rcParams.update({
        "font.family": korean_font() if lang == "ko" else "DejaVu Sans",
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": INK_2, "axes.edgecolor": GRID,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "figure.dpi": 150,
    })


def compute(X: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """(base value, SHAP values with the same rows and columns as X) for the shipped model."""
    from src.explainer.shap_explainer import get_explainer

    base, values = get_explainer().explain(X)
    return base, pd.DataFrame(values, index=X.index, columns=X.columns)


def importance(shap_values: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per feature, largest first."""
    return shap_values.abs().mean().sort_values(ascending=False)


def _scaled(x: pd.Series) -> np.ndarray:
    """Feature value mapped to 0..1 by rank (robust to skew) for the color scale; NaN stays NaN."""
    return x.rank(pct=True).to_numpy()


def plot_summary(shap_values: pd.DataFrame, X: pd.DataFrame, top: int, lang: str):
    """Beeswarm: one row per feature, one dot per applicant, colored by the feature value."""
    t, names = TEXT[lang], labels(lang)
    order = importance(shap_values).index[:top][::-1]  # most important at the top
    rng = np.random.default_rng(0)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("lohi", [BLUE, "#c9c3d6", ORANGE])

    fig, ax = plt.subplots(figsize=(8, 0.42 * top + 1.4))
    for row, feat in enumerate(order):
        s = shap_values[feat].to_numpy()
        color = _scaled(X[feat])
        # Spread dots vertically by local density so overlapping points stay visible
        bins = np.round(s / (np.ptp(s) / 100 + 1e-12))
        _, inv, counts = np.unique(bins, return_inverse=True, return_counts=True)
        jitter = (rng.random(len(s)) - 0.5) * np.minimum(counts[inv] / counts.max(), 1) * 0.8
        missing = np.isnan(color)
        ax.scatter(s[missing], row + jitter[missing], s=2, color="#bdbdbd", alpha=0.5, lw=0)
        ax.scatter(s[~missing], row + jitter[~missing], s=2, c=color[~missing], cmap=cmap,
                   vmin=0, vmax=1, alpha=0.6, lw=0, rasterized=True)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(order)), [names[f] for f in order])
    ax.set_xlabel(t["summary_x"])
    ax.set_title(t["summary_title"].format(n=top), loc="left")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, 1))
    bar = fig.colorbar(sm, ax=ax, ticks=[0, 1], aspect=40, pad=0.02)
    bar.ax.set_yticklabels([t["low"], t["high"]])
    bar.set_label(t["value"])
    bar.outline.set_visible(False)
    fig.tight_layout()
    return fig


def plot_dependence(shap_values: pd.DataFrame, X: pd.DataFrame, features: list[str], lang: str):
    """SHAP value against the feature's own value, one panel per feature."""
    t, names = TEXT[lang], labels(lang)
    cols = 2
    rows = int(np.ceil(len(features) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8, 3.1 * rows), squeeze=False)
    for ax, feat in zip(axes.flat, features, strict=False):
        x, s = X[feat], shap_values[feat]
        ok = x.notna()
        notes = []
        if (~ok).sum():
            notes.append(t["missing"].format(n=int((~ok).sum())))
        hi = x[ok].quantile(0.99)
        if x[ok].nunique() > 20 and x[ok].max() > 1.5 * hi:  # long right tail: zoom in
            ok &= x <= hi
            notes.append(t["clipped"])
        ax.scatter(x[ok], s[ok], s=3, color=BLUE, alpha=0.25, lw=0, rasterized=True)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(names[feat], loc="left")
        ax.set_ylabel(t["dep_y"])
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        if notes:
            ax.text(1, 1.02, ", ".join(notes), transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color=MUTED)
    for ax in axes.flat[len(features):]:
        ax.set_visible(False)
    fig.suptitle(t["dep_title"], x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


def main() -> None:
    from src.fairness.data import load_split
    from src.models.predictor import get_model

    cfg = load_config()["explainer"]
    test = load_split("test")
    X = test[get_model().features].astype(float)
    base, shap_values = compute(X)

    deps = cfg["dependence_features"]
    unknown = [f for f in deps if f not in X.columns]
    if unknown or len(deps) < cfg["dependence_plot_min_features"]:
        raise ValueError(f"explainer.dependence_features: unknown {unknown} or too few {deps}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for lang in ("ko", "en"):
        set_style(lang)
        for name, fig in (
            ("13_shap_summary", plot_summary(shap_values, X, cfg["summary_top_features"], lang)),
            ("14_shap_dependence", plot_dependence(shap_values, X, deps, lang)),
        ):
            fig.savefig(FIG_DIR / f"{name}_{lang}.png", bbox_inches="tight")
            plt.close(fig)

    imp = importance(shap_values)
    print(f"Test set: {len(X)} rows, base value {base:.3f} (log-odds)")
    print("Top features by mean |SHAP|:")
    print(imp.head(cfg["summary_top_features"]).round(4).to_string())
    print("Saved reports/figures/13_shap_summary_{ko,en}.png, 14_shap_dependence_{ko,en}.png")


if __name__ == "__main__":
    main()
