"""Local SHAP explanation: waterfall plot images for individual applicants.

    python -m src.explainer.waterfall

Writes one waterfall plot per example applicant of the API, in Korean and English. The examples
are the approved and rejected applicants in src/api/schemas.py, which the dashboard also offers:
- reports/figures/15_shap_waterfall_{rejected,approved}_{ko,en}.png
Bars start at the base value E[f(X)] and add each feature's SHAP value until they reach the
applicant's default log-odds f(x). Red bars raise the risk, blue bars lower it.
The dashboard draws the same explanation interactively from POST /explain.
Settings: `explainer` in configs/config.yaml. Educational use only.

Korean figures need a Korean font. Pass one in as for global_shap, for example:
    docker compose --profile dev run --rm \
      -v C:/Windows/Fonts/NanumGothic.ttf:/fonts/ko.ttf:ro -e KO_FONT_PATH=/fonts/ko.ttf \
      dev python -m src.explainer.waterfall
"""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")  # no display in the container

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from src.config import load_config  # noqa: E402
from src.explainer.global_shap import FIG_DIR, MUTED, labels, set_style  # noqa: E402

EXAMPLES = ("rejected", "approved")
TEXT = {
    "ko": {
        "title": "SHAP Waterfall Plot: {who} (부도 확률 {pd:.2%}, {score}점, {decision})",
        "note": "E[f(X)] = 기준값 (평균 고객), f(x) = 이 고객의 부도 log-odds. "
        "빨강 = 부도 위험을 높인 변수, 파랑 = 낮춘 변수. 변수 이름 앞의 숫자는 입력값",
        "rejected": "거절 예시 고객",
        "approved": "승인 예시 고객",
        "REJECTED": "거절",
        "APPROVED": "승인",
    },
    "en": {
        "title": "SHAP waterfall plot: {who} (default probability {pd:.2%}, score {score}, "
        "{decision})",
        "note": "E[f(X)] = base value (average applicant), f(x) = this applicant's default "
        "log-odds. Red = raised the default risk, blue = lowered it. Numbers before the names "
        "are the inputs",
        "rejected": "rejected example applicant",
        "approved": "approved example applicant",
        "REJECTED": "rejected",
        "APPROVED": "approved",
    },
}


def localize_rest_label(text: str, lang: str) -> str:
    """shap's '11 other features' row in the figure's language."""
    if lang != "ko":
        return text
    return re.sub(r"^(\d+) other features?$", r"기타 변수 \1개", text)


def plot_waterfall(
    base_value: float,
    shap_values: pd.Series,
    values: pd.Series,
    lang: str,
    max_display: int,
    title: str | None = None,
    note: str | None = None,
):
    """One applicant's SHAP waterfall plot (shap.plots.waterfall), with display names.

    shap_values and values share the index (model feature names). SHAP values are in default
    log-odds, so base_value + sum(shap_values) = logit(PD).
    """
    names = labels(lang)
    explanation = shap.Explanation(
        values=shap_values.to_numpy(dtype=float),
        base_values=float(base_value),
        data=values.reindex(shap_values.index).to_numpy(dtype=float),
        feature_names=[names.get(f, f) for f in shap_values.index],
    )
    plt.figure()
    ax = shap.plots.waterfall(explanation, max_display=max_display, show=False)
    ticks = [localize_rest_label(t.get_text(), lang) for t in ax.get_yticklabels()]
    ax.set_yticks(ax.get_yticks(), ticks)
    fig = ax.figure
    # shap writes U+2212 minus signs, which some Korean fonts (NanumGothic) do not have
    for text in fig.findobj(matplotlib.text.Text):
        text.set_text(text.get_text().replace("−", "-"))
    # Centered over the bars, clear of shap's f(x) label above and E[f(X)] label below
    if title:
        ax.set_title(title, pad=34, fontsize=11)
    if note:
        ax.annotate(note, xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -58),
                    textcoords="offset points", ha="center", va="top", fontsize=7, color=MUTED)
    return fig


def main() -> None:
    from src.api.schemas import _APPROVE_EXAMPLE, _REJECT_EXAMPLE, CreditApplication
    from src.explainer.shap_explainer import get_explainer
    from src.models.predictor import get_model

    cfg = load_config()["explainer"]
    model = get_model()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for key, example in zip(EXAMPLES, (_REJECT_EXAMPLE, _APPROVE_EXAMPLE), strict=True):
        X = model.prepare([CreditApplication(**example).model_inputs()])
        row = model.predict(X).iloc[0]
        base, values = get_explainer().explain(X)
        shap_row = pd.Series(values[0], index=X.columns)
        decision = "APPROVED" if int(row["approved"]) == 1 else "REJECTED"
        for lang in ("ko", "en"):
            set_style(lang)
            t = TEXT[lang]
            title = t["title"].format(
                who=t[key], pd=float(row["pd"]), score=int(row["score"]), decision=t[decision]
            )
            fig = plot_waterfall(base, shap_row, X.iloc[0], lang, cfg["waterfall_max_display"],
                                 title, t["note"])
            fig.savefig(FIG_DIR / f"15_shap_waterfall_{key}_{lang}.png", bbox_inches="tight")
            plt.close(fig)
        print(f"{key} example: {decision}, PD {row['pd']:.4f}, score {int(row['score'])}, "
              f"base value {base:.3f}, log-odds {base + shap_row.sum():.3f}")
    print("Saved reports/figures/15_shap_waterfall_{rejected,approved}_{ko,en}.png")


if __name__ == "__main__":
    main()
