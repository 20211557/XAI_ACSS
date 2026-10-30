"""심사역용 Streamlit 대시보드 (5개 탭: 고객 심사, XAI 분석, 공정성, 모델 성능, 통계 검증).

실행: streamlit run src/dashboard/app.py
(PYTHONPATH 에 프로젝트 루트가 있어야 한다. Docker 이미지는 자동 설정)
모델 파일을 직접 로드하지 않고 모든 결과를 FastAPI 에서 HTTP 로 받는다 (MSA).
"""

from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yaml

from src.config import PROJECT_ROOT, load_config
from src.dashboard.api_client import ApiClient, ApiError

config = load_config()
client = ApiClient()
with open(PROJECT_ROOT / "configs" / "scoring.yaml", encoding="utf-8") as _f:
    SCORING = yaml.safe_load(_f)

RISK_LEVELS = config["dashboard"]["risk_levels"]
# 위험도: (표시 이름, 배경색, 글자색) - 고위험 빨강, 중위험 노랑, 저위험 초록
RISK_STYLE = {
    "low": ("저위험", "#2e7d32", "#ffffff"),
    "medium": ("중위험", "#f9c80e", "#1f1f1f"),
    "high": ("고위험", "#c62828", "#ffffff"),
}
RISK_UP, RISK_DOWN, NEUTRAL = "#c62828", "#2e7d32", "#546e7a"
SERIES = ["#1565c0", "#ef6c00"]

st.set_page_config(page_title="XAI 대안신용평가", page_icon="💳", layout="wide")
st.title("XAI 대안신용평가 · 심사역 대시보드")
st.caption(f"⚠️ {config['project']['disclaimer']}")


@st.cache_data(ttl=60, show_spinner=False)
def fetch(name: str, **kwargs):
    """GET 엔드포인트 결과 (60초 캐시)."""
    return getattr(client, name)(**kwargs)


def short(label: str) -> str:
    return re.sub(r"\s*\([^()]*\)\s*$", "", label).strip()


def source_caption(is_dummy: bool, source: str | None = None, note: str | None = None) -> None:
    if is_dummy:
        st.caption(f"🟡 **더미 데이터** · {note or '실제 값을 계산할 수 없어 넣은 예시 값'}")
    else:
        st.caption(f"✅ 실제 데이터 · {source or ''}" + (f" · {note}" if note else ""))


def risk_badge(grade: str) -> str:
    name, bg, fg = RISK_STYLE[RISK_LEVELS.get(grade, "high")]
    return (
        f"<span style='background:{bg};color:{fg};padding:6px 14px;border-radius:14px;"
        f"font-weight:700;font-size:1.1rem'>등급 {grade} · {name}</span>"
    )


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def chart(fig: go.Figure, height: int = 360) -> None:
    fig.update_layout(height=height, margin={"l": 10, "r": 10, "t": 40, "b": 10})
    st.plotly_chart(fig, width="stretch")


with st.sidebar:
    st.subheader("API 서버 상태")
    try:
        health = client.health()
        st.success(f"연결됨 · 모델 `{health['model_version']}`")
        if not health.get("model_available"):
            st.warning("학습된 모델 파일이 없습니다. 고객 심사와 XAI 분석은 사용할 수 없습니다.")
        if health.get("model_version_match") is False:
            st.warning(
                f"API 모델 `{health['model_version']}` 이 최근 학습 결과 "
                f"`{health['trained_model_version']}` 와 다릅니다 "
                "(MODEL_NAME / MODEL_VERSION 확인)."
            )
    except requests.RequestException:
        st.error(f"API 서버에 연결할 수 없습니다: {client.base_url}")
        st.stop()
    st.caption("🟡 표시는 아직 계산할 수 없어 넣은 더미 데이터입니다.")


# ---------------------------------------------------------------------------
# 1. 고객 심사
# ---------------------------------------------------------------------------
def _initial(f: dict, value):
    value = f["default"] if value is None else value
    value = 0 if value is None else value
    return int(round(value)) if f["integer"] else float(value)


def _step(f: dict) -> float | int:
    if f["integer"]:
        return 1
    base = f["default"] or 0
    return 100.0 if base >= 1000 else 1.0 if base >= 10 else 0.01


def _load_values(feats: list[dict], values: dict) -> None:
    for f in feats:
        st.session_state[f"in_{f['feature']}"] = _initial(f, values.get(f["feature"]))
    st.session_state["in_customer_id"] = values.get("customer_id") or ""


def score_gauge(score: int) -> go.Figure:
    grades = sorted(SCORING["grades"], key=lambda g: g["min_score"])
    bounds = [g["min_score"] for g in grades] + [SCORING["score"]["max"]]
    steps = [
        {"range": [bounds[i], bounds[i + 1]], "color": RISK_STYLE[RISK_LEVELS[g["grade"]]][1]}
        for i, g in enumerate(grades)
    ]
    cutoff = min(
        g["min_score"] for g in grades if g["grade"] in SCORING["approval"]["approve_grades"]
    )
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "신용 점수"},
            gauge={
                "axis": {"range": [SCORING["score"]["min"], SCORING["score"]["max"]]},
                "bar": {"color": "#263238", "thickness": 0.25},
                "steps": steps,
                "threshold": {"line": {"color": "#000000", "width": 3}, "value": cutoff},
            },
        )
    )


def tab_review() -> None:
    feats = fetch("features")["features"]
    if "in_customer_id" not in st.session_state:
        _load_values(feats, {})
    try:
        examples = fetch("examples")
    except ApiError:
        examples = []

    st.markdown(
        "고객 정보를 입력하고 **심사하기** 를 누르세요. "
        "입력란은 학습 데이터 중앙값으로 미리 채워져 있습니다."
    )
    b1, b2, b3, _ = st.columns([1, 1, 1, 3])
    if b1.button("승인 예시", disabled=len(examples) < 1):
        _load_values(feats, examples[0])
    if b2.button("거절 예시", disabled=len(examples) < 2):
        _load_values(feats, examples[1])
    if b3.button("중앙값으로 초기화"):
        _load_values(feats, {})

    with st.form("application"):
        st.text_input("고객 ID", key="in_customer_id")
        left, right = st.columns(2)
        for column, group, title in (
            (left, "credit", "신용 정보"),
            (right, "alternative", "대안 데이터 (시뮬레이션)"),
        ):
            with column:
                st.markdown(f"**{title}**")
                for f in (f for f in feats if f["group"] == group):
                    cast = int if f["integer"] else float
                    st.number_input(
                        f["label_ko"],
                        key=f"in_{f['feature']}",
                        step=_step(f),
                        min_value=None if f["min"] is None else cast(f["min"]),
                        max_value=None if f["max"] is None else cast(f["max"]),
                        help=f"{f['feature']} · 중앙값 {f['default']}",
                    )
        submitted = st.form_submit_button("심사하기", type="primary")

    if submitted:
        payload = {f["feature"]: st.session_state[f"in_{f['feature']}"] for f in feats}
        payload["customer_id"] = st.session_state["in_customer_id"] or None
        try:
            st.session_state["result"] = client.predict(payload)
            st.session_state["explanation"] = client.explain(payload)
        except ApiError as exc:
            st.error(str(exc))

    result = st.session_state.get("result")
    if not result:
        return
    st.divider()
    approved = result["approval_status"] == "APPROVED"
    (st.success if approved else st.error)(
        f"**{'승인' if approved else '거절'} ({result['approval_status']})** · "
        f"고객 {result.get('customer_id') or '-'}"
    )
    gauge, info = st.columns(2)
    with gauge:
        chart(score_gauge(result["credit_score"]), height=300)
    with info:
        st.markdown(risk_badge(result["risk_grade"]), unsafe_allow_html=True)
        st.write("")
        c1, c2 = st.columns(2)
        c1.metric("신용 점수", result["credit_score"], border=True)
        c2.metric("부도 확률", f"{result['default_probability'] * 100:.2f}%", border=True)
        c1.metric("리스크 등급", result["risk_grade"], border=True)
        c2.metric("씬파일러", "예" if result["is_thin_filer"] else "아니오", border=True)
    explanation = st.session_state.get("explanation")
    if explanation and explanation["rejection_reasons"]:
        st.markdown("**주요 위험 요인** (자세한 내용은 XAI 분석 탭)")
        for r in explanation["rejection_reasons"][:3]:
            st.markdown(f"- {r['explanation']}")


# ---------------------------------------------------------------------------
# 2. XAI 분석
# ---------------------------------------------------------------------------
def waterfall(exp: dict, top: int = 10) -> go.Figure:
    shap = exp["shap_values"]
    names, values = exp["feature_names_kr"], exp["feature_values"]
    ordered = sorted(shap, key=lambda f: -abs(shap[f]))
    shown, rest = ordered[:top], ordered[top:]
    labels = ["기준값 (평균 고객)"]
    for f in shown:
        v = values.get(f)
        labels.append(f"{short(names.get(f, f))} = {'없음' if v is None else round(v, 2)}")
    x = [exp["base_value"]] + [shap[f] for f in shown]
    measure = ["absolute"] + ["relative"] * len(shown)
    if rest:
        labels.append(f"기타 {len(rest)}개 변수")
        x.append(sum(shap[f] for f in rest))
        measure.append("relative")
    labels.append(f"최종 (PD {exp['default_probability'] * 100:.2f}%)")
    x.append(0)
    measure.append("total")
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=measure,
            y=labels,
            x=x,
            increasing={"marker": {"color": RISK_UP}},
            decreasing={"marker": {"color": RISK_DOWN}},
            totals={"marker": {"color": NEUTRAL}},
            connector={"line": {"color": "#b0bec5"}},
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="SHAP Waterfall (부도 log-odds, 빨강 = 위험 증가)", showlegend=False)
    return fig


def tab_xai() -> None:
    glob = fetch("explain_global")
    left, right = st.columns(2)
    with left:
        df = pd.DataFrame(glob["features"]).sort_values("mean_abs_shap")
        fig = go.Figure(
            go.Bar(
                x=df["mean_abs_shap"],
                y=[short(n) for n in df["feature_name_kr"]],
                orientation="h",
                marker_color=SERIES[0],
                hovertext=df["actionability"],
            )
        )
        fig.update_layout(
            title="SHAP Summary · 전역 변수 중요도 (평균 |SHAP|)", xaxis_title="평균 |SHAP|"
        )
        chart(fig, height=420)
        source_caption(glob["is_dummy"], glob["source"], glob["note"])
    with right:
        exp = st.session_state.get("explanation")
        if not exp:
            st.info(
                "고객 심사 탭에서 심사를 실행하면 "
                "개별 고객의 Waterfall Plot 과 거절 사유가 표시됩니다."
            )
            return
        chart(waterfall(exp), height=420)
        st.caption(
            f"기준값 {exp['base_value']:.3f} + SHAP 합 = 최종 log-odds → 부도 확률 "
            f"{exp['default_probability'] * 100:.2f}% · 실제 데이터 (배포 모델)"
        )

    title = (
        "거절 사유 Top 5"
        if exp["approval_status"] == "REJECTED"
        else "주요 위험 요인 Top 5 (승인 고객)"
    )
    st.subheader(title)
    st.caption(
        f"부도 위험을 높인 변수 {exp['total_rejection_factors']}개 중 기여도가 큰 순서. "
        "기여도는 승인 방향 기준 log-odds (음수 = 거절 쪽)."
    )
    for r in exp["rejection_reasons"]:
        st.markdown(
            f"**{r['rank']}. {r['feature_name_kr']}** · `{r['actionability_label']}`  \n"
            f"{r['explanation']}"
        )
    if not exp["rejection_reasons"]:
        st.success("부도 위험을 눈에 띄게 높인 변수가 없습니다.")


# ---------------------------------------------------------------------------
# 3. 공정성
# ---------------------------------------------------------------------------
ATTRIBUTES = {"gender": "성별 (시뮬레이션)", "age_band": "연령대"}


def tab_fairness() -> None:
    fair = fetch("fairness")
    th = fair["thresholds"]
    lo, hi = th["di_ratio_bounds"]
    cols = st.columns(2)
    for column, (attr, title) in zip(cols, ATTRIBUTES.items(), strict=True):
        s = fair["attributes"][attr]
        with column:
            st.subheader(title)
            groups = list(s["groups"])
            rates = [s["groups"][g]["approval_rate"] for g in groups]
            ci = [
                s["groups"][g]["approval_ci95"] or [r, r]
                for g, r in zip(groups, rates, strict=True)
            ]
            fig = go.Figure(
                go.Bar(
                    x=groups,
                    y=rates,
                    marker_color=SERIES[0],
                    text=[f"{r * 100:.1f}%" for r in rates],
                    textposition="outside",
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": [c[1] - r for c, r in zip(ci, rates, strict=True)],
                        "arrayminus": [r - c[0] for c, r in zip(ci, rates, strict=True)],
                    },
                )
            )
            fig.update_layout(
                title="집단별 승인율 (95% 신뢰구간)",
                yaxis={"range": [0, 1.05], "tickformat": ".0%"},
            )
            chart(fig, height=320)
            m1, m2, m3 = st.columns(3)
            m1.metric(
                "DI Ratio",
                f"{s['di_ratio']:.3f}",
                "통과" if s["di_pass"] else "기준 위반",
                delta_color="normal" if s["di_pass"] else "inverse",
                border=True,
                help=f"집단 승인율 최소/최대, 허용 범위 {lo}~{hi}",
            )
            m2.metric(
                "EO · TPR 차이",
                f"{s['equalized_odds_tpr_diff']:.3f}",
                border=True,
                help="상환자 중 승인 비율의 집단 간 최대 차이 "
                f"(기준 ≤ {th['equalized_odds_max_diff']})",
            )
            m3.metric(
                "EO · FPR 차이",
                f"{s['equalized_odds_fpr_diff']:.3f}",
                "통과" if s["eo_pass"] else "기준 위반",
                delta_color="normal" if s["eo_pass"] else "inverse",
                border=True,
                help="부도자 중 승인 비율의 집단 간 최대 차이 "
                f"(기준 ≤ {th['equalized_odds_max_diff']})",
            )
            for w in s["warnings"]:
                st.warning(w)
    source_caption(fair["is_dummy"], fair["source"], fair["note"])

    st.divider()
    st.subheader("Bias Mitigation 전/후 비교")
    mit = fetch("mitigation")
    names = [ATTRIBUTES[a] for a in mit["before"]]
    left, right = st.columns(2)
    for column, metric, title, line in (
        (left, lambda s: s["di_ratio"], "DI Ratio (높을수록 공정)", lo),
        (
            right,
            lambda s: max(s["equalized_odds_tpr_diff"], s["equalized_odds_fpr_diff"]),
            "Equalized Odds 최대 차이 (낮을수록 공정)",
            th["equalized_odds_max_diff"],
        ),
    ):
        fig = go.Figure()
        for (side, label), color in zip(
            (
                ("before", "적용 전"),
                ("after", "적용 후 (더미)" if mit["after_is_dummy"] else "적용 후"),
            ),
            SERIES,
            strict=True,
        ):
            fig.add_bar(
                name=label, x=names, y=[metric(mit[side][a]) for a in mit[side]], marker_color=color
            )
        fig.add_hline(y=line, line_dash="dash", line_color=RISK_UP, annotation_text=f"기준 {line}")
        fig.update_layout(title=title, barmode="group")
        with column:
            chart(fig, height=320)
    source_caption(False, mit["source"], "적용 전")
    source_caption(mit["after_is_dummy"], note=f"적용 후 ({mit['method']}): {mit['note']}")


# ---------------------------------------------------------------------------
# 4. 모델 성능
# ---------------------------------------------------------------------------
def tab_performance() -> None:
    m = fetch("metrics")
    cols = st.columns(5)
    for column, (label, key) in zip(
        cols,
        (
            ("AUC-ROC", "auc"),
            ("KS", "ks"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1", "f1"),
        ),
        strict=True,
    ):
        ci = m.get(f"{key}_ci95")
        column.metric(
            label,
            f"{m[key]:.3f}",
            border=True,
            help=f"95% 신뢰구간 {ci[0]:.3f}~{ci[1]:.3f}" if ci else None,
        )
    cols = st.columns(4)
    cols[0].metric(
        "씬파일러 AUC 향상",
        f"{m['thin_auc_lift']:+.3f}",
        border=True,
        help=f"대안 데이터 포함 {m['thin_auc']:.3f} vs 미포함 {m['thin_auc_gmsc_only']:.3f}",
    )
    cols[1].metric("PSI", f"{m['psi']:.4f}", border=True)
    cols[2].metric("승인율", f"{m['approval_rate'] * 100:.1f}%", border=True)
    cols[3].metric("test 행 수", f"{m['rows']:,}", border=True)
    source_caption(
        m["is_dummy"], m["source"], f"모델 {m['model_version']}, {m['evaluated_on']} 세트"
    )

    left, mid, right = st.columns(3)
    with left:
        roc = m["roc_curve"]
        fig = go.Figure()
        fig.add_scatter(
            x=roc["fpr"],
            y=roc["tpr"],
            mode="lines",
            name=f"모델 (AUC {m['auc']:.3f})",
            line={"color": SERIES[0], "width": 3},
        )
        fig.add_scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="무작위", line={"dash": "dash", "color": NEUTRAL}
        )
        fig.update_layout(
            title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", legend={"x": 0.45, "y": 0.1}
        )
        chart(fig)
        source_caption(roc["is_dummy"], note=roc["note"])
    with mid:
        pr = m["pr_curve"]
        fig = go.Figure()
        fig.add_scatter(
            x=pr["recall"],
            y=pr["precision"],
            mode="lines",
            name="모델",
            line={"color": SERIES[0], "width": 3},
        )
        fig.add_hline(
            y=pr["baseline"],
            line_dash="dash",
            line_color=NEUTRAL,
            annotation_text=f"부도율 {pr['baseline']:.3f}",
        )
        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            showlegend=False,
        )
        chart(fig)
        source_caption(pr["is_dummy"], note=pr["note"])
    with right:
        cm = m["confusion_matrix"]
        z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=["예측: 승인", "예측: 거절"],
                y=["실제: 상환", "실제: 부도"],
                colorscale="Blues",
                text=[[f"{v:,}" for v in row] for row in z],
                texttemplate="%{text}",
                showscale=False,
            )
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title="혼동행렬 (양성 = 부도 = 거절)")
        chart(fig)
        source_caption(cm["is_dummy"], note=cm["note"])

    st.subheader("등급별 분포 (test 세트)")
    grades = pd.DataFrame(m["grade_table"]).rename(
        columns={
            "grade": "등급",
            "rows": "고객 수",
            "share": "비중",
            "mean_pd": "평균 PD",
            "default_rate": "실제 부도율",
            "min_score": "최저 점수",
            "max_score": "최고 점수",
        }
    )
    grades["위험도"] = grades["등급"].map(lambda g: RISK_STYLE[RISK_LEVELS.get(g, "high")][0])
    st.dataframe(
        grades,
        hide_index=True,
        width="stretch",
        column_config={
            "비중": st.column_config.NumberColumn(format="percent"),
            "평균 PD": st.column_config.NumberColumn(format="percent"),
            "실제 부도율": st.column_config.NumberColumn(format="percent"),
        },
    )


# ---------------------------------------------------------------------------
# 5. 통계 검증
# ---------------------------------------------------------------------------
def tab_statistics() -> None:
    a = fetch("anova")
    for key in ("test_1", "test_2"):
        t = a[key]
        st.subheader(t["test_name"])
        left, right = st.columns([3, 2])
        with left:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "F 통계량": t["f_statistic"],
                            "p-value": fmt_p(t["p_value"]),
                            "η² (효과 크기)": t["effect_size_eta_squared"],
                            "해석": t["effect_size_ko"],
                            "유의 (p < α)": "예" if t["is_significant"] else "아니오",
                            "α": t["alpha"],
                            "집단당 표본": ", ".join(str(n) for n in t["group_sizes"].values()),
                        }
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            if t["post_hoc_tukey"]:
                st.markdown("**Tukey HSD 사후 검정**")
                tukey = pd.DataFrame(t["post_hoc_tukey"]).rename(
                    columns={
                        "comparison": "비교",
                        "mean_diff": "평균 차이",
                        "p_adj": "보정 p",
                        "ci_lower": "CI 하한",
                        "ci_upper": "CI 상한",
                        "significant": "유의",
                    }
                )
                tukey["보정 p"] = tukey["보정 p"].map(fmt_p)
                st.dataframe(tukey, hide_index=True, width="stretch")
            else:
                st.caption("p ≥ α 이므로 사후 검정(Tukey HSD)을 하지 않습니다.")
        with right:
            means = t["group_means"]
            low, high = min(means.values()), max(means.values())
            pad = max((high - low) * 0.5, 0.005)
            fig = go.Figure(
                go.Bar(
                    x=list(means),
                    y=list(means.values()),
                    marker_color=SERIES[0],
                    text=[f"{v:.4f}" for v in means.values()],
                    textposition="outside",
                )
            )
            fig.update_layout(title="집단별 평균 AUC", yaxis={"range": [low - pad, high + pad]})
            chart(fig, height=300)
    source_caption(a["is_dummy"], a["source"], a["note"])


TABS = {
    "고객 심사": tab_review,
    "XAI 분석": tab_xai,
    "공정성": tab_fairness,
    "모델 성능": tab_performance,
    "통계 검증": tab_statistics,
}

for tab, render in zip(st.tabs(list(TABS)), TABS.values(), strict=True):
    with tab:
        try:
            render()
        except (ApiError, requests.RequestException) as exc:
            st.error(f"API 호출에 실패했습니다: {exc}")
