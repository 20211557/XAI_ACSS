"""DiCE 승인 경로 서빙.

- POST /recourse: 입력한 고객 한 명의 경로를 배포 모델로 바로 찾는다 (보통 1초 안팎).
- GET /recourse/summary: src.explainer.recourse 가 test 세트 거절자 전체에 대해 미리 계산한
  기간별 커버리지 (reports/recourse/summary.json).

경로 탐색은 src/explainer/recourse.py 의 함수를 그대로 쓴다. 배치 실행과 같은 규칙: dice_vary
변수만, 허용 방향, 정수 단계, 기간 안, 변수 max_changed_features 개 이하,
목표 recourse_target_score. 이 모듈은 결과를 응답 형태로 바꾸기만 한다.
교육용이며 실제 금융 의사결정에 쓰지 않는다.
"""

from __future__ import annotations

import math
import threading
import time
from functools import lru_cache

import pandas as pd

from src.api import reports
from src.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# dice_path 는 random / numpy 의 전역 시드를 설정하므로 탐색은 한 번에 하나씩 한다
_LOCK = threading.Lock()


@lru_cache
def _setup() -> dict:
    """배포 모델의 점수 계산기와 DiCE 가 바꿀 수 있는 변수 규칙 (서버당 한 번)."""
    from src.explainer.recourse import Scorer, load_specs
    from src.models.predictor import get_model
    from src.models.scoring import load_config as load_scoring

    model = get_model()
    rcfg = load_config()["explainer"]["recourse"]
    scoring = load_scoring()
    link = rcfg["link_income_debt"]
    return {
        "rcfg": rcfg,
        "scorer": Scorer(model.pipeline, model.features, model.scoring_cfg, link),
        "specs": load_specs(model.features),
        "target": int(scoring["recourse_target_score"]),
    }


def warm_up() -> None:
    """dice_ml import 와 점수 계산기 준비를 서버 시작 때 해 둔다 (첫 요청이 느려지지 않도록)."""
    import dice_ml  # noqa: F401

    _setup()


def rules() -> dict:
    """목표 점수, 변수 수 상한, 기간 후보 (응답과 검증용)."""
    from src.models.scoring import load_config as load_scoring

    rcfg = load_config()["explainer"]["recourse"]
    return {
        "target": int(load_scoring()["recourse_target_score"]),
        "max_vars": int(rcfg["max_changed_features"]),
        "horizons": [int(h) for h in rcfg["horizons_months"]],
    }


def approval_cutoff(scoring_cfg: dict) -> int:
    """승인 등급의 최저 점수 (configs/scoring.yaml, 475)."""
    approved = scoring_cfg["approval"]["approve_grades"]
    return int(min(g["min_score"] for g in scoring_cfg["grades"] if g["grade"] in approved))


def _labels() -> tuple[dict, dict]:
    from src.explainer.reasons import feature_metadata

    cfg = load_config()["explainer"]
    return feature_metadata()["features"], cfg["value_format"]


def _short(label: str) -> str:
    from src.explainer.reasons import short_label

    return short_label(label)


def _has_batchim(word: str) -> bool:
    """Last Hangul syllable has a final consonant (decides 은/는, 을/를)."""
    code = ord(word.strip()[-1]) - 0xAC00
    return 0 <= code <= 11171 and code % 28 != 0


def _fixed_examples(features: list[str], meta: dict) -> list[str]:
    """경로에서 뺀 항목 예시: 불변 1개 + 감소 불가 1개 (예: 연령, 30~59일 연체 횟수)."""
    out = []
    for cls in ("IMMUTABLE", "NON_DECREASING"):
        names = [f for f in features if meta.get(f, {}).get("actionability") == cls]
        if names:
            out.append(_short(meta[names[0]]["label_ko"]))
    return out


def _blockers(shap: dict[str, float] | None, meta: dict, n: int = 3) -> list[str]:
    """경로로 바꿀 수 없는 항목 중 지금 점수를 가장 많이 깎는 것 (SHAP 순)."""
    if not shap:
        return []
    fixed = [f for f, v in shap.items() if v > 0 and not meta.get(f, {}).get("dice_vary")]
    return [_short(meta[f]["label_ko"]) for f in sorted(fixed, key=lambda f: -shap[f])[:n]]


def _change_why(c: dict, direction: int, kind: str, now: int, target: int, single: bool) -> str:
    """항목 하나를 바꾸는 이유를 쉬운 말로: 지금 상태, 무엇을 얼마나, 점수가 어떻게 되는지.

    시간 의존 항목은 configs/actionability.yaml 규칙대로 행동을 권하지 않고 기간만 말한다."""
    from src.explainer.reasons import ro_particle, subject_particle

    name = _short(c["feature_name_kr"])
    cur, new, months = c["current_text"], c["proposed_text"], f"{c['months']:g}"
    parts = []
    shap = c["shap_now"]
    if shap is not None:
        parts.append("지금 점수를 깎고 있는 항목입니다." if shap > 0
                     else "지금도 나쁘지 않지만, 더 좋아지면 점수가 더 오릅니다.")
    up = direction > 0
    if c["actionability"] == "TIME_ONLY":
        parts.append(f"시간이 지나면 저절로 늘어나는 항목으로, {months}개월이 지나면 "
                     f"{cur}에서 {new}{subject_particle(new)} 됩니다.")
    elif c["actionability"] == "CONDITIONAL":
        verb = ("높아지면" if kind == "pct" else "늘면") if up else (
            "낮아지면" if kind == "pct" else "줄면")
        parts.append(f"{cur}에서 {new}{ro_particle(new)} {verb} 되고, 약 {months}개월을 "
                     "잡았습니다. 본인 노력만으로 정하기 어려운 항목이라 어려운 변화로 봤습니다.")
    else:
        verb = ("높이면" if kind in ("pct", "decimal") else "늘리면") if up else (
            "낮추면" if kind in ("pct", "decimal") else "줄이면")
        parts.append(f"{cur}에서 {new}{ro_particle(new)} {verb} 되고, 약 {months}개월이 "
                     "걸립니다.")
    alone, without = c["score_if_only_this"], c["score_without_this"]
    if single:
        parts.append(f"이 변화 하나로 점수가 {now}점에서 {alone}점이 되어 목표 {target}점을 "
                     "넘습니다.")
    elif without < target:
        parts.append(f"이것만으로는 {alone}점 ({alone - now:+d}점) 이지만 꼭 필요합니다. "
                     f"이것이 빠지면 {without}점에 그쳐 목표 {target}점에 못 미칩니다.")
    else:
        parts.append(f"이것만으로 {alone}점 ({alone - now:+d}점) 이 됩니다.")
    topic = "은" if _has_batchim(name) else "는"
    return f"{name}{topic} " + " ".join(parts)


def approval_table() -> dict | None:
    """재학습 모델 기준 승인 유지 확률 표 (python -m src.models.path_robustness). 없으면 None."""
    try:
        return reports.read_json("recourse_approval")
    except (FileNotFoundError, ValueError):
        return None


def approval_probability(score: int, table: dict | None) -> float | None:
    """경로 후 점수에서의 승인 유지 확률 (표의 증가 곡선을 따라 읽는다)."""
    if not table:
        return None
    from src.models.path_robustness import probability_at

    return round(probability_at(table["curve"], score), 4)


def _path_summary(k: int, months: float, now: int, after: int, target: int, cutoff: int,
                  fixed: list[str], prob: float | None, models: int | None) -> str:
    """경로를 이렇게 정한 이유 (한두 문장, 쉬운 말)."""
    skip = f"{', '.join(fixed)}처럼 바꿀 수 없는 항목은 빼고, " if fixed else ""
    text = (f"{skip}바꿀 수 있는 항목 {k}가지만으로 점수를 {now}점에서 {after}점으로 올리는 "
            f"가장 쉬운 방법을 골랐습니다. 여러 항목은 함께 진행하므로 약 {months:g}개월이면 "
            "됩니다." if k > 1 else
            f"{skip}바꿀 수 있는 항목 1가지만으로 점수를 {now}점에서 {after}점으로 올리는 "
            f"가장 쉬운 방법을 골랐습니다. 약 {months:g}개월이면 됩니다.")
    if target > cutoff:
        text += (f" 승인 기준은 {cutoff}점이지만 {target}점을 목표로 잡아, 나중에 평가가 조금 "
                 "달라져도 승인이 유지되도록 여유를 두었습니다.")
    text += " 그래도 이 경로를 따른다고 승인이 보장되지는 않습니다."
    if prob is not None and models:
        text += (f" 평가 모델을 조금씩 다른 데이터로 {models}번 다시 만들어 이 점수의 경로를 "
                 f"확인해 보면, 승인이 유지될 확률은 약 {prob * 100:.0f}% 입니다.")
    return text


def find_path(X: pd.DataFrame, horizon: int, shap: dict[str, float] | None = None,
              cutoff: int | None = None) -> dict:
    """모델 입력 한 행(X)의 경로. status, path (응답용 dict 또는 None), blockers, elapsed_seconds.

    shap: 이 고객의 변수별 SHAP (부도 log-odds). 경로 설명과 '경로 없음' 사유에 쓴다."""
    from src.explainer.reasons import format_value
    from src.explainer.recourse import DEBT_RATIO, INCOME, _corner_seeds, dice_path

    s = _setup()
    rcfg, scorer, specs, target = s["rcfg"], s["scorer"], s["specs"], s["target"]
    x0 = X.iloc[0]
    # 같은 입력이면 같은 경로가 나오도록 시드를 고정한다 (배치 실행의 행 번호 자리에 0)
    seed = rcfg["seed"] * 100_003 + horizon
    start = time.perf_counter()
    with _LOCK:
        # 코너 점(변수 몇 개를 기간 끝까지 바꾼 점) 중 목표에 닿는 것. 없으면 경로가 있을 수 없다
        seeds = _corner_seeds(scorer, x0, specs, horizon, target, rcfg["max_changed_features"])
        path = dice_path(scorer, x0, specs, horizon, target, rcfg, seed, seeds) if seeds else None
    elapsed = round(time.perf_counter() - start, 3)
    meta, fmt = _labels()
    if path is None:
        return {"status": "not_found" if seeds else "not_feasible", "path": None,
                "blockers": _blockers(shap, meta), "elapsed_seconds": elapsed}

    X0 = pd.DataFrame([x0.to_numpy(float)], columns=x0.index)

    def score_with(values: dict[str, float]) -> int:
        Xc = X0.copy()
        for v, val in values.items():
            Xc.loc[0, v] = val
        return int(scorer.score(Xc, X0)[0])

    values = {v: float(path["values"][v]) for v in path["steps"]}
    now = score_with({})
    changes = []
    for v in specs:  # 모델 변수 순서
        if v not in path["steps"]:
            continue
        n, cur, new = path["steps"][v], float(x0[v]), values[v]
        changes.append({
            "feature": v,
            "feature_name_kr": meta[v]["label_ko"],
            "current_value": cur,
            "proposed_value": new,
            "current_text": format_value(cur, fmt.get(v, {})),
            "proposed_text": format_value(new, fmt.get(v, {})),
            "steps": int(n),
            "months": float(n * specs[v].months_per_step),
            "difficulty": float(specs[v].difficulty),
            "actionability": meta[v].get("actionability", ""),
            "shap_now": None if not shap or v not in shap else round(float(shap[v]), 4),
            "score_if_only_this": score_with({v: new}),
            "score_without_this": score_with({u: val for u, val in values.items() if u != v}),
        })
    single = len(changes) == 1
    for c in changes:
        kind = fmt.get(c["feature"], {}).get("kind", "decimal")
        c["why"] = _change_why(c, specs[c["feature"]].direction, kind, now, target, single)
    linked = None
    if rcfg["link_income_debt"] and INCOME in path["steps"]:
        inc0, inc1, dr = float(x0[INCOME]), float(path["values"][INCOME]), float(x0[DEBT_RATIO])
        if inc0 > 0 and inc1 > 0 and math.isfinite(dr):
            dr_new = float(path["values"].get(DEBT_RATIO, dr))
            linked = round(dr_new * inc0 / inc1, 6)
    after = int(path["score"])
    table = approval_table()
    prob = approval_probability(after, table)
    models = table["models"] if table else None
    summary = _path_summary(len(changes), float(path["months"]), now, after, target,
                            target if cutoff is None else cutoff,
                            _fixed_examples(scorer.features, meta), prob, models)
    return {
        "status": "found",
        "path": {
            "changes": changes,
            "months": float(path["months"]),
            "score_after": after,
            "cost": round(float(path["cost"]), 4),
            "linked_debt_ratio": linked,
            "summary": summary,
            "approval_probability": prob,
            "approval_models": models,
        },
        "blockers": [],
        "elapsed_seconds": elapsed,
    }


def _model_match(rejected: int) -> bool | None:
    """미리 계산한 거절자 수가 배포 모델의 test 세트 거절자 수와 맞는지.

    final_summary.json 의 승인율은 소수 4자리로 반올림되어 있어 2명까지 차이를 허용한다."""
    try:
        summary = reports.read_json("final_summary")
        rows = int(summary["test"]["rows"])
        expected = rows - round(rows * float(summary["approval_rate"]))
    except (FileNotFoundError, KeyError, ValueError):
        return None
    return abs(rejected - expected) <= 2


def summary(default_horizon: int, coverage_target: float) -> dict:
    """reports/recourse/summary.json 을 기간 순서의 목록으로 (변수 이름은 한국어 라벨)."""
    data = reports.read_json("recourse_summary")
    meta, _ = _labels()

    def label(path: str) -> list[str]:
        return [meta.get(v, {}).get("label_ko", v) for v in path.split(" + ")]

    horizons = []
    for h, r in sorted(data["horizons"].items(), key=lambda kv: int(kv[0])):
        horizons.append({
            "horizon_months": int(h),
            "rejected": r["rejected"],
            "dice_coverage": r["dice_coverage"],
            "dice_coverage_thin": r.get("dice_coverage_thin"),
            "dice_coverage_not_thin": r.get("dice_coverage_not_thin"),
            "corner_bound": r["corner_bound"],
            "median_months": r.get("median_months"),
            "vars_needed": r.get("vars_needed", {}),
            "top_paths": [{"features": p.split(" + "), "labels": label(p), "count": c}
                          for p, c in r.get("top_paths", {}).items()],
            "meets_target": r["dice_coverage"] >= coverage_target,
        })
    rejected = horizons[0]["rejected"] if horizons else 0
    table = approval_table()
    keys = ("horizon_months", "models", "sample_share", "paths", "overall", "bins")
    approval = None if table is None else {k: table[k] for k in keys}
    return {
        "approval": approval,
        "horizons": horizons,
        "default_horizon": default_horizon,
        "target_score": data["target_score"],
        "coverage_target": coverage_target,
        "max_changed_features": data["max_changed_features"],
        "model_match": _model_match(rejected) if horizons else None,
    }
