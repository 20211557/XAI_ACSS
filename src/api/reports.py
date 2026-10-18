"""API 가 읽는 학습·점검 산출물 (reports/). 파일 위치는 configs/config.yaml 의 api.reports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import get_settings, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def report_path(key: str) -> Path:
    return get_settings().reports_dir / load_config()["api"]["reports"][key]


def source_of(key: str) -> str:
    """응답의 source 필드: 프로젝트 기준 상대 경로 (예: reports/final_summary.json)."""
    return f"reports/{load_config()['api']['reports'][key]}"


def read_json(key: str) -> dict:
    return json.loads(report_path(key).read_text(encoding="utf-8"))


def read_csv(key: str) -> pd.DataFrame:
    return pd.read_csv(report_path(key))


def confusion_matrix(summary: dict) -> dict:
    """반올림된 test 지표(행 수, 부도율, 정밀도, 재현율)로 혼동행렬을 복원한다 (양성 = 부도 = 거절).

    지표가 소수 4자리로 반올림되어 있어 각 칸은 몇 건 이내의 근사값이다."""
    t = summary["test"]
    n = t["rows"]
    positives = round(n * summary["default_rate_test"])
    tp = round(t["recall"] * positives)
    predicted = round(tp / t["precision"]) if t["precision"] else 0
    fp, fn = predicted - tp, positives - tp
    return {"tn": n - tp - fp - fn, "fp": fp, "fn": fn, "tp": tp}


def fairness_summary(attribute: str, audit: dict, cfg: dict) -> dict:
    """보호 속성 하나의 DI Ratio, Equalized Odds, 집단별 지표와 기준 위반 경고."""
    a = audit[attribute]
    lo, hi = cfg["di_ratio_bounds"]
    eo_max = cfg["equalized_odds_max_diff"]
    di, tpr_diff, fpr_diff = a["di_ratio_min_over_max"], a["eo_tpr_diff"], a["eo_fpr_diff"]
    warnings = []
    if not lo <= di <= hi:
        warnings.append(f"{attribute}: DI Ratio {di:.3f} 이 허용 범위 [{lo}, {hi}] 를 벗어났습니다")
    for name, diff in (("TPR", tpr_diff), ("FPR", fpr_diff)):
        if diff > eo_max:
            warnings.append(
                f"{attribute}: Equalized Odds {name} 차이 {diff:.3f} 가 기준 {eo_max} 를 넘었습니다"
            )
    for w in warnings:
        logger.warning(w)
    groups = {
        g: {
            "n": v["n"],
            "default_rate": v["default_rate"],
            "approval_rate": v["approval_rate"],
            "approval_ci95": v.get("approval_ci95"),
            "tpr": v["tpr_approval_if_repaid"],
            "fpr": v["fpr_approval_if_defaulted"],
        }
        for g, v in a["groups"].items()
    }
    return {
        "attribute": attribute,
        "di_ratio": di,
        "equalized_odds_tpr_diff": tpr_diff,
        "equalized_odds_fpr_diff": fpr_diff,
        "di_pass": lo <= di <= hi,
        "eo_pass": max(tpr_diff, fpr_diff) <= eo_max,
        "groups": groups,
        "warnings": warnings,
    }
