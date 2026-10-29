"""Fairness check of the shipped model on the test set.

    python -m src.fairness

Writes reports/fairness/fairness_metrics.json (path in configs/config.yaml, fairness.output).
Takes a few seconds: it only scores the test set, it does not train.
"""

from __future__ import annotations

import json

from src.config import get_settings, load_config
from src.fairness.data import load_split, protected_groups, score, target_column
from src.fairness.metrics import fairness_report


def main() -> None:
    settings = get_settings()
    cfg = load_config()["fairness"]
    test = load_split("test")
    decisions = score(test)
    groups = protected_groups(test)
    y = test[target_column()]

    report = {}
    for attribute, expected in cfg["protected_attributes"].items():
        found = sorted(groups[attribute].dropna().unique())
        if found != sorted(expected):
            raise ValueError(f"{attribute}: groups {found} differ from config {expected}")
        report[attribute] = fairness_report(
            decisions["approved"], y, groups[attribute], cfg, attribute
        )
    report["rule"] = {
        "di_band": cfg["di_ratio_bounds"],
        "eo_limit": cfg["equalized_odds_max_diff"],
        "positive_for_tpr_fpr": "approved; TPR = approval rate among repaid, FPR = among defaulted",
    }
    report["source"] = {
        "model_file": settings.model_path.relative_to(settings.model_dir.parent).as_posix(),
        "split": "test",
        "rows": len(test),
        "approval_rate": round(float(decisions["approved"].mean()), 4),
    }

    out = settings.reports_dir / cfg["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for attribute in cfg["protected_attributes"]:
        r = report[attribute]
        print(
            f"{attribute:9s} DI {r['di_ratio_min_over_max']:.4f} (pass {r['di_pass']})  "
            f"EO TPR {r['eo_tpr_diff']:.4f} FPR {r['eo_fpr_diff']:.4f} (pass {r['eo_pass']})"
        )
    print(f"Saved {out.relative_to(settings.reports_dir.parent).as_posix()}")


if __name__ == "__main__":
    main()
