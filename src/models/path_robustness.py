"""Approval probability of DiCE approval paths when the model is retrained.

    python -m src.models.path_robustness      (about 10 to 15 minutes in the dev container)

The shipped model approves a path for sure (its score is at or above the cutoff), but a credit
model is retrained over time, and a slightly different model may score the same applicant lower.
Here the shipped setup is refit `models` times, each time on a random `sample_share` of
train+validation (the audit's stability check in notebooks/audit.py did this with 3 models). Every
DiCE path of the rejected test applicants (reports/recourse/paths_<h>m.csv, h =
api.recourse_default_horizon) is rescored by each refit. The share of refits that still approve a
path is its approval rate. The rate depends mostly on how far the path score is above the cutoff,
so an increasing (isotonic) curve of rate against the shipped model's path score is fitted. The API
reads the curve to show an approval probability for any path (POST /recourse).

Writes reports/recourse/approval_probability.json (api.reports.recourse_approval in
configs/config.yaml). Educational use only. Not for real financial decisions.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.config import get_settings, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def path_frames(
    test: pd.DataFrame, features: list[str], paths: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(X0, X1, found): original and proposed model inputs of the paths DiCE found.

    paths: a paths_<h>m.csv frame indexed by row_id; new_<feature> columns hold proposed values.
    """
    found = paths[paths["dice_found"].astype(bool)]
    X0 = test.loc[found.index, features].astype(float)
    X1 = X0.copy()
    for col in (c for c in paths.columns if c.startswith("new_")):
        has = found[col].notna()
        X1.loc[has[has].index, col.removeprefix("new_")] = found.loc[has, col].astype(float)
    return X0, X1, found


def approval_curve(scores: np.ndarray, rates: np.ndarray) -> dict:
    """Increasing step curve of approval rate against path score (isotonic regression)."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(increasing=True, y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(np.asarray(scores, float), np.asarray(rates, float))
    return {
        "score": [float(x) for x in iso.X_thresholds_],
        "probability": [round(float(y), 4) for y in iso.y_thresholds_],
    }


def probability_at(curve: dict, score: float) -> float:
    """Approval probability of a path score: linear between curve points, flat beyond the ends."""
    return float(np.interp(score, curve["score"], curve["probability"]))


def score_bins(scores: np.ndarray, approved: np.ndarray, edges: list[int]) -> list[dict]:
    """Paths per path-score band: count, mean approval rate, share approved by every refit."""
    rate = approved.mean(axis=0)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (scores >= lo) & (scores < hi)
        if sel.any():
            out.append({
                "min_score": int(lo),
                "max_score": int(hi) - 1,
                "paths": int(sel.sum()),
                "approval_rate": round(float(rate[sel].mean()), 4),
                "approved_by_all_models": round(float(approved[:, sel].all(axis=0).mean()), 4),
            })
    return out


def main() -> None:
    from sklearn.metrics import roc_auc_score

    from src.explainer.recourse import Scorer
    from src.fairness.data import load_splits, target_column
    from src.fairness.mitigation import base_cutoff, fit_shipped_setup
    from src.models.predictor import get_model
    from src.models.scoring import load_config as load_scoring

    settings = get_settings()
    cfg = load_config()
    acfg, rcfg = cfg["recourse_approval"], cfg["explainer"]["recourse"]
    horizon = cfg["api"]["recourse_default_horizon"]
    model = get_model()
    cutoff = base_cutoff(model.scoring_cfg)
    link = rcfg["link_income_debt"]

    splits = load_splits()
    test = splits["test"]
    train_val = pd.concat([splits["train"], splits["validation"]])
    paths_file = settings.reports_dir / rcfg["output_dir"] / f"paths_{horizon}m.csv"
    X0, X1, found = path_frames(test, model.features, pd.read_csv(paths_file).set_index("row_id"))
    shipped = Scorer(model.pipeline, model.features, model.scoring_cfg, link).score(X1, X0)
    if not (shipped == found["score_after"].to_numpy().astype(int)).all():
        raise ValueError(f"{paths_file.name} was not built with the shipped model: "
                         "rerun python -m src.explainer.recourse")
    thin = found["thin_filer"].to_numpy() == 1

    X_te, y_te = test[model.features].astype(float), test[target_column()].to_numpy()
    approved = np.zeros((acfg["models"], len(X1)), bool)
    aucs = []
    for k in range(acfg["models"]):
        start = time.perf_counter()
        sub = train_val.sample(frac=acfg["sample_share"], random_state=acfg["seed"] + k)
        est = fit_shipped_setup(sub)
        approved[k] = Scorer(est, model.features, model.scoring_cfg, link).score(X1, X0) >= cutoff
        aucs.append(round(float(roc_auc_score(y_te, est.predict_proba(X_te)[:, 1])), 4))
        logger.info("refit %d/%d: test AUC %.4f, paths still approved %.4f (%.0fs)", k + 1,
                    acfg["models"], aucs[-1], approved[k].mean(), time.perf_counter() - start)

    rate = approved.mean(axis=0)
    report = {
        "horizon_months": horizon,
        "approval_cutoff": cutoff,
        "target_score": int(load_scoring()["recourse_target_score"]),
        "models": acfg["models"],
        "sample_share": acfg["sample_share"],
        "seed": acfg["seed"],
        "paths": int(len(X1)),
        "overall": {
            "approval_rate": round(float(rate.mean()), 4),
            "approval_rate_thin": round(float(rate[thin].mean()), 4) if thin.any() else None,
            "approved_by_all_models": round(float(approved.all(axis=0).mean()), 4),
            "approval_rate_per_model": [round(float(r), 4) for r in approved.mean(axis=1)],
        },
        "test_auc_per_model": aucs,
        "curve": approval_curve(shipped, rate),
        "bins": score_bins(shipped, approved, acfg["score_bins"]),
        "source": {
            "model_file": settings.model_path.relative_to(settings.model_dir.parent).as_posix(),
            "paths_file": paths_file.relative_to(settings.reports_dir.parent).as_posix(),
        },
    }
    out = settings.reports_dir / cfg["api"]["reports"]["recourse_approval"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    o = report["overall"]
    print(f"{report['paths']} paths ({horizon} months), {acfg['models']} refits: mean approval "
          f"{o['approval_rate']:.4f}, approved by all {o['approved_by_all_models']:.4f}")
    for b in report["bins"]:
        print(f"  score {b['min_score']}-{b['max_score']}: {b['paths']} paths, "
              f"approval {b['approval_rate']:.4f}, all models {b['approved_by_all_models']:.4f}")
    print(f"Saved {out.relative_to(settings.reports_dir.parent).as_posix()}")


if __name__ == "__main__":
    main()
