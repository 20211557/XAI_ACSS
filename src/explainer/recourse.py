"""Actionable Recourse with DiCE (genetic): approval paths for rejected test applicants, per time horizon.

    python -m src.explainer.recourse              # every horizon in explainer.recourse.horizons_months
    python -m src.explainer.recourse --limit 100  # pilot on the first 100 rejected applicants (separate files)

A path changes only model features with dice_vary: true in configs/actionability.yaml, in the allowed
direction, in whole steps, within the bounds, and by at most floor(horizon / months_per_step) steps.
Variables change in parallel, so a path takes as long as its slowest variable. A path counts when the
shipped model scores it at recourse_target_score (configs/scoring.yaml) or higher with at most
max_changed_features changed variables.

Income and debt are linked (link_income_debt): DebtRatio = debt payments / income, so when income rises
with debt held fixed, the ratio the model sees is DebtRatio x old income / new income.

For each horizon two coverages are reported:
- corner bound: some subset of at most max_changed_features variables, each moved to its horizon limit,
  reaches the target. All varied features are monotone in the model, so no path can exist without this.
- DiCE: the genetic search found such a path. Never above the corner bound.

Outputs (reports/<output_dir>/): paths_<h>m.csv per horizon, summary.json. One MLflow run.
Educational use only. Not for real financial decisions.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config

INCOME, DEBT_RATIO = "MonthlyIncome", "DebtRatio"


@dataclass(frozen=True)
class Spec:
    """How one feature may change, from configs/actionability.yaml."""

    name: str
    direction: int          # +1 increase, -1 decrease
    step: float             # additive step, or growth rate when multiplicative
    multiplicative: bool    # step given as "5%"
    lo: float
    hi: float
    months_per_step: float
    difficulty: float


def load_specs(features: list[str]) -> dict[str, Spec]:
    """Model features DiCE may vary, in model feature order."""
    from src.explainer.reasons import feature_metadata

    meta = feature_metadata()["features"]
    specs = {}
    for f in features:
        e = meta[f]
        if not e.get("dice_vary"):
            continue
        step = e["step"]
        mult = isinstance(step, str) and step.endswith("%")
        lo, hi = e["bounds"]
        specs[f] = Spec(
            name=f,
            direction=1 if e["direction"] == "increase" else -1,
            step=float(step.rstrip("%")) / 100 if mult else float(step),
            multiplicative=mult,
            lo=-np.inf if lo is None else float(lo),
            hi=np.inf if hi is None else float(hi),
            months_per_step=float(e["months_per_step"]),
            difficulty=float(e["difficulty"]),
        )
    return specs


def max_steps(spec: Spec, horizon: int) -> int:
    return int(math.floor(horizon / spec.months_per_step + 1e-9))


def apply_steps(x, spec: Spec, n):
    """Value after n steps in the allowed direction, clipped to the bounds; never moves backwards.
    NaN stays NaN."""
    x = np.asarray(x, float)
    n = np.asarray(n, float)
    y = x * (1 + spec.step) ** n if spec.multiplicative else x + spec.direction * spec.step * n
    y = np.clip(y, spec.lo, spec.hi)
    return np.maximum(y, x) if spec.direction > 0 else np.minimum(y, x)


def steps_needed(x0: float, x1: float, spec: Spec) -> int:
    """Whole steps needed to move from x0 to at least x1 (rounded up, so the change is never smaller)."""
    if spec.multiplicative:
        if x0 <= 0 or x1 <= x0:
            return 0
        raw = math.log(x1 / x0) / math.log(1 + spec.step)
    else:
        raw = spec.direction * (x1 - x0) / spec.step
    if raw <= 1e-9:
        return 0
    return math.ceil(raw - 1e-6)


def can_vary(x0: float, spec: Spec) -> bool:
    """False for missing values, zero income under a growth step, and values already at their limit."""
    if not np.isfinite(x0) or (spec.multiplicative and x0 <= 0):
        return False
    return bool(apply_steps(x0, spec, 1) != x0)


def link_income_debt(X: pd.DataFrame, X0: pd.DataFrame) -> pd.DataFrame:
    """DebtRatio the model sees when income changes and debt payments stay fixed.

    X: proposed values (DebtRatio here is the ratio at the old income, after any debt paydown).
    X0: original values, same index. Rows with missing or zero income keep DebtRatio as it is.
    """
    out = X.copy()
    inc0, inc1 = X0[INCOME].to_numpy(float), X[INCOME].to_numpy(float)
    ok = np.isfinite(inc0) & np.isfinite(inc1) & (inc0 > 0) & (inc1 > 0)
    dr = out[DEBT_RATIO].to_numpy(float).copy()
    dr[ok] = dr[ok] * inc0[ok] / inc1[ok]
    out[DEBT_RATIO] = dr
    return out


class Scorer:
    """Shipped model + score scale, with the income-debt link applied before scoring."""

    def __init__(self, pipeline, features: list[str], scoring_cfg: dict, link: bool) -> None:
        self.pipeline, self.features, self.cfg, self.link = pipeline, features, scoring_cfg, link

    def _pd(self, X: pd.DataFrame, X0: pd.DataFrame | None) -> np.ndarray:
        if self.link and X0 is not None:
            X = link_income_debt(X, X0)
        return self.pipeline.predict_proba(X[self.features].astype(float))[:, 1]

    def raw_score(self, X: pd.DataFrame, X0: pd.DataFrame | None = None) -> np.ndarray:
        """Unrounded, unclipped score (smooth target for the search)."""
        s = self.cfg["score"]
        p = np.clip(self._pd(X, X0), *s["pd_clip"])
        return s["base_score"] + s["pdo"] / np.log(2) * np.log((1 - p) / p / s["base_odds"])

    def score(self, X: pd.DataFrame, X0: pd.DataFrame | None = None) -> np.ndarray:
        """Official integer score (src.models.scoring.pd_to_score)."""
        from src.models.scoring import pd_to_score

        return pd_to_score(self._pd(X, X0), self.cfg)


# ---------------------------------------------------------------------------------------------
# Corner bound
# ---------------------------------------------------------------------------------------------


def corner_frame(X: pd.DataFrame, specs: dict[str, Spec], subset, horizon: int) -> pd.DataFrame:
    Xc = X.copy()
    for v in subset:
        Xc[v] = apply_steps(X[v].to_numpy(float), specs[v], max_steps(specs[v], horizon))
    return Xc


def corner_bound(scorer: Scorer, X: pd.DataFrame, specs: dict[str, Spec], horizon: int, target: int,
                 max_vars: int) -> pd.DataFrame:
    """Per row: whether some subset of <= max_vars variables at their horizon limit reaches target,
    the fewest variables that do, and whether all variables together do."""
    names = list(specs)
    best_k = np.zeros(len(X), int)
    for k in range(1, max_vars + 1):
        for sub in itertools.combinations(names, k):
            ok = scorer.score(corner_frame(X, specs, sub, horizon), X) >= target
            best_k = np.where((best_k == 0) & ok, k, best_k)
    all_ok = scorer.score(corner_frame(X, specs, names, horizon), X) >= target
    return pd.DataFrame({"corner_feasible": best_k > 0, "corner_min_vars": best_k,
                         "corner_all_vars": all_ok}, index=X.index)


# ---------------------------------------------------------------------------------------------
# DiCE search for one applicant
# ---------------------------------------------------------------------------------------------


def _batched_genetic_class():
    from dice_ml.explainer_interfaces.dice_genetic import DiceGenetic

    class BatchedGenetic(DiceGenetic):
        """DiceGenetic with a batched random start. The stock start scores one random point at a time
        until it has num_inits valid ones: tens of thousands of model calls for applicants far below
        the target. Here points are drawn and scored in batches, the corner points that reach the
        target are added, and the population is filled with random points if too few are valid."""

        init_batch = 2000
        init_max_samples = 100_000
        seeds: np.ndarray | None = None   # valid starting points (corner paths), DiCE feature order

        def do_random_init(self, num_inits, features_to_vary, query_instance, desired_class, desired_range):
            names = self.data_interface.feature_names
            precisions = self.data_interface.get_decimal_precisions()
            lo = np.array([self.feature_range[f][0] for f in names], float)
            hi = np.array([self.feature_range[f][1] for f in names], float)
            vary = np.array([f in features_to_vary for f in names])

            def draw(n):
                pts = np.tile(np.asarray(query_instance, float), (n, 1))
                for j in np.flatnonzero(vary):
                    pts[:, j] = np.round(np.random.uniform(lo[j], hi[j], n), precisions[j])
                return pts

            valid = [] if self.seeds is None else [self.seeds]
            spare = None
            drawn = 0
            while sum(len(v) for v in valid) < num_inits and drawn < self.init_max_samples:
                pts = draw(self.init_batch)
                drawn += len(pts)
                pred = np.asarray(self.predict_fn_scores(pts), float).reshape(-1)
                ok = (pred >= desired_range[0]) & (pred <= desired_range[1])
                valid.append(pts[ok])
                spare = pts[~ok] if spare is None else spare
            pop = np.concatenate(valid) if valid else np.empty((0, len(names)))
            pop = pop[:num_inits]
            if len(pop) < num_inits:
                fill = spare if spare is not None else draw(num_inits)
                pop = np.concatenate([pop, fill[: num_inits - len(pop)]])
            return pop

    return BatchedGenetic


class _DiceModel:
    """sklearn-style regressor for DiCE: the varied features in, the official score scale out
    (unrounded), with every other feature held at the applicant's value."""

    def __init__(self, scorer: Scorer, x0: pd.Series, varied: list[str]) -> None:
        self.scorer, self.x0, self.varied = scorer, x0, varied
        self.X0 = pd.DataFrame([x0.to_numpy(float)], columns=x0.index)

    def full(self, values) -> tuple[pd.DataFrame, pd.DataFrame]:
        arr = np.asarray(values, float).reshape(-1, len(self.varied))
        X = pd.DataFrame(np.repeat(self.X0.to_numpy(), len(arr), 0), columns=self.X0.columns)
        X[self.varied] = arr
        return X, pd.DataFrame(np.repeat(self.X0.to_numpy(), len(arr), 0), columns=self.X0.columns)

    def predict(self, D) -> np.ndarray:
        values = D[self.varied].to_numpy(float) if isinstance(D, pd.DataFrame) else D
        X, X0 = self.full(values)
        return self.scorer.raw_score(X, X0)


def _precision(x0: float, spec: Spec):
    whole = float(x0).is_integer() and float(spec.step).is_integer() and not spec.multiplicative
    return "int" if whole else ["float", 4]


def snap_and_trim(scorer: Scorer, x0: pd.Series, cand: dict[str, float], specs: dict[str, Spec],
                  horizon: int, target: int, max_vars: int) -> dict | None:
    """Turn a DiCE counterfactual into a path: whole steps (rounded up), at most max_vars variables
    (greedily drop the change whose removal keeps the highest score), then the fewest steps per
    variable that still reach the target. None if no such path."""
    steps = {v: min(steps_needed(x0[v], cand[v], specs[v]), max_steps(specs[v], horizon)) for v in cand}
    steps = {v: n for v, n in steps.items() if n > 0}
    X0 = pd.DataFrame([x0.to_numpy(float)], columns=x0.index)

    def frame(rows: list[dict[str, int]]) -> pd.DataFrame:
        X = pd.DataFrame(np.repeat(X0.to_numpy(), len(rows), 0), columns=X0.columns)
        for i, st in enumerate(rows):
            for v, n in st.items():
                X.loc[i, v] = float(apply_steps(x0[v], specs[v], n))
        return X

    def scores(rows):
        return scorer.score(frame(rows), pd.concat([X0] * len(rows), ignore_index=True))

    if not steps or scores([steps])[0] < target:
        return None
    # At most max_vars variables, and drop every change the target does not need
    while steps:
        drops = [{u: m for u, m in steps.items() if u != v} for v in steps]
        sc = scores(drops)
        best = int(np.argmax(sc))
        if len(steps) > max_vars or sc[best] >= target:
            if sc[best] < target:
                return None
            steps = drops[best]
        else:
            break
    # Fewest steps per variable (the score is monotone in each varied feature)
    for v in sorted(steps, key=lambda u: -specs[u].difficulty * specs[u].months_per_step):
        lo, hi = 1, steps[v]
        while lo < hi:
            mid = (lo + hi) // 2
            if scores([{**steps, v: mid}])[0] >= target:
                hi = mid
            else:
                lo = mid + 1
        steps[v] = hi
    final = frame([steps])
    months = {v: n * specs[v].months_per_step for v, n in steps.items()}
    return {
        "steps": steps,
        "values": {v: float(final.loc[0, v]) for v in steps},
        "months": max(months.values()),
        "cost": sum(specs[v].difficulty * months[v] / horizon for v in steps),
        "score": int(scores([steps])[0]),
    }


def dice_path(scorer: Scorer, x0: pd.Series, specs: dict[str, Spec], horizon: int, target: int,
              rcfg: dict, seed: int, seeds: list[dict[str, float]] | None = None) -> dict | None:
    """Best DiCE path for one applicant (lowest cost, then fewest variables), or None."""
    import dice_ml

    varied = [v for v, s in specs.items() if can_vary(x0[v], s)]
    if not varied:
        return None
    ranges, precision = {}, {}
    for v in varied:
        end = float(apply_steps(x0[v], specs[v], max_steps(specs[v], horizon)))
        ranges[v] = sorted([float(x0[v]), end])
        precision[v] = _precision(x0[v], specs[v])
    varied = [v for v in varied if ranges[v][1] > ranges[v][0]]
    if not varied:
        return None
    ranges = {v: ranges[v] for v in varied}

    random.seed(seed)
    np.random.seed(seed % 2**32)
    model = _DiceModel(scorer, x0, varied)
    data = dice_ml.Data(features=ranges, outcome_name="score",
                        type_and_precision={v: precision[v] for v in varied})
    dmodel = dice_ml.Model(model=model, backend="sklearn", model_type="regressor")
    exp = _batched_genetic_class()(data, dmodel)
    exp.init_batch, exp.init_max_samples = rcfg["init_batch"], rcfg["init_max_samples"]
    if seeds:
        exp.seeds = np.array([[s.get(v, x0[v]) for v in varied] for s in seeds], float)
    d = rcfg["dice"]
    query = pd.DataFrame([x0[varied].astype(float)])
    res = exp.generate_counterfactuals(
        query, total_CFs=d["total_cfs"], desired_range=[target, scorer.cfg["score"]["max"]],
        initialization="random", feature_weights={v: specs[v].difficulty for v in varied},
        proximity_weight=d["proximity_weight"], sparsity_weight=d["sparsity_weight"],
        diversity_weight=d["diversity_weight"], maxiterations=d["maxiterations"], verbose=False)
    cfs = res.cf_examples_list[0].final_cfs_df
    if cfs is None or len(cfs) == 0:
        return None
    paths = []
    for _, row in cfs.iterrows():
        p = snap_and_trim(scorer, x0, {v: float(row[v]) for v in varied}, specs, horizon, target,
                          rcfg["max_changed_features"])
        if p is not None:
            paths.append(p)
    if not paths:
        return None
    return min(paths, key=lambda p: (round(p["cost"], 9), len(p["steps"]), p["months"]))


# ---------------------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------------------

_WORKER: dict = {}


def _init_worker(link: bool) -> None:
    from src.models.predictor import get_model

    m = get_model()
    m.pipeline.set_params(model__n_jobs=1)  # one core per worker process
    _WORKER["scorer"] = Scorer(m.pipeline, m.features, m.scoring_cfg, link)
    _WORKER["specs"] = load_specs(m.features)


def _corner_seeds(scorer, x0: pd.Series, specs, horizon, target, max_vars) -> list[dict[str, float]]:
    """Corner points (subsets of <= max_vars variables at their limit) that reach the target."""
    X = pd.DataFrame([x0.to_numpy(float)], columns=x0.index)
    seeds = []
    for k in range(1, max_vars + 1):
        for sub in itertools.combinations(specs, k):
            Xc = corner_frame(X, specs, sub, horizon)
            if scorer.score(Xc, X)[0] >= target:
                seeds.append({v: float(Xc.loc[0, v]) for v in sub})
    return seeds


def _run_one(task: tuple) -> dict:
    row_id, x0, horizon, target, rcfg = task
    scorer, specs = _WORKER["scorer"], _WORKER["specs"]
    t0 = time.time()
    seeds = _corner_seeds(scorer, x0, specs, horizon, target, rcfg["max_changed_features"])
    path = dice_path(scorer, x0, specs, horizon, target, rcfg, rcfg["seed"] * 100_003 + int(row_id) * 31 + horizon,
                     seeds) if seeds else None
    out = {"row_id": row_id, "dice_found": path is not None, "seconds": round(time.time() - t0, 3)}
    if path:
        out.update({"n_vars": len(path["steps"]), "months": path["months"], "cost": round(path["cost"], 4),
                    "score_after": path["score"], "variables": " + ".join(path["steps"])})
        for v, val in path["values"].items():
            out[f"new_{v}"] = val
    return out


def rejected_applicants(limit: int | None = None):
    """Test applicants below the approval cutoff: (X, thin flag, score), index = row_id."""
    from src.fairness.data import load_split
    from src.fairness.mitigation import base_cutoff
    from src.models.predictor import get_model

    m = get_model()
    test = load_split("test")
    X = test[m.features].astype(float)
    pred = m.predict(X)
    s0 = pred["score"].to_numpy()
    rej = s0 < base_cutoff(m.scoring_cfg)
    Xr, thin, sr = X[rej], pred["thin_filer"].to_numpy()[rej] == 1, s0[rej]
    if limit:
        Xr, thin, sr = Xr.iloc[:limit], thin[:limit], sr[:limit]
    return Xr, thin, sr


def summarize(df: pd.DataFrame, thin: np.ndarray) -> dict:
    found = df["dice_found"].to_numpy()
    corner = df["corner_feasible"].to_numpy()
    f = df[found]
    return {
        "rejected": int(len(df)),
        "corner_bound": round(float(corner.mean()), 4),
        "corner_bound_thin": round(float(corner[thin].mean()), 4) if thin.any() else None,
        "dice_coverage": round(float(found.mean()), 4),
        "dice_coverage_thin": round(float(found[thin].mean()), 4) if thin.any() else None,
        "dice_coverage_not_thin": round(float(found[~thin].mean()), 4),
        "dice_share_of_corner": round(float(found.sum() / max(corner.sum(), 1)), 4),
        "vars_needed": {str(k): int((f["n_vars"] == k).sum()) for k in (1, 2, 3)},
        "median_months": float(f["months"].median()) if len(f) else None,
        "top_paths": {k: int(v) for k, v in f["variables"].value_counts().head(5).items()},
        "not_feasible_even_all_vars": int((~df["corner_all_vars"]).sum()),
        "seconds_per_applicant_median": float(df.loc[corner, "seconds"].median()) if corner.any() else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N rejected applicants (pilot)")
    ap.add_argument("--horizons", type=int, nargs="*", default=None)
    args = ap.parse_args()

    from src.models.predictor import get_model
    from src.models.scoring import load_config as load_scoring

    rcfg = load_config()["explainer"]["recourse"]
    target = load_scoring()["recourse_target_score"]
    horizons = args.horizons or rcfg["horizons_months"]
    m = get_model()
    scorer = Scorer(m.pipeline, m.features, m.scoring_cfg, rcfg["link_income_debt"])
    specs = load_specs(m.features)
    Xr, thin, s0 = rejected_applicants(args.limit)
    suffix = "_pilot" if args.limit else ""
    out_dir = PROJECT_ROOT / "reports" / rcfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = os.cpu_count() if rcfg["n_jobs"] == -1 else rcfg["n_jobs"]
    print(f"{len(Xr)} rejected applicants, horizons {horizons}, target {target}, {n_jobs} workers")

    summary = {"target_score": target, "max_changed_features": rcfg["max_changed_features"],
               "link_income_debt": rcfg["link_income_debt"], "dice": rcfg["dice"],
               "varied_features": list(specs), "horizons": {}}
    t_all = time.time()
    with ProcessPoolExecutor(n_jobs, initializer=_init_worker, initargs=(rcfg["link_income_debt"],)) as pool:
        for h in horizons:
            t0 = time.time()
            cb = corner_bound(scorer, Xr, specs, h, target, rcfg["max_changed_features"])
            tasks = [(rid, Xr.loc[rid], h, target, rcfg) for rid in Xr.index[cb["corner_feasible"].to_numpy()]]
            found = pd.DataFrame(list(pool.map(_run_one, tasks, chunksize=4))) if tasks else pd.DataFrame()
            df = cb.assign(score_before=s0, thin_filer=thin.astype(int))
            df = df.join(found.set_index("row_id")) if len(found) else df.assign(dice_found=False, seconds=np.nan)
            df["dice_found"] = df["dice_found"].fillna(False).astype(bool)
            df.index.name = "row_id"
            df.to_csv(out_dir / f"paths_{h}m{suffix}.csv")
            res = summarize(df, thin)
            res["wall_seconds"] = round(time.time() - t0, 1)
            summary["horizons"][str(h)] = res
            print(f"[{h:>2}m] corner {res['corner_bound']:.4f}  DiCE {res['dice_coverage']:.4f} "
                  f"(thin {res['dice_coverage_thin']})  {res['wall_seconds']:.0f}s")
    summary["wall_seconds"] = round(time.time() - t_all, 1)
    (out_dir / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                                   encoding="utf-8")
    if not args.limit:
        _log_mlflow(rcfg, summary, out_dir)
    print(f"Saved reports/{rcfg['output_dir']}/summary{suffix}.json")


def _log_mlflow(rcfg: dict, summary: dict, out_dir) -> None:
    import mlflow

    from src.models.train import load_yaml, setup_mlflow

    setup_mlflow(load_yaml("train.yaml"))
    mlflow.set_experiment(rcfg["mlflow_experiment"])
    with mlflow.start_run(run_name="dice_horizons"):
        mlflow.set_tags({"owner": "chaeminkyu", "split": "test"})
        mlflow.log_params({"target_score": summary["target_score"], "max_changed_features": summary["max_changed_features"],
                           "link_income_debt": summary["link_income_debt"], **{f"dice_{k}": v for k, v in rcfg["dice"].items()}})
        for h, r in summary["horizons"].items():
            mlflow.log_metrics({f"dice_coverage_{h}m": r["dice_coverage"], f"corner_bound_{h}m": r["corner_bound"]})
        mlflow.log_artifacts(str(out_dir), artifact_path="recourse")


if __name__ == "__main__":
    main()
