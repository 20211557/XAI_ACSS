"""Register the final model in the MLflow Model Registry.

Run from the project root after training (python -m src.models.train --stage all):
  python -m src.models.registry

The tracking location follows the training code (MLFLOW_TRACKING_URI, else the local ./mlruns).
The latest run tagged stage=final in the final experiment is registered under
mlflow.registry.model_name (configs/train.yaml). MLflow 3 deprecates stages, so an alias marks
the version instead: `production` when the run meets every charter target (configs/config.yaml
performance_targets, monotonic.max_auc_loss), `staging` otherwise. A staging version leaves the
current production version in place. Registering the same run again reuses its version.

Load the registered model (pyfunc returns probabilities, column 1 = PD):
  mlflow.pyfunc.load_model("models:/xai-credit-scoring@production")
"""
from __future__ import annotations

import json
import operator

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run

from src.config import load_config
from src.models import evaluate as ev
from src.models.train import load_yaml, setup_mlflow

OPS = {">=": operator.ge, "<": operator.lt, "<=": operator.le}
TRAIN_HINT = "run `python -m src.models.train --stage all` first"


def charter_checks(metrics: dict[str, float], targets: dict, max_mono_loss: float) -> dict:
    """Charter targets checked on the final run's metrics.

    pass is False when a metric is missing, except the monotonic loss: a model without
    monotonic constraints logs none, so that check is not applicable (None)."""
    rules = [
        (f"test_{ev.KEY_AUC}", ">=", targets["auc_roc_min"]),
        (f"test_{ev.KEY_KS}", ">=", targets["ks_statistic_min"]),
        ("test_thin_auc_lift", ">=", targets["thin_filer_auc_uplift_min"]),
        ("test_psi_trainval_vs_test", "<", targets["psi_max"]),
        (ev.KEY_MONO_LOSS, "<=", max_mono_loss),
    ]
    out = {}
    for key, op, bound in rules:
        value = metrics.get(key)
        if value is None:
            ok = None if key == ev.KEY_MONO_LOSS else False
        else:
            ok = bool(OPS[op](value, bound))
        out[key] = {"value": value, "target": f"{op} {bound}", "pass": ok}
    return out


def find_final_run(client: MlflowClient, experiment: str) -> Run:
    """Latest run tagged stage=final in the given experiment."""
    exp = client.get_experiment_by_name(experiment)
    runs = [] if exp is None else client.search_runs(
        [exp.experiment_id], filter_string="tags.stage = 'final'",
        order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise LookupError(f"no run tagged stage=final in experiment '{experiment}' "
                          f"at {mlflow.get_tracking_uri()}: {TRAIN_HINT}")
    return runs[0]


def logged_model_uri(run: Run) -> str:
    """models:/<model_id> of the model the run logged (MLflow 3 logged model)."""
    outputs = run.outputs.model_outputs if run.outputs else []
    if not outputs:
        raise LookupError(f"final run {run.info.run_id} has no logged model: {TRAIN_HINT}")
    return f"models:/{outputs[0].model_id}"


def register_final(cfg: dict, shared: dict, client: MlflowClient | None = None) -> dict:
    """Register the latest final run's model and point the alias at its version.

    cfg: configs/train.yaml. shared: configs/config.yaml (charter targets, disclaimer).
    Uses the current MLflow tracking URI."""
    client = client or MlflowClient()
    reg = cfg["mlflow"]["registry"]
    name = reg["model_name"]
    run = find_final_run(client, cfg["mlflow"]["experiments"]["final"])
    run_id, metrics, params = run.info.run_id, run.data.metrics, run.data.params

    targets = shared["performance_targets"]
    charter = charter_checks(metrics, targets, cfg["monotonic"]["max_auc_loss"])
    failed = [k for k, c in charter.items() if c["pass"] is False]
    alias = reg["fallback_alias"] if failed else reg["alias"]

    if not client.search_registered_models(filter_string=f"name = '{name}'"):
        client.create_registered_model(name, description=(
            f"{shared['project']['disclaimer']}\n"
            "Final credit scoring model. pyfunc predict returns probabilities (column 1 = PD)."))
    existing = client.search_model_versions(f"name = '{name}' and run_id = '{run_id}'")
    if existing:
        version = existing[0].version
    else:
        version = mlflow.register_model(logged_model_uri(run), name).version

    scores = (f"test_{ev.KEY_AUC}", f"test_{ev.KEY_KS}")
    tags = {
        "run_name": run.info.run_name,
        "model_version": params.get("model_version"),
        "model_file": params.get("model_file"),
        "data_md5": params.get("data_md5"),
        **{k: f"{metrics[k]:.4f}" for k in scores if k in metrics},
        "charter_pass": str(not failed).lower(),
        "charter_failed": ",".join(failed) or "none",
    }
    for key, value in tags.items():
        if value:
            client.set_model_version_tag(name, version, key, value)
    client.set_registered_model_alias(name, alias, version)
    return {"model_name": name, "version": int(version), "alias": alias,
            "uri": f"models:/{name}@{alias}", "new_version": not existing,
            "run_id": run_id, "run_name": run.info.run_name, "charter": charter}


def main() -> None:
    cfg = load_yaml("train.yaml")
    setup_mlflow(cfg)
    try:
        result = register_final(cfg, load_config())
    except LookupError as e:
        raise SystemExit(f"error: {e}") from None
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
