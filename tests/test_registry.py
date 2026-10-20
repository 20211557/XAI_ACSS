"""Model Registry: the final run gets a registered version and an alias (no data, no server)."""
import copy
import os
import uuid

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression

from src.data.features import feature_sets
from src.models.registry import charter_checks, register_final
from src.models.train import log_final_model
from tests.helpers import load_yaml

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

CFG = load_yaml("train.yaml")
SHARED = load_yaml("config.yaml")
TARGETS = SHARED["performance_targets"]
MAX_LOSS = CFG["monotonic"]["max_auc_loss"]
REG = CFG["mlflow"]["registry"]
# Metrics of a final run that meets every charter target
GOOD = {"test_auc": 0.95, "test_ks": 0.76, "test_thin_auc_lift": 0.07,
        "test_psi_trainval_vs_test": 0.001, "mono_auc_loss": 0.002}


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """Temporary SQLite tracking store (never the MLflow server); restores the previous URI."""
    root = tmp_path_factory.mktemp("mlflow")
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    yield root
    mlflow.set_tracking_uri(previous)


@pytest.fixture
def cfg():
    """train.yaml with its own experiment and model name (tests share one store)."""
    c = copy.deepcopy(CFG)
    uid = uuid.uuid4().hex[:8]
    c["mlflow"]["experiments"]["final"] = f"final-{uid}"
    c["mlflow"]["registry"]["model_name"] = f"model-{uid}"
    return c


@pytest.fixture(scope="module")
def model():
    feats = feature_sets()["all"]
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, len(feats))), columns=feats)
    y = (X.iloc[:, 0] + rng.normal(size=len(X)) > 0).astype(int)
    return LogisticRegression().fit(X, y), X


def log_final_run(store, cfg, model, metrics, stage="final") -> str:
    """A run shaped like train.py's final run: stage tag, params, test metrics, pyfunc model."""
    est, X = model
    name = cfg["mlflow"]["experiments"]["final"]
    exp = mlflow.get_experiment_by_name(name)
    exp_id = exp.experiment_id if exp else mlflow.create_experiment(
        name, artifact_location=(store / name).as_uri())
    with mlflow.start_run(experiment_id=exp_id, run_name=f"lr__both__none__{stage}") as run:
        mlflow.set_tag("stage", stage)
        mlflow.log_params({"model_version": CFG["model_version"], "data_md5": "0123abcd",
                           "model_file": "models/logistic_regression_v1.0.joblib"})
        mlflow.log_metrics(metrics)
        log_final_model(est, X.iloc[:5])
    return run.info.run_id


def by_alias(cfg, alias: str):
    return MlflowClient().get_model_version_by_alias(cfg["mlflow"]["registry"]["model_name"], alias)


def test_charter_checks():
    assert all(c["pass"] for c in charter_checks(GOOD, TARGETS, MAX_LOSS).values())
    low = charter_checks({**GOOD, "test_auc": TARGETS["auc_roc_min"] - 0.01}, TARGETS, MAX_LOSS)
    assert low["test_auc"]["pass"] is False
    missing_auc = {k: v for k, v in GOOD.items() if k != "test_auc"}
    assert charter_checks(missing_auc, TARGETS, MAX_LOSS)["test_auc"]["pass"] is False
    at_max = {**GOOD, "test_psi_trainval_vs_test": TARGETS["psi_max"]}   # PSI must stay below it
    assert charter_checks(at_max, TARGETS, MAX_LOSS)["test_psi_trainval_vs_test"]["pass"] is False
    no_constraints = {k: v for k, v in GOOD.items() if k != "mono_auc_loss"}
    assert charter_checks(no_constraints, TARGETS, MAX_LOSS)["mono_auc_loss"]["pass"] is None


def test_register_production_and_rerun_reuses_version(store, cfg, model):
    run_id = log_final_run(store, cfg, model, GOOD)
    first = register_final(cfg, SHARED)
    name = cfg["mlflow"]["registry"]["model_name"]
    assert (first["version"], first["alias"], first["run_id"]) == (1, REG["alias"], run_id)
    assert first["new_version"] and first["uri"] == f"models:/{name}@{REG['alias']}"

    again = register_final(cfg, SHARED)
    assert (again["version"], again["new_version"]) == (1, False)
    assert len(MlflowClient().search_model_versions(f"name = '{name}'")) == 1

    mv = by_alias(cfg, REG["alias"])
    assert (mv.version, mv.run_id) == (1, run_id)
    assert mv.tags["model_version"] == CFG["model_version"] and mv.tags["test_auc"] == "0.9500"
    assert (mv.tags["charter_pass"], mv.tags["charter_failed"]) == ("true", "none")
    assert SHARED["project"]["disclaimer"] in MlflowClient().get_registered_model(name).description


def test_newer_final_run_moves_alias(store, cfg, model):
    log_final_run(store, cfg, model, GOOD)
    register_final(cfg, SHARED)
    newer = log_final_run(store, cfg, model, GOOD)
    result = register_final(cfg, SHARED)
    assert (result["version"], result["run_id"]) == (2, newer)
    assert by_alias(cfg, REG["alias"]).version == 2


def test_failing_run_goes_to_staging_and_keeps_production(store, cfg, model):
    log_final_run(store, cfg, model, GOOD)
    register_final(cfg, SHARED)
    log_final_run(store, cfg, model, {**GOOD, "test_auc": TARGETS["auc_roc_min"] - 0.01})
    result = register_final(cfg, SHARED)
    assert (result["version"], result["alias"]) == (2, REG["fallback_alias"])
    assert by_alias(cfg, REG["alias"]).version == 1
    staging = by_alias(cfg, REG["fallback_alias"])
    assert staging.version == 2
    assert (staging.tags["charter_pass"], staging.tags["charter_failed"]) == ("false", "test_auc")


def test_only_final_runs_are_registered(store, cfg, model):
    with pytest.raises(LookupError, match="stage=final"):   # no final experiment yet
        register_final(cfg, SHARED)
    log_final_run(store, cfg, model, GOOD, stage="compare")
    with pytest.raises(LookupError, match="stage=final"):   # no run tagged stage=final
        register_final(cfg, SHARED)


def test_alias_uri_serves_probabilities(store, cfg, model):
    est, X = model
    log_final_run(store, cfg, model, GOOD)
    served = mlflow.pyfunc.load_model(register_final(cfg, SHARED)["uri"])
    out = np.asarray(served.predict(X))
    assert out.shape == (len(X), 2)
    np.testing.assert_allclose(out[:, 1], est.predict_proba(X)[:, 1], atol=1e-6)
