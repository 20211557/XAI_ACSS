"""The committed MLflow snapshot (mlflow_seed/) that seeds an empty server on first start."""
import json
import sqlite3

import pytest

from tests.helpers import ROOT, load_yaml

SEED = ROOT / "mlflow_seed"
CFG = load_yaml("train.yaml")
MLFLOW = CFG["mlflow"]
PREFIX = "mlflow-artifacts:/"   # server-relative URIs, resolved against --artifacts-destination
ACTIVE = "lifecycle_stage = 'active'"


@pytest.fixture(scope="module")
def db():
    path = SEED / "mlflow.db"
    assert path.exists(), f"{path} missing: it is committed so a fresh clone shows the MLflow runs"
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    yield con
    con.close()


def artifact_dir(uri: str):
    """mlflow-artifacts:/<path> -> mlflow_seed/artifacts/<path>."""
    assert uri.startswith(PREFIX), f"not portable (absolute path?): {uri}"
    return SEED / "artifacts" / uri.removeprefix(PREFIX)


def test_seed_has_mission_experiments(db):
    names = {n for (n,) in db.execute(f"select name from experiments where {ACTIVE}")}
    wanted = {MLFLOW["experiments"][k] for k in ("lr", "lgbm", "xgb", "final")}
    assert wanted <= names    # mission checklist: 3 or more experiments in the MLflow UI
    runs = db.execute(f"select count(*) from runs where {ACTIVE}").fetchone()[0]
    assert runs >= 27         # 27 comparison runs plus monotonic and final runs


def test_seed_registry_points_at_the_shipped_final_run(db):
    reg = MLFLOW["registry"]
    rows = db.execute(
        "select v.run_id, v.storage_location from registered_model_aliases a "
        "join model_versions v on v.name = a.name and v.version = a.version "
        "where a.name = ? and a.alias = ?", (reg["model_name"], reg["alias"])).fetchall()
    assert len(rows) == 1, "production alias missing in the seed"
    run_id, storage = rows[0]
    summary = json.loads((ROOT / CFG["outputs"]["final_summary"]).read_text())
    assert run_id == summary["mlflow_run_id"], "seed and final_summary.json disagree: refresh it"
    assert (artifact_dir(storage) / "MLmodel").exists()


def test_seed_artifacts_are_portable_and_present(db):
    uris = [u for (u,) in db.execute("select artifact_location from experiments")]
    uris += [u for (u,) in db.execute(f"select artifact_uri from runs where {ACTIVE}")]
    for uri in uris:
        artifact_dir(uri)     # fails on host-specific paths
    final = db.execute("select r.artifact_uri from runs r join tags t on t.run_uuid = r.run_uuid "
                       "where t.key = 'stage' and t.value = 'final'").fetchall()
    assert final and all(artifact_dir(u).is_dir() for (u,) in final)
