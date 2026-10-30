"""Inputs for the fairness checks: one split of the simulated data, the shipped model's decisions
on it, and the protected groups (simulated gender, age band).

No training happens here. The split is rebuilt with the same seeded rule as training
(src.data.preprocessor.make_split) and checked against data/processed/split.csv when it exists.
"""

from __future__ import annotations

import pandas as pd
import yaml

from src.config import PROJECT_ROOT
from src.data.features import is_thin_filer_proxy
from src.data.preprocessor import make_split
from src.models.predictor import get_model
from src.models.scoring import score_frame


def _load_yaml(name: str) -> dict:
    with open(PROJECT_ROOT / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def target_column() -> str:
    """Default flag column (1 = default), from configs/train.yaml."""
    return _load_yaml("train.yaml")["target"]


def load_split(split: str = "test") -> pd.DataFrame:
    """Rows of data/processed/gmsc_sim.csv in one split (train, validation or test).

    The index is the row_id used in split.csv.
    """
    splits = load_splits()
    if split not in splits:
        raise ValueError(f"unknown split {split!r}, expected one of {sorted(splits)}")
    return splits[split]


def load_splits() -> dict[str, pd.DataFrame]:
    """All three splits as {"train": ..., "validation": ..., "test": ...}, in file row order."""
    cfg = _load_yaml("train.yaml")
    data_path = PROJECT_ROOT / cfg["data"]
    if not data_path.exists():
        raise FileNotFoundError(
            f"{cfg['data']} not found: run `python -m src.data` and `python -m src.data.simulator`"
        )
    df = pd.read_csv(data_path)
    labels = make_split(df, cfg)
    saved = PROJECT_ROOT / cfg["outputs"]["split"]
    if saved.exists():
        on_disk = pd.read_csv(saved).set_index("row_id")["split"]
        if not on_disk.reindex(labels.index).equals(labels):
            raise ValueError(f"{cfg['outputs']['split']} differs from the seeded split rule")
    return {name: df[labels == name] for name in ("train", "validation", "test")}


def score(df: pd.DataFrame, estimator=None) -> pd.DataFrame:
    """Decisions: pd, score, grade, approved (1/0), thin_filer (1/0).

    estimator: a fitted model with predict_proba; None means the shipped model.
    """
    model = get_model()
    X = df[model.features].astype(float)
    if estimator is None:
        out = model.predict(X)
    else:
        out = score_frame(estimator.predict_proba(X)[:, 1], model.scoring_cfg)
        out["thin_filer"] = is_thin_filer_proxy(X, model.thin_rule).to_numpy()
    out.index = df.index
    return out


def protected_groups(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Group label per row for each protected attribute in configs/config.yaml.

    gender comes from the simulated gender_female column; age bands from configs/simulator.yaml.
    """
    bands = _load_yaml("simulator.yaml")["age_bands"]
    return {
        "gender": df["gender_female"].map({0: "male", 1: "female"}),
        "age_band": pd.cut(df["age"], bands["edges"], labels=bands["labels"]).astype(str),
    }
