"""Example datasets shipped with the package (precomputed metric tables from the paper)."""

from __future__ import annotations

from importlib import resources

import pandas as pd

_FILES = {
    "rac": "rac_compressive_strength.csv",
    "case_study_1": "case_study_1_huang2023.csv",
    "case_study_2": "case_study_2_cho2024.csv",
}


def _load(key: str) -> pd.DataFrame:
    with resources.files("mlmrpi.data").joinpath(_FILES[key]).open("r", encoding="utf-8") as f:
        return pd.read_csv(f, index_col="Model")


def load_rac() -> pd.DataFrame:
    """Fourteen ML regressors predicting the compressive strength of recycled
    aggregate concrete (1163 samples): MSE, R2, RMSE, MAE on the Train, Test and
    Total sets (main dataset of the MLmrpi paper)."""
    return _load("rac")


def load_case_study_1() -> pd.DataFrame:
    """Twelve ML models predicting the compressive strength of self-compacting
    concrete (R2, RMSE, MAE on Train / Test / Total) - Huang et al. (2023)."""
    return _load("case_study_1")


def load_case_study_2() -> pd.DataFrame:
    """Seven ML models predicting concrete compressive strength from RCA moisture
    content (R2, RMSE, MAE and training time on Train; R2, RMSE, MAE on Test) -
    Cho et al. (2024)."""
    return _load("case_study_2")


def list_datasets():
    return list(_FILES)
