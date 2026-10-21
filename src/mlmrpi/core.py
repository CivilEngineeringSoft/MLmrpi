"""Core computation of the ML Multi-Response Performance Index (MLmrpi).

Five multi-response methods are implemented, exactly following the equations of
the MLmrpi paper (Methodology section):

* ``AW``   - Assignment of Weights
* ``DEAR`` - Data Envelopment Analysis-based Ranking
* ``GRA``  - Grey Relational Analysis (grey relational grade)
* ``FA``   - Factor Analysis (principal-component weighting)
* ``MRNV`` - Multi-Response Normalized Value

Input is a table with one row per ML model and one column per precomputed
performance metric (e.g. ``Train_R2``, ``Train_RMSE``, ``Test_MAE`` ...).
A higher MLmrpi always means a better model.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .objectives import LARGER, SMALLER, resolve_objectives

METHODS = ("AW", "DEAR", "GRA", "FA", "MRNV")
_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def prepare_input(
    data: Union[pd.DataFrame, str],
    model_col: Optional[str] = None,
    metrics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return a clean numeric DataFrame indexed by model name (Step 1 of the paper).

    ``data`` may be a DataFrame or a path to a CSV / Excel file. If ``model_col``
    is not given, the first non-numeric column (or the existing index) is used
    as the model identifier.
    """
    if isinstance(data, str):
        if data.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(data)
        else:
            df = pd.read_csv(data)
    else:
        df = data.copy()

    if model_col is not None:
        df = df.set_index(model_col)
    elif not pd.api.types.is_numeric_dtype(df.iloc[:, 0]) and df.index.dtype.kind in "iu":
        df = df.set_index(df.columns[0])
    df.index = df.index.astype(str)
    df.index.name = df.index.name or "Model"

    if metrics is not None:
        df = df[list(metrics)]
    else:
        df = df.select_dtypes(include=[np.number])

    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Missing or non-numeric values in metric columns: {bad}")
    if df.shape[0] < 2:
        raise ValueError("At least two models are needed to compute a ranking.")
    if df.shape[1] < 1:
        raise ValueError("No numeric metric columns found.")
    if df.index.duplicated().any():
        raise ValueError(f"Duplicated model names: {df.index[df.index.duplicated()].tolist()}")
    return df


def select_phase(df: pd.DataFrame, phase: Union[str, Iterable[str]], sep: str = "_",
                 strip: bool = False) -> pd.DataFrame:
    """Keep only the columns of one (or several) phases, e.g. ``"Train"`` or ``["Train", "Test"]``.

    Columns are expected to be named ``<phase><sep><metric>`` (``Train_RMSE``).
    With ``strip=True`` the phase prefix is removed (only for a single phase).
    """
    phases = [phase] if isinstance(phase, str) else list(phase)
    keep = [c for c in df.columns
            if any(str(c).lower().startswith(p.lower() + sep.lower()) for p in phases)]
    if not keep:
        raise ValueError(f"No column starts with any of {phases} + '{sep}'.")
    out = df[keep]
    if strip and len(phases) == 1:
        out = out.rename(columns=lambda c: str(c)[len(phases[0]) + len(sep):])
    return out


# --------------------------------------------------------------------------- #
# Step 2 - normalization (GRA, FA, MRNV)
# --------------------------------------------------------------------------- #
def normalize(df: pd.DataFrame, objectives: Mapping[str, str]) -> pd.DataFrame:
    """Min-max normalization of every metric to [0, 1] where 1 = best (paper Eqs. GRA).

    larger-the-better : eta = (R - min) / (max - min)
    smaller-the-better: eta = (max - R) / (max - min)
    A metric that is identical for all models (max == min) carries no ranking
    information and is set to 1 for every model.
    """
    eta = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        x = df[col].to_numpy(dtype=float)
        lo, hi = x.min(), x.max()
        if hi - lo < _EPS:
            eta[col] = 1.0
        elif objectives[col] == LARGER:
            eta[col] = (x - lo) / (hi - lo)
        else:
            eta[col] = (hi - x) / (hi - lo)
    return eta


# --------------------------------------------------------------------------- #
# Step 3 - weights (AW, DEAR)
# --------------------------------------------------------------------------- #
def response_weights(df: pd.DataFrame, objectives: Mapping[str, str]) -> pd.DataFrame:
    """Weights W_ij of the AW and DEAR methods (paper Eqs. AW_Larger / AW_Smaller).

    larger-the-better : W_ij = R_ij / sum_i R_ij
    smaller-the-better: W_ij = (1 / R_ij) / sum_i (1 / R_ij)
    """
    W = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        x = df[col].to_numpy(dtype=float)
        if objectives[col] == LARGER:
            s = x.sum()
            W[col] = x / s if abs(s) > _EPS else 1.0 / len(x)
        else:
            if np.any(x <= 0):
                raise ValueError(
                    f"Smaller-the-better metric '{col}' contains values <= 0; the inverse "
                    "weighting of AW/DEAR (1/R) is undefined. Use GRA, FA or MRNV, or shift the metric."
                )
            inv = 1.0 / x
            W[col] = inv / inv.sum()
    return W


# --------------------------------------------------------------------------- #
# Step 4 - MLmrpi for each method
# --------------------------------------------------------------------------- #
def mlmrpi_aw(df: pd.DataFrame, objectives: Mapping[str, str]) -> pd.Series:
    """Method I - Assignment of Weights: MLmrpi_i = sum_j W_ij * R_ij."""
    W = response_weights(df, objectives)
    return (W * df).sum(axis=1).rename("AW")


def mlmrpi_dear(df: pd.DataFrame, objectives: Mapping[str, str]) -> pd.Series:
    """Method II - DEAR: sum of weighted larger-the-better responses divided by the
    sum of weighted smaller-the-better responses (paper Eq. Eq_Mrpi_DEAR).

    If the table has no smaller-the-better metric the denominator is taken as 1;
    if it has no larger-the-better metric the numerator is taken as 1.
    """
    W = response_weights(df, objectives)
    WR = W * df
    ltb = [c for c in df.columns if objectives[c] == LARGER]
    stb = [c for c in df.columns if objectives[c] == SMALLER]
    num = WR[ltb].sum(axis=1) if ltb else pd.Series(1.0, index=df.index)
    den = WR[stb].sum(axis=1) if stb else pd.Series(1.0, index=df.index)
    return (num / den).rename("DEAR")


def grey_coefficients(eta: pd.DataFrame, lam: float = 0.5) -> pd.DataFrame:
    """Grey relational coefficients GC_ij = (D_min + lam*D_max) / (D_ij + lam*D_max)
    with D_ij = eta_0j - eta_ij and eta_0j = max_i(eta_ij)."""
    if not 0 <= lam <= 1:
        raise ValueError("The distinguishing coefficient lambda must lie in [0, 1].")
    delta = eta.max(axis=0) - eta
    dmin, dmax = float(delta.values.min()), float(delta.values.max())
    if dmax < _EPS:
        return pd.DataFrame(1.0, index=eta.index, columns=eta.columns)
    return (dmin + lam * dmax) / (delta + lam * dmax)


def _weighted_mean(mat: pd.DataFrame, weights: Optional[Mapping[str, float]]) -> pd.Series:
    if weights is None:
        return mat.mean(axis=1)
    w = np.array([float(weights.get(c, 1.0)) for c in mat.columns])
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Metric weights must be >= 0 and not all zero.")
    return (mat * w).sum(axis=1) / w.sum()


def mlmrpi_gra(df: pd.DataFrame, objectives: Mapping[str, str], lam: float = 0.5,
               weights: Optional[Mapping[str, float]] = None) -> pd.Series:
    """Method III - Grey Relational Analysis: MLmrpi_i = (1/m) sum_j GC_ij (grey grade)."""
    gc = grey_coefficients(normalize(df, objectives), lam)
    return _weighted_mean(gc, weights).rename("GRA")


def fa_weights(eta: pd.DataFrame, eigen_threshold: float = 1.0) -> pd.Series:
    """Metric weights PC_j derived from principal-component factor analysis of eta.

    1. Correlation matrix of the normalized metrics (constant columns are given weight 0).
    2. Eigen-decomposition; retain the components with eigenvalue > ``eigen_threshold``
       (Kaiser criterion, paper Step 2). At least one component is always kept.
    3. Loadings L_jk = v_jk * sqrt(lambda_k). The weight of metric j is the
       variance-weighted mean of its absolute loadings on the retained components,
       PC_j = sum_k (lambda_k / sum lambda) |L_jk|, rescaled so that sum_j PC_j = 1.
    """
    var = eta.var(axis=0, ddof=1)
    active = var[var > _EPS].index.tolist()
    weights = pd.Series(0.0, index=eta.columns)
    if len(active) == 0:
        return weights + 1.0 / len(weights)
    if len(active) == 1:
        weights[active[0]] = 1.0
        return weights
    corr = np.corrcoef(eta[active].to_numpy(dtype=float), rowvar=False)
    corr = np.nan_to_num(corr)
    eigval, eigvec = np.linalg.eigh(corr)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    keep = eigval > eigen_threshold
    if not keep.any():
        keep[0] = True
    lam = eigval[keep]
    loadings = np.abs(eigvec[:, keep] * np.sqrt(np.clip(lam, 0, None)))
    pc = loadings @ (lam / lam.sum())
    pc = pc / pc.sum()
    weights[active] = pc
    return weights


def mlmrpi_fa(df: pd.DataFrame, objectives: Mapping[str, str],
              eigen_threshold: float = 1.0) -> pd.Series:
    """Method IV - Factor Analysis: MLmrpi_i = sum_j eta_ij * PC_j."""
    eta = normalize(df, objectives)
    pc = fa_weights(eta, eigen_threshold)
    return (eta * pc).sum(axis=1).rename("FA")


def mlmrpi_mrnv(df: pd.DataFrame, objectives: Mapping[str, str],
                weights: Optional[Mapping[str, float]] = None) -> pd.Series:
    """Method V - Multi-Response Normalized Value: MLmrpi_i = (1/m) sum_j eta_ij."""
    return _weighted_mean(normalize(df, objectives), weights).rename("MRNV")


# --------------------------------------------------------------------------- #
# Step 5 - ranking
# --------------------------------------------------------------------------- #
def rank_scores(scores: Union[pd.Series, pd.DataFrame], ascending: bool = False,
                tie_method: str = "min") -> Union[pd.Series, pd.DataFrame]:
    """Rank models (1 = best). Higher MLmrpi = better, so ranks are descending by default.

    Ties share the best (smallest) rank (``tie_method='min'``), as in the paper tables.
    Scores are rounded to 12 significant digits first so floating-point noise does
    not break genuine ties.
    """
    rounded = scores.apply(lambda s: s.round(12)) if isinstance(scores, pd.DataFrame) else scores.round(12)
    return rounded.rank(ascending=ascending, method=tie_method).astype(int)


def compute_mlmrpi(
    data: Union[pd.DataFrame, str],
    method: str = "GRA",
    objectives: Optional[Mapping[str, str]] = None,
    lam: float = 0.5,
    weights: Optional[Mapping[str, float]] = None,
    eigen_threshold: float = 1.0,
    model_col: Optional[str] = None,
) -> pd.Series:
    """Compute the MLmrpi of every model with a single method.

    Returns a Series indexed by model (higher = better).
    """
    df = prepare_input(data, model_col=model_col)
    obj = resolve_objectives(df.columns, objectives)
    m = method.upper()
    if m == "AW":
        s = mlmrpi_aw(df, obj)
    elif m == "DEAR":
        s = mlmrpi_dear(df, obj)
    elif m == "GRA":
        s = mlmrpi_gra(df, obj, lam=lam, weights=weights)
    elif m == "FA":
        s = mlmrpi_fa(df, obj, eigen_threshold=eigen_threshold)
    elif m == "MRNV":
        s = mlmrpi_mrnv(df, obj, weights=weights)
    else:
        raise ValueError(f"Unknown method {method!r}. Choose among {METHODS}.")
    if weights is not None and m in ("AW", "DEAR", "FA"):
        import warnings
        warnings.warn(f"'weights' are only used by GRA and MRNV; ignored for {m}.")
    return s.rename(m)


# --------------------------------------------------------------------------- #
# Ranking from individual precomputed metrics (rank-sum, Hamed et al. 2025)
# --------------------------------------------------------------------------- #
def rank_by_metrics(
    data: Union[pd.DataFrame, str],
    objectives: Optional[Mapping[str, str]] = None,
    tie_method: str = "min",
    model_col: Optional[str] = None,
) -> pd.DataFrame:
    """Rank models on every metric separately, then sum the ranks.

    Returns a DataFrame (models x metrics) of individual ranks plus the columns
    ``Total Rank`` and ``Overall Rank`` (lowest total = best), as in the paper's
    rank-aggregation table.
    """
    df = prepare_input(data, model_col=model_col)
    obj = resolve_objectives(df.columns, objectives)
    ranks = pd.DataFrame(index=df.index)
    for col in df.columns:
        ranks[col] = df[col].round(12).rank(ascending=(obj[col] == SMALLER),
                                            method=tie_method).astype(int)
    ranks["Total Rank"] = ranks.sum(axis=1)
    ranks["Overall Rank"] = ranks["Total Rank"].rank(method=tie_method).astype(int)
    return ranks
