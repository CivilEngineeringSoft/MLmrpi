"""Robustness / consistency checks of MLmrpi rankings (paper: MLmrpi robustness).

* correlation between MLmrpi scores and individual metrics (Pearson, Spearman)
* performance grouping: Top / Middle / Bottom tiers and their mean metrics
* Shapiro-Wilk normality of rank differences + Wilcoxon signed-rank test
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .core import rank_scores
from .objectives import SMALLER


def correlation_with_metrics(scores: pd.DataFrame, metrics: pd.DataFrame,
                             method: str = "spearman") -> pd.DataFrame:
    """Correlation (``'pearson'`` or ``'spearman'``) between each MLmrpi method
    (columns of ``scores``) and each metric (columns of ``metrics``).

    Expected sign: positive for larger-the-better metrics (R2), negative for
    error metrics (RMSE, MAE).
    """
    func = {"pearson": stats.pearsonr, "spearman": stats.spearmanr}[method]
    metrics = metrics.loc[scores.index]
    out = pd.DataFrame(index=scores.columns, columns=metrics.columns, dtype=float)
    for s in scores.columns:
        for m in metrics.columns:
            if scores[s].nunique() < 2 or metrics[m].nunique() < 2:
                out.loc[s, m] = np.nan
            else:
                out.loc[s, m] = func(scores[s], metrics[m])[0]
    return out


def performance_tiers(scores: pd.DataFrame, metrics: pd.DataFrame,
                      n_tiers: int = 3, labels=("Top", "Middle", "Bottom")) -> pd.DataFrame:
    """Inverse check: sort models by MLmrpi, split them into ``n_tiers`` groups and
    return the mean of every metric per tier and method.

    Returns a DataFrame with a (method, tier) MultiIndex.
    """
    if len(labels) != n_tiers:
        labels = [f"Tier {i + 1}" for i in range(n_tiers)]
    metrics = metrics.loc[scores.index]
    rows = []
    for meth in scores.columns:
        order = scores[meth].sort_values(ascending=False).index
        for lab, grp in zip(labels, np.array_split(np.asarray(order), n_tiers)):
            row = metrics.loc[list(grp)].mean()
            row.name = (meth, lab)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.index = pd.MultiIndex.from_tuples(out.index, names=["Method", "Tier"])
    return out


def rank_difference_tests(method_ranks: pd.DataFrame, metric_ranks: pd.DataFrame,
                          alpha: float = 0.05) -> pd.DataFrame:
    """Compare every MLmrpi ranking with every single-metric ranking.

    For each (method, metric) pair: Shapiro-Wilk p-value of the rank differences,
    then the Wilcoxon signed-rank p-value (H0: no systematic difference between
    the two rankings). ``p > alpha`` means the rankings are not significantly
    different.
    """
    metric_ranks = metric_ranks.loc[method_ranks.index]
    records = []
    for meth in method_ranks.columns:
        for met in metric_ranks.columns:
            d = method_ranks[meth].to_numpy(float) - metric_ranks[met].to_numpy(float)
            if np.allclose(d, 0):
                sw_p, wil_p = np.nan, 1.0
            else:
                sw_p = stats.shapiro(d)[1] if len(d) >= 3 and np.ptp(d) > 0 else np.nan
                try:
                    wil_p = stats.wilcoxon(d, zero_method="zsplit")[1]
                except ValueError:
                    wil_p = 1.0
            records.append({"Method": meth, "Metric": met, "Shapiro_p": sw_p,
                            "Wilcoxon_p": wil_p, "Significant": bool(wil_p <= alpha)})
    return pd.DataFrame(records)


def metric_rank_table(metrics: pd.DataFrame, objectives: Mapping[str, str]) -> pd.DataFrame:
    """Rank every metric separately (1 = best)."""
    out = pd.DataFrame(index=metrics.index)
    for c in metrics.columns:
        sign = 1.0 if objectives[c] == SMALLER else -1.0
        out[c] = rank_scores(-sign * metrics[c])
    return out


def stability_to_perturbation(df: pd.DataFrame, compute, n_iter: int = 500,
                              noise: float = 0.02, random_state: Optional[int] = 0) -> pd.DataFrame:
    """Monte-Carlo sensitivity: perturb every metric by a multiplicative Gaussian
    noise of relative s.d. ``noise`` and record how often each model keeps its rank.

    ``compute`` is a callable ``DataFrame -> Series of scores``.
    Returns, per model, the base rank, mean / s.d. of the perturbed ranks and the
    share of iterations with an unchanged rank.
    """
    rng = np.random.default_rng(random_state)
    base = rank_scores(compute(df))
    ranks = []
    for _ in range(n_iter):
        pert = df * (1 + rng.normal(0, noise, size=df.shape))
        ranks.append(rank_scores(compute(pert)))
    R = pd.concat(ranks, axis=1)
    return pd.DataFrame({
        "Base rank": base,
        "Mean rank": R.mean(axis=1),
        "Rank s.d.": R.std(axis=1),
        "Same rank (%)": (R.eq(base, axis=0).mean(axis=1) * 100).round(1),
    }).sort_values("Base rank")
