"""Agreement statistics between rankings (paper: Methodology - Ranking Agreement).

* Cohen's kappa   - pairwise categorical agreement between two rankings
* Fleiss' kappa   - overall categorical agreement among several rankings
* Kendall's W     - ordinal concordance among several rankings (tie-corrected)

Each ranking is a vector of ranks of the same models (1 = best). The rank values
are treated as category labels for the kappa statistics, as in the paper.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def cohen_kappa(r1, r2, weights: Optional[str] = None) -> float:
    """Cohen's kappa between two rankings.

    Parameters
    ----------
    r1, r2 : array-like
        Ranks of the same models.
    weights : None, "linear" or "quadratic"
        ``None`` (default, used in the paper) gives the unweighted kappa, where
        only identical ranks count as agreement. Weighted variants give partial
        credit to near-identical ranks.
    """
    a, b = np.asarray(r1), np.asarray(r2)
    if a.shape != b.shape:
        raise ValueError("Both rankings must have the same length.")
    cats = np.unique(np.concatenate([a, b]))
    k = len(cats)
    if k == 1:
        return 1.0
    idx = {c: i for i, c in enumerate(cats)}
    conf = np.zeros((k, k))
    for x, y in zip(a, b):
        conf[idx[x], idx[y]] += 1
    n = conf.sum()
    expected = np.outer(conf.sum(1), conf.sum(0)) / n
    if weights is None:
        w = 1.0 - np.eye(k)
    else:
        d = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
        if weights == "linear":
            w = d
        elif weights == "quadratic":
            w = d ** 2
        else:
            raise ValueError("weights must be None, 'linear' or 'quadratic'.")
    denom = (w * expected).sum()
    if denom == 0:
        return 1.0
    return float(1.0 - (w * conf).sum() / denom)


def cohen_kappa_matrix(rankings: pd.DataFrame, weights: Optional[str] = None) -> pd.DataFrame:
    """Pairwise Cohen's kappa between all columns of a (models x rankings) table."""
    cols = list(rankings.columns)
    mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for c1, c2 in combinations(cols, 2):
        k = cohen_kappa(rankings[c1], rankings[c2], weights=weights)
        mat.loc[c1, c2] = mat.loc[c2, c1] = k
    return mat


def fleiss_kappa(rankings: pd.DataFrame) -> float:
    """Fleiss' kappa: subjects = models (rows), raters = rankings (columns),
    categories = rank values."""
    data = np.asarray(rankings)
    n_subj, n_raters = data.shape
    if n_raters < 2:
        raise ValueError("Fleiss' kappa needs at least two rankings.")
    cats = np.unique(data)
    counts = np.stack([(data == c).sum(axis=1) for c in cats], axis=1)  # subj x cat
    p_j = counts.sum(axis=0) / (n_subj * n_raters)
    P_i = ((counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    if np.isclose(P_e, 1.0):
        return 1.0
    return float((P_bar - P_e) / (1 - P_e))


def kendall_w(rankings: pd.DataFrame, return_p: bool = False, tie_correction: bool = True):
    """Kendall's coefficient of concordance W.

    Rows = models (n), columns = rankings / raters (k).

    Algorithm 2.5 of the paper: R_i = sum of the ranks of model i,
    S = sum (R_i - mean R)^2, W = 12 S / (k^2 (n^3 - n) - k T).
    With ``tie_correction=True`` (default, reproduces the values reported in the
    paper) tied ranks are replaced by average ranks and T = sum (t^3 - t) over the
    groups of t tied models; with ``tie_correction=False`` T = 0 and the ranks are
    used exactly as given.

    With ``return_p=True`` also returns the chi-square p-value
    (chi2 = k (n - 1) W, df = n - 1).
    """
    R = np.asarray(rankings, dtype=float)
    n, m = R.shape
    T = 0.0
    if tie_correction:
        R = np.column_stack([stats.rankdata(R[:, j]) for j in range(m)])
        for j in range(m):
            _, cnt = np.unique(R[:, j], return_counts=True)
            T += (cnt ** 3 - cnt).sum()
    Rsum = R.sum(axis=1)
    S = ((Rsum - Rsum.mean()) ** 2).sum()
    denom = m ** 2 * (n ** 3 - n) - m * T
    W = float(12 * S / denom) if denom > 0 else 1.0
    W = min(W, 1.0)
    if not return_p:
        return W
    chi2 = m * (n - 1) * W
    return W, float(stats.chi2.sf(chi2, n - 1))


def interpret_kappa(k: float) -> str:
    """Verbal scale of the paper (Table 2, Landis & Koch 1977):
    0.00-0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate, 0.61-0.80 substantial,
    0.81-1.00 almost perfect (negative values: poor)."""
    if k < 0:
        return "poor"
    for lim, lab in ((0.20, "slight"), (0.40, "fair"), (0.60, "moderate"),
                     (0.80, "substantial")):
        if k <= lim:
            return lab
    return "almost perfect"


def agreement_report(rankings: pd.DataFrame, weights: Optional[str] = None) -> dict:
    """Cohen's kappa matrix, mean pairwise kappa per ranking, Fleiss' kappa and Kendall's W."""
    cm = cohen_kappa_matrix(rankings, weights=weights)
    off = cm.where(~np.eye(len(cm), dtype=bool))
    fk = fleiss_kappa(rankings)
    W, p = kendall_w(rankings, return_p=True)
    return {
        "cohen_kappa": cm,
        "mean_cohen_kappa": off.mean(axis=1),
        "fleiss_kappa": fk,
        "fleiss_interpretation": interpret_kappa(fk),
        "kendall_w": W,
        "kendall_p_value": p,
    }
