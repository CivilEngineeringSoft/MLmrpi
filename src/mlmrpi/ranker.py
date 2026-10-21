"""High-level, scikit-learn-like interface: ``MLmrpi().fit(table)``."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import pandas as pd

from . import agreement as _agr
from . import robustness as _rob
from .core import (METHODS, compute_mlmrpi, fa_weights, normalize, prepare_input,
                   rank_by_metrics, rank_scores, response_weights)
from .objectives import resolve_objectives


class MLmrpi:
    """Rank ML models from their precomputed performance metrics.

    Parameters
    ----------
    methods : sequence of str, default all five ("AW", "DEAR", "GRA", "FA", "MRNV")
    objectives : dict, optional
        Direction of each metric (``"larger"`` / ``"smaller"``); inferred from the
        column names when omitted (R2 -> larger; MSE, RMSE, MAE, time -> smaller).
    lam : float, default 0.5
        Distinguishing coefficient lambda of Grey Relational Analysis.
    weights : dict, optional
        Optional metric weights (used by GRA and MRNV only; default equal weights).
    eigen_threshold : float, default 1.0
        Kaiser criterion for retaining principal components in FA.

    Attributes (after ``fit``)
    --------------------------
    data_        : cleaned input table (models x metrics)
    objectives_  : resolved metric directions
    normalized_  : min-max normalized metrics (1 = best)
    scores_      : MLmrpi of every model for every method
    ranks_       : rank of every model for every method (1 = best)
    metric_ranks_: rank of every model for every individual metric
    fa_weights_  : PC_j weights used by the FA method
    """

    def __init__(self, methods: Sequence[str] = METHODS,
                 objectives: Optional[Mapping[str, str]] = None, lam: float = 0.5,
                 weights: Optional[Mapping[str, float]] = None,
                 eigen_threshold: float = 1.0):
        self.methods = [m.upper() for m in methods]
        bad = set(self.methods) - set(METHODS)
        if bad:
            raise ValueError(f"Unknown method(s) {sorted(bad)}; choose among {METHODS}.")
        self.objectives = objectives
        self.lam = lam
        self.weights = weights
        self.eigen_threshold = eigen_threshold

    # ------------------------------------------------------------------ fit
    def fit(self, data: Union[pd.DataFrame, str], model_col: Optional[str] = None,
            metrics: Optional[Sequence[str]] = None) -> "MLmrpi":
        df = prepare_input(data, model_col=model_col, metrics=metrics)
        self.data_ = df
        self.objectives_ = resolve_objectives(df.columns, self.objectives)
        self.normalized_ = normalize(df, self.objectives_)
        scores = {}
        for m in self.methods:
            scores[m] = compute_mlmrpi(
                df, method=m, objectives=self.objectives_, lam=self.lam,
                weights=self.weights if m in ("GRA", "MRNV") else None,
                eigen_threshold=self.eigen_threshold)
        self.scores_ = pd.DataFrame(scores)
        self.ranks_ = rank_scores(self.scores_)
        self.metric_ranks_ = _rob.metric_rank_table(df, self.objectives_)
        if "FA" in self.methods:
            self.fa_weights_ = fa_weights(self.normalized_, self.eigen_threshold)
        if {"AW", "DEAR"} & set(self.methods):
            self.aw_weights_ = response_weights(df, self.objectives_)
        return self

    def fit_transform(self, data, **kw) -> pd.DataFrame:
        return self.fit(data, **kw).summary()

    def _check(self):
        if not hasattr(self, "scores_"):
            raise RuntimeError("Call .fit(data) first.")

    # -------------------------------------------------------------- results
    def summary(self, sort_by: Optional[str] = None) -> pd.DataFrame:
        """Scores and ranks side by side, plus a consensus (mean) rank."""
        self._check()
        out = pd.concat({"MLmrpi": self.scores_, "Rank": self.ranks_}, axis=1)
        out[("Rank", "Mean")] = self.ranks_.mean(axis=1).round(2)
        out[("Rank", "Consensus")] = self.ranks_.mean(axis=1).rank(method="min").astype(int)
        key = ("Rank", sort_by.upper()) if sort_by else ("Rank", "Consensus")
        return out.sort_values(key)

    def best_model(self, method: Optional[str] = None) -> str:
        """Best model for one method, or by consensus rank if ``method`` is None."""
        self._check()
        if method is None:
            return self.ranks_.mean(axis=1).idxmin()
        return self.scores_[method.upper()].idxmax()

    def rank_by_metrics(self) -> pd.DataFrame:
        """Classic rank-sum aggregation of the individual metrics."""
        self._check()
        return rank_by_metrics(self.data_, objectives=self.objectives_)

    # ------------------------------------------------------------ agreement
    def agreement(self, include_metrics: bool = False, weights: Optional[str] = None) -> dict:
        """Cohen's kappa matrix, Fleiss' kappa and Kendall's W among the methods
        (optionally together with the single-metric rankings)."""
        self._check()
        R = self.ranks_
        if include_metrics:
            R = pd.concat([self.metric_ranks_, R], axis=1)
        return _agr.agreement_report(R, weights=weights)

    # ----------------------------------------------------------- robustness
    def correlations(self, method: str = "spearman") -> pd.DataFrame:
        self._check()
        return _rob.correlation_with_metrics(self.scores_, self.data_, method=method)

    def tiers(self, n_tiers: int = 3) -> pd.DataFrame:
        self._check()
        return _rob.performance_tiers(self.scores_, self.data_, n_tiers=n_tiers)

    def significance(self, alpha: float = 0.05) -> pd.DataFrame:
        self._check()
        return _rob.rank_difference_tests(self.ranks_, self.metric_ranks_, alpha=alpha)

    def stability(self, method: str = "GRA", n_iter: int = 500, noise: float = 0.02,
                  random_state: Optional[int] = 0) -> pd.DataFrame:
        """Rank stability of one method under small random perturbations of the metrics."""
        self._check()
        obj, lam, eig = self.objectives_, self.lam, self.eigen_threshold
        w = self.weights if method.upper() in ("GRA", "MRNV") else None
        f = lambda d: compute_mlmrpi(d, method=method, objectives=obj, lam=lam,
                                     weights=w, eigen_threshold=eig)
        return _rob.stability_to_perturbation(self.data_, f, n_iter=n_iter, noise=noise,
                                              random_state=random_state)

    # --------------------------------------------------------------- output
    def plot(self, kind: str = "scores", **kw):
        """Shortcut to :mod:`mlmrpi.plotting` (``kind`` = scores | ranks | kappa | radar)."""
        self._check()
        from . import plotting
        if kind == "scores":
            return plotting.plot_scores(self.scores_, **kw)
        if kind == "ranks":
            return plotting.plot_rank_heatmap(self.ranks_, **kw)
        if kind == "kappa":
            return plotting.plot_kappa(self.agreement(**{k: kw.pop(k) for k in
                                       ("include_metrics",) if k in kw})["cohen_kappa"], **kw)
        if kind == "radar":
            return plotting.plot_radar(self.normalized_, **kw)
        raise ValueError("kind must be 'scores', 'ranks', 'kappa' or 'radar'.")

    def to_excel(self, path: str) -> str:
        """Write every result table to one Excel workbook (one sheet per table)."""
        self._check()
        agr = self.agreement()
        with pd.ExcelWriter(path) as xl:
            self.data_.to_excel(xl, sheet_name="Input")
            pd.Series(self.objectives_, name="Objective").to_excel(xl, sheet_name="Objectives")
            self.normalized_.to_excel(xl, sheet_name="Normalized")
            self.summary().to_excel(xl, sheet_name="MLmrpi_ranking")
            self.rank_by_metrics().to_excel(xl, sheet_name="Metric_rank_sum")
            agr["cohen_kappa"].to_excel(xl, sheet_name="Cohen_kappa")
            pd.Series({"Fleiss kappa": agr["fleiss_kappa"], "Kendall W": agr["kendall_w"],
                       "Kendall p-value": agr["kendall_p_value"]}).to_excel(xl, sheet_name="Global_agreement")
            self.correlations("pearson").to_excel(xl, sheet_name="Pearson")
            self.correlations("spearman").to_excel(xl, sheet_name="Spearman")
            self.tiers().to_excel(xl, sheet_name="Tiers")
        return path

    def __repr__(self):
        fitted = f", fitted on {self.data_.shape[0]} models x {self.data_.shape[1]} metrics" \
            if hasattr(self, "data_") else ""
        return f"MLmrpi(methods={self.methods}, lam={self.lam}{fitted})"
