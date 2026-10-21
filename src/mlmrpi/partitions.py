"""Multi-partition analysis, as in the paper (Train / Test / All data and joint Train+Test).

The input table holds the metrics of every partition with a prefix
(``Train_R2``, ``Test_RMSE``, ``Total_MAE`` ...). Each partition is ranked
separately with the five methods; a joint ranking can also be computed on the
Train and Test metrics together. Rankings are labelled ``AW(Train)``,
``R2(Test)``, ``GRA(Train+Test)`` ... as in the paper figures.

>>> from mlmrpi import datasets
>>> from mlmrpi.partitions import PartitionAnalysis
>>> pa = PartitionAnalysis(datasets.load_rac()).fit()
>>> pa.split_agreement("methods")          # Fleiss' kappa & Kendall's W (Fig. 18a, c)
>>> pa.plot_all("figures/")                  # every figure of the paper
"""

from __future__ import annotations

import os
from itertools import combinations
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import agreement as _agr
from .core import METHODS, select_phase
from .ranker import MLmrpi
from .robustness import performance_tiers

# column prefix -> label used in the paper
DEFAULT_LABELS = {"Train": "Train", "Test": "Test", "Total": "All data", "All": "All data",
                  "Full": "All data", "Val": "Validation", "Validation": "Validation"}


def _detect_partitions(df: pd.DataFrame, sep: str = "_") -> Dict[str, str]:
    prefixes = []
    for c in df.columns:
        if sep in str(c):
            p = str(c).split(sep, 1)[0]
            if p not in prefixes:
                prefixes.append(p)
    if not prefixes:
        raise ValueError("No partition prefix found; name columns like 'Train_R2', 'Test_RMSE'.")
    return {p: DEFAULT_LABELS.get(p, p) for p in prefixes}


class PartitionAnalysis:
    """Rank the models in every data partition and compute all the agreement,
    correlation, tier and significance analyses reported in the paper.

    Parameters
    ----------
    data : DataFrame
        Models x metrics, columns prefixed by partition (``Train_``, ``Test_``, ``Total_``).
    partitions : dict, optional
        ``{prefix: label}``, e.g. ``{"Train": "Train", "Test": "Test", "Total": "All data"}``.
        Detected from the column prefixes when omitted.
    joint : tuple of two prefixes or None, default ("Train", "Test")
        Partitions combined into the joint ranking ``MLmrpi(Train+Test)``.
    methods, objectives, lam, eigen_threshold :
        Passed to :class:`mlmrpi.MLmrpi`. ``objectives`` may use bare metric names.
    """

    def __init__(self, data: pd.DataFrame, partitions: Optional[Mapping[str, str]] = None,
                 joint: Optional[Sequence[str]] = ("Train", "Test"),
                 methods: Sequence[str] = METHODS, objectives: Optional[Mapping[str, str]] = None,
                 lam: float = 0.5, eigen_threshold: float = 1.0, sep: str = "_"):
        self.data = data
        self.partitions = dict(partitions) if partitions else _detect_partitions(data, sep)
        self.joint = tuple(joint) if joint and all(j in self.partitions for j in joint) else None
        self.methods = [m.upper() for m in methods]
        self.kw = dict(methods=self.methods, objectives=objectives, lam=lam,
                       eigen_threshold=eigen_threshold)
        self.sep = sep

    # ------------------------------------------------------------------ fit
    def fit(self) -> "PartitionAnalysis":
        self.results_: Dict[str, MLmrpi] = {}
        for prefix, label in self.partitions.items():
            sub = select_phase(self.data, prefix, sep=self.sep, strip=True)
            self.results_[label] = MLmrpi(**self.kw).fit(sub)
        self.labels_ = list(self.results_)
        self.metric_names_ = list(next(iter(self.results_.values())).data_.columns)

        self.joint_label_ = None
        if self.joint:
            self.joint_label_ = "+".join(self.partitions[j] for j in self.joint)
            sub = select_phase(self.data, list(self.joint), sep=self.sep)
            self.joint_result_ = MLmrpi(**self.kw).fit(sub)

        models = next(iter(self.results_.values())).data_.index
        mr, me, sc = {}, {}, {}
        for lab, r in self.results_.items():
            for m in self.methods:
                mr[f"{m}({lab})"] = r.ranks_[m]
                sc[f"{m}({lab})"] = r.scores_[m]
            for k in r.data_.columns:
                me[f"{k}({lab})"] = r.metric_ranks_[k]
        if self.joint_label_:
            for m in self.methods:
                mr[f"{m}({self.joint_label_})"] = self.joint_result_.ranks_[m]
                sc[f"{m}({self.joint_label_})"] = self.joint_result_.scores_[m]
        self.method_ranks_ = pd.DataFrame(mr).loc[models]
        self.metric_ranks_ = pd.DataFrame(me).loc[models]
        self.scores_ = pd.DataFrame(sc).loc[models]
        return self

    def _check(self):
        if not hasattr(self, "results_"):
            raise RuntimeError("Call .fit() first.")

    def _method_cols(self, labels):
        return [f"{m}({l})" for l in labels for m in self.methods]

    # ------------------------------------------------------------ rankings
    def scores(self, label: str) -> pd.DataFrame:
        """MLmrpi of every model for one partition (or the joint label)."""
        self._check()
        if label == self.joint_label_:
            return self.joint_result_.scores_
        return self.results_[label].scores_

    def ranks(self, label: str) -> pd.DataFrame:
        self._check()
        if label == self.joint_label_:
            return self.joint_result_.ranks_
        return self.results_[label].ranks_

    def rank_sum_table(self) -> pd.DataFrame:
        """Classic rank-sum table of the individual metrics for every partition
        (paper Table 5), plus the Train-Test total when both exist."""
        self._check()
        blocks = {lab: r.rank_by_metrics().T for lab, r in self.results_.items()}
        if self.joint:
            a, b = (self.partitions[j] for j in self.joint)
            tot = blocks[a].loc["Overall Rank"] + blocks[b].loc["Overall Rank"]
            blocks[self.joint_label_] = pd.DataFrame(
                {"Total Rank": tot, "Overall Rank": tot.rank(method="min").astype(int)}).T
        return pd.concat(blocks)

    # ----------------------------------------------------------- agreement
    def cohen(self, kind: str = "methods", label: Optional[str] = None) -> pd.DataFrame:
        """Cohen's kappa matrices of the paper.

        kind = "metrics"         : metrics across partitions (Fig. 9)
               "methods"         : methods across partitions (Fig. 14)
               "metrics+methods" : metrics and methods of one partition ``label`` (Fig. 15 a-c);
                                   with ``label=None`` all partitions together (Fig. 15 d)
               "joint"           : metrics of partition ``label`` vs MLmrpi(Train+Test) (Fig. 16)
        """
        self._check()
        if kind == "metrics":
            cols = [f"{k}({l})" for k in self.metric_names_ for l in self.labels_]
            return _agr.cohen_kappa_matrix(self.metric_ranks_[cols])
        if kind == "methods":
            return _agr.cohen_kappa_matrix(self.method_ranks_[self._method_cols(self.labels_)])
        if kind == "metrics+methods":
            if label is not None:
                r = self.results_[label]
                return _agr.cohen_kappa_matrix(pd.concat([r.metric_ranks_, r.ranks_], axis=1))
            # stacked over partitions: each (model, partition) pair is one item
            stacked = pd.concat([pd.concat([r.metric_ranks_, r.ranks_], axis=1)
                                 for r in self.results_.values()], axis=0, ignore_index=True)
            return _agr.cohen_kappa_matrix(stacked)
        if kind == "joint":
            if not self.joint_label_:
                raise ValueError("No joint partition defined.")
            label = label or self.labels_[0]
            me = self.metric_ranks_[[f"{k}({label})" for k in self.metric_names_]]
            mr = self.method_ranks_[self._method_cols([self.joint_label_])]
            return _agr.cohen_kappa_matrix(pd.concat([me, mr], axis=1))
        raise ValueError("kind must be 'metrics', 'methods', 'metrics+methods' or 'joint'.")

    def split_combinations(self):
        labs = self.labels_
        return [c for k in range(1, len(labs) + 1) for c in combinations(labs, k)]

    def split_agreement(self, which: str = "methods") -> pd.DataFrame:
        """Fleiss' kappa and Kendall's W for every data-split combination
        (Train, Test, All data, Train+Test, ..., Train+Test+All data).

        which = "metrics" (Fig. 10), "methods" (Fig. 18 a, c) or "both" (Fig. 18 b, d).

        For a combination of splits:
        * Fleiss' kappa - the splits are stacked: every (model, split) pair is one
          rated item and the raters are the metrics / methods;
        * Kendall's W   - the rankings of all splits are pooled as raters of the
          same models (tie-corrected).
        These rules reproduce the values reported in the paper.
        """
        self._check()
        names = []
        if which in ("metrics", "both"):
            names += self.metric_names_
        if which in ("methods", "both"):
            names += self.methods
        allr = pd.concat([self.metric_ranks_, self.method_ranks_], axis=1)
        rows = {}
        for combo in self.split_combinations():
            blocks = [allr[[f"{k}({l})" for k in names]].set_axis(names, axis=1) for l in combo]
            stacked = pd.concat(blocks, axis=0, ignore_index=True)
            pooled = pd.concat(blocks, axis=1)
            rows["+".join(combo)] = {"Fleiss kappa": _agr.fleiss_kappa(stacked),
                                     "Kendall W": _agr.kendall_w(pooled)}
        return pd.DataFrame(rows).T

    def mean_kappa(self) -> pd.DataFrame:
        """Average Cohen's kappa of every metric and method with all the others
        (paper Fig. 17), per partition and versus the joint MLmrpi."""
        self._check()
        out = {}
        names = {"Train": "Training", "Test": "Testing"}
        for lab in self.labels_:
            K = self.cohen("metrics+methods", lab)
            out[names.get(lab, lab)] = K.where(~np.eye(len(K), dtype=bool)).mean(axis=1)
        if self.joint_label_:
            for lab in self.labels_:
                K = self.cohen("joint", lab)
                K.index = [i.replace(f"({lab})", "").replace(f"({self.joint_label_})", "")
                           for i in K.index]
                key = f"{'All-data' if lab == 'All data' else lab} metrics vs MLmrpi ({self.joint_label_.lower()})"
                out[key] = K.where(~np.eye(len(K), dtype=bool)).mean(axis=1).values
                out[key] = pd.Series(out[key], index=K.index)
        return pd.DataFrame(out)

    # ---------------------------------------------------------- robustness
    def correlations(self, method: str = "pearson",
                     metrics: Sequence[str] = ("R2", "RMSE", "MAE")) -> pd.DataFrame:
        """Correlation between MLmrpi and the individual metrics (Fig. 12).

        Pearson uses MLmrpi values and metric values; Spearman uses the rankings
        (Algorithm 2.1). Each method is correlated with the metrics of its own
        partition; the joint Train+Test MLmrpi with the metrics of every partition.
        Other cells are NaN.
        """
        self._check()
        metrics = [k for k in metrics if k in self.metric_names_] or self.metric_names_
        rows = [f"{k}({l})" for l in self.labels_ for k in metrics]
        cols = list(self.scores_.columns)
        out = pd.DataFrame(np.nan, index=rows, columns=cols)
        for col in cols:
            meth_lab = col[col.index("(") + 1:-1]
            for row in rows:
                k, lab = row[:row.index("(")], row[row.index("(") + 1:-1]
                if meth_lab != lab and meth_lab != self.joint_label_:
                    continue
                if method == "pearson":
                    x, y = self.scores_[col], self.results_[lab].data_[k]
                    out.loc[row, col] = stats.pearsonr(x, y)[0]
                else:
                    # rank 1 = best for both; sign flipped so that error metrics give
                    # negative correlations, as in the paper
                    x = self.method_ranks_[col]
                    y = self.metric_ranks_[row]
                    rho = stats.spearmanr(x, y)[0]
                    larger = self.results_[lab].objectives_[k] == "larger"
                    out.loc[row, col] = rho if larger else -rho
        return out

    def tiers(self, fractions=(0.25, 0.50, 0.25),
              metrics: Sequence[str] = ("R2", "RMSE", "MAE")) -> pd.DataFrame:
        """Tiered mean metrics for every partition (paper Table 7)."""
        self._check()
        blocks = {}
        for lab, r in self.results_.items():
            cols = [k for k in metrics if k in r.data_.columns]
            blocks[lab] = performance_tiers(r.scores_, r.data_[cols], fractions=fractions)
        if self.joint_label_:
            r = self.joint_result_
            first = self.joint[0]
            cols = [f"{first}{self.sep}{k}" for k in metrics if f"{first}{self.sep}{k}" in r.data_.columns]
            t = performance_tiers(r.scores_, r.data_[cols], fractions=fractions)
            t.columns = [c.split(self.sep, 1)[1] for c in t.columns]
            blocks[self.joint_label_] = t
        return pd.concat(blocks, names=["Partition"])

    def significance(self, metrics: Optional[Sequence[str]] = None) -> Dict[str, pd.DataFrame]:
        """Shapiro-Wilk and Wilcoxon signed-rank p-values (Fig. 13), method x metric.

        For each (method, metric) pair the rank differences of all partitions are
        pooled. Returns ``{"shapiro": DataFrame, "wilcoxon": DataFrame}``.
        """
        self._check()
        metrics = list(metrics) if metrics else self.metric_names_
        sw = pd.DataFrame(index=self.methods, columns=metrics, dtype=float)
        wx = sw.copy()
        for m in self.methods:
            for k in metrics:
                d = np.concatenate([(r.ranks_[m] - r.metric_ranks_[k]).to_numpy(float)
                                    for r in self.results_.values()])
                sw.loc[m, k] = stats.shapiro(d)[1] if np.ptp(d) > 0 else np.nan
                wx.loc[m, k] = stats.wilcoxon(d, zero_method="zsplit")[1] if np.any(d) else 1.0
        return {"shapiro": sw, "wilcoxon": wx}

    # ---------------------------------------------------------------- output
    def plot_all(self, folder: str = "mlmrpi_figures", fmt: str = "pdf") -> list:
        """Write every figure of the paper to ``folder``; returns the file names."""
        from . import plotting as P
        import matplotlib.pyplot as plt
        self._check()
        os.makedirs(folder, exist_ok=True)
        files = []

        def save(fig, name):
            path = os.path.join(folder, f"{name}.{fmt}")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            files.append(path)

        for lab in self.labels_ + ([self.joint_label_] if self.joint_label_ else []):
            save(P.plot_mlmrpi_bars(self.scores(lab)), f"MLmrpi_ranking_{lab.replace(' ', '_').replace('+', '_')}")
        save(P.plot_kappa(self.cohen("metrics")), "Cohen_kappa_metrics")
        sa = self.split_agreement("metrics")
        save(P.plot_split_agreement(sa, "Fleiss kappa"), "Fleiss_kappa_metrics")
        save(P.plot_split_agreement(sa, "Kendall W"), "Kendall_W_metrics")
        save(P.plot_correlation(self.correlations("pearson"), "Correlation Coefficient"), "Pearson_MLmrpi_metrics")
        save(P.plot_correlation(self.correlations("spearman"), "Spearman rank correlation coefficient"),
             "Spearman_MLmrpi_metrics")
        sig = self.significance()
        save(P.plot_pvalues(sig["shapiro"], "Shapiro p-value", cmap="YlGnBu"), "Shapiro_Wilk")
        save(P.plot_pvalues(sig["wilcoxon"], "Wilcoxon p-value", cmap="coolwarm"), "Wilcoxon")
        save(P.plot_kappa(self.cohen("methods")), "Cohen_kappa_methods")
        for lab in self.labels_:
            save(P.plot_kappa(self.cohen("metrics+methods", lab), annot_size=7),
                 f"Cohen_kappa_metrics_methods_{lab.replace(' ', '_')}")
        save(P.plot_kappa(self.cohen("metrics+methods"), annot_size=7), "Cohen_kappa_metrics_methods_combined")
        if self.joint_label_:
            for lab in self.labels_:
                save(P.plot_kappa(self.cohen("joint", lab)),
                     f"Cohen_kappa_{lab.replace(' ', '_')}_metrics_vs_MLmrpi_joint")
        save(P.plot_mean_kappa(self.mean_kappa()), "Average_Cohen_kappa")
        for which, color in (("methods", "lightsteelblue"), ("both", "lightsteelblue")):
            sa = self.split_agreement(which)
            save(P.plot_split_agreement(sa, "Fleiss kappa", color=color), f"Fleiss_kappa_{which}")
            save(P.plot_split_agreement(sa, "Kendall W"), f"Kendall_W_{which}")
        return files

    def to_excel(self, path: str) -> str:
        """All tables of the multi-partition analysis in one workbook."""
        self._check()
        sig = self.significance()
        with pd.ExcelWriter(path) as xl:
            self.data.to_excel(xl, sheet_name="Input")
            self.scores_.to_excel(xl, sheet_name="MLmrpi_scores")
            self.method_ranks_.to_excel(xl, sheet_name="MLmrpi_ranks")
            self.rank_sum_table().to_excel(xl, sheet_name="Metric_rank_sum")
            self.cohen("metrics").to_excel(xl, sheet_name="Kappa_metrics")
            self.cohen("methods").to_excel(xl, sheet_name="Kappa_methods")
            self.split_agreement("metrics").to_excel(xl, sheet_name="Fleiss_Kendall_metrics")
            self.split_agreement("methods").to_excel(xl, sheet_name="Fleiss_Kendall_methods")
            self.split_agreement("both").to_excel(xl, sheet_name="Fleiss_Kendall_both")
            self.mean_kappa().to_excel(xl, sheet_name="Average_kappa")
            self.correlations("pearson").to_excel(xl, sheet_name="Pearson")
            self.correlations("spearman").to_excel(xl, sheet_name="Spearman")
            self.tiers().to_excel(xl, sheet_name="Tiers")
            sig["shapiro"].to_excel(xl, sheet_name="Shapiro")
            sig["wilcoxon"].to_excel(xl, sheet_name="Wilcoxon")
        return path
