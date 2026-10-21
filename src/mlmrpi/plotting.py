"""Figures in the style and colours of the MLmrpi paper (matplotlib only).

* :func:`plot_mlmrpi_bars`      - grouped bars, MLmrpi (x100) with ranks on top (Fig. 11)
* :func:`plot_kappa`            - Cohen's kappa heat map, ``coolwarm`` (Figs. 9, 14-16)
* :func:`plot_split_agreement`  - Fleiss' kappa / Kendall's W per data split (Figs. 10, 18)
* :func:`plot_mean_kappa`       - average Cohen's kappa (Fig. 17)
* :func:`plot_correlation`      - Pearson / Spearman heat map (Fig. 12)
* :func:`plot_pvalues`          - Shapiro-Wilk / Wilcoxon p-values (Fig. 13)
* :func:`plot_scores`           - one horizontal bar chart per method
* :func:`plot_rank_heatmap`, :func:`plot_radar`
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .core import rank_scores

# ---------------------------------------------------------------- paper palette
METHOD_COLORS = {"AW": "green", "DEAR": "blue", "GRA": "orange", "FA": "red", "MRNV": "gray"}
METHOD_LABELS = {"AW": "Method 1 (AW)", "DEAR": "Method 2 (DEAR)", "GRA": "Method 3 (GRA)",
                 "FA": "Method 4 (FA)", "MRNV": "Method 5 (MRNV)"}
FLEISS_COLOR_METRICS = "skyblue"
FLEISS_COLOR_METHODS = "lightsteelblue"
KENDALL_COLOR = "lightcoral"
HEATMAP_CMAP = "coolwarm"
SHAPIRO_CMAP = "YlGnBu"
GROUP_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown",
                "tab:pink", "tab:gray"]


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _bold_axes(ax, xlabel=None, ylabel=None, fs=9):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight="bold", fontsize=fs)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=fs)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")


def _save(fig, save):
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig


# --------------------------------------------------------------- Fig. 11 style
def plot_mlmrpi_bars(scores: pd.DataFrame, scale: float = 100.0,
                     colors: Optional[Mapping[str, str]] = None,
                     labels: Optional[Mapping[str, str]] = None,
                     sort_models: bool = True, figsize=None, save: Optional[str] = None):
    """Grouped bar chart of the MLmrpi of every model and method, with each
    method's rank printed on top of its bar (paper Fig. 11).

    Colours: AW green, DEAR blue, GRA orange, FA red, MRNV gray.
    """
    plt = _plt()
    colors = {**METHOD_COLORS, **(colors or {})}
    labels = {**METHOD_LABELS, **(labels or {})}
    S = scores.sort_index(key=lambda i: i.str.lower()) if sort_models else scores
    R = rank_scores(S)
    n, k = S.shape
    width = 0.8 / k
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=figsize or (max(8, 0.75 * n + 1.5), 4.6))
    for j, m in enumerate(S.columns):
        pos = x - 0.4 + width * (j + 0.5)
        vals = S[m].to_numpy(float) * scale
        ax.bar(pos, vals, width, color=colors.get(m, GROUP_COLORS[j % 8]),
               label=labels.get(m, m))
        for p, v, rk in zip(pos, vals, R[m]):
            ax.text(p, v, str(rk), ha="center", va="bottom", fontsize=5.5, fontweight="bold")
    ax.set_xticks(x, S.index)
    ax.set_xlim(-0.6, n - 0.4)
    ax.margins(y=0.06)
    ylab = f"MLmrpi (x {scale:g})" if scale != 1 else "MLmrpi"
    _bold_axes(ax, "Model", ylab)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=k, fontsize=7,
              prop={"weight": "bold", "size": 7}, frameon=True)
    fig.tight_layout()
    return _save(fig, save)


# -------------------------------------------------------------- heat maps
def _heatmap(ax, fig, mat: pd.DataFrame, cmap, vmin, vmax, fmt, cbar_label,
             annot_size=8, linewidths=0.5, cbar_fmt=None):
    data = mat.values.astype(float)
    masked = np.ma.masked_invalid(data)
    im = ax.pcolormesh(masked, cmap=cmap, vmin=vmin, vmax=vmax,
                       edgecolors="white", linewidth=linewidths)
    ax.set_xlim(0, data.shape[1])
    ax.set_ylim(data.shape[0], 0)
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, mat.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, mat.index)
    ax.set_aspect("equal" if data.shape[0] == data.shape[1] else "auto")
    cm = im.cmap
    norm = im.norm
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                ax.text(j + 0.5, i + 0.5, "-", ha="center", va="center", fontsize=annot_size)
            else:
                r, g, b, _ = cm(norm(v))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(j + 0.5, i + 0.5, format(v, fmt), ha="center", va="center",
                        fontsize=annot_size, fontweight="bold",
                        color="white" if lum < 0.5 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format=cbar_fmt)
    cb.set_label(cbar_label, fontweight="bold")
    cb.ax.tick_params(labelsize=8)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
        t.set_fontsize(8)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def plot_kappa(kappa: pd.DataFrame, cmap: str = HEATMAP_CMAP, vmin=None, vmax: float = 1.0,
               lower: bool = False, annot_size: float = 8, figsize=None,
               save: Optional[str] = None):
    """Cohen's kappa matrix as an annotated ``coolwarm`` heat map (paper Figs. 9, 14-16)."""
    plt = _plt()
    K = kappa.copy().astype(float)
    if lower:
        K = K.where(np.tril(np.ones(K.shape, dtype=bool)))
    if vmin is None:
        vmin = float(np.nanmin(K.values))
    n = K.shape[0]
    fig, ax = plt.subplots(figsize=figsize or (0.55 * K.shape[1] + 3, 0.5 * n + 1.6))
    _heatmap(ax, fig, K, cmap, vmin, vmax, ".2f", "Cohen's Kappa", annot_size)
    fig.tight_layout()
    return _save(fig, save)


def plot_correlation(corr: pd.DataFrame, cbar_label: str = "Correlation Coefficient",
                     cmap: str = HEATMAP_CMAP, figsize=None, save: Optional[str] = None):
    """Correlation between MLmrpi (columns) and metrics (rows), paper Fig. 12."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=figsize or (0.42 * corr.shape[1] + 3, 0.38 * corr.shape[0] + 2))
    _heatmap(ax, fig, corr, cmap, -1, 1, ".2f", cbar_label, annot_size=5.5)
    _bold_axes(ax, "MLmrpi", "Metric")
    fig.tight_layout()
    return _save(fig, save)


def plot_pvalues(pvals: pd.DataFrame, cbar_label: str = "p-value", cmap: str = SHAPIRO_CMAP,
                 figsize=(5.0, 3.6), save: Optional[str] = None):
    """Method x metric p-values (paper Fig. 13: Shapiro ``YlGnBu``, Wilcoxon ``coolwarm``)."""
    plt = _plt()
    P = pvals.astype(float)
    fig, ax = plt.subplots(figsize=figsize)
    lo, hi = float(np.nanmin(P.values)), float(np.nanmax(P.values))
    _heatmap(ax, fig, P, cmap, lo, hi if hi > lo else lo + 1e-9, ".3f", cbar_label,
             annot_size=8, cbar_fmt="%.3g")
    ax.set_aspect("auto")
    _bold_axes(ax, "Individual Metric", "MLmrpi")
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    fig.tight_layout()
    return _save(fig, save)


# --------------------------------------------------------- Figs. 10 and 18
def plot_split_agreement(table: pd.DataFrame, column: str = "Fleiss kappa",
                         color: Optional[str] = None, figsize=(4.4, 3.2),
                         save: Optional[str] = None):
    """Bar chart of Fleiss' kappa or Kendall's W per data-split combination.

    ``table`` is the output of ``PartitionAnalysis.split_agreement`` (or any
    DataFrame indexed by split name). Colours as in the paper: Fleiss skyblue
    (lightsteelblue for methods), Kendall lightcoral.
    """
    plt = _plt()
    if color is None:
        color = KENDALL_COLOR if "kendall" in column.lower() else FLEISS_COLOR_METRICS
    vals = table[column].astype(float)
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(vals)), vals.values, color=color, edgecolor="black", zorder=3)
    for b, v in zip(bars, vals.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center",
                va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(range(len(vals)), vals.index, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", color="gray", zorder=0)
    ylabel = "Kendall's W" if "kendall" in column.lower() else "Fleiss' Kappa"
    _bold_axes(ax, "Data Split Combination", ylabel)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _save(fig, save)


def plot_mean_kappa(table: pd.DataFrame, colors: Optional[Sequence[str]] = None,
                    figsize=(9, 4.2), save: Optional[str] = None):
    """Grouped bars of the average Cohen's kappa of each metric/method (paper Fig. 17)."""
    plt = _plt()
    colors = list(colors or GROUP_COLORS)
    T = table.astype(float)
    n, k = T.shape
    width = 0.84 / k
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=figsize)
    for j, c in enumerate(T.columns):
        pos = x - 0.42 + width * (j + 0.5)
        ax.bar(pos, T[c].values, width, color=colors[j % len(colors)], label=c, zorder=3)
        for p, v in zip(pos, T[c].values):
            if np.isfinite(v):
                ax.text(p, v + 0.01, f"{v:.2f}", ha="center", va="bottom", rotation=90,
                        fontsize=5.5, fontweight="bold")
    ax.set_xticks(x, T.index)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle=":", color="gray", zorder=0)
    _bold_axes(ax, None, "Average Cohen's Kappa", fs=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3,
              prop={"weight": "bold", "size": 7})
    fig.tight_layout()
    return _save(fig, save)


# ------------------------------------------------------------ other figures
def plot_scores(scores: pd.DataFrame, ncols: Optional[int] = None,
                colors: Optional[Mapping[str, str]] = None, figsize=None,
                save: Optional[str] = None):
    """One horizontal bar chart per method (best model on top), MLmrpi value and
    rank written next to each bar, in the method colours of the paper."""
    plt = _plt()
    colors = {**METHOD_COLORS, **(colors or {})}
    k = scores.shape[1]
    ncols = ncols or min(k, 3)
    nrows = int(np.ceil(k / ncols))
    n = scores.shape[0]
    figsize = figsize or (4.6 * ncols, 0.38 * n * nrows + 1.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    ranks = rank_scores(scores)
    for ax, meth in zip(axes.flat, scores.columns):
        s = scores[meth].sort_values()
        ax.barh(s.index, s.values, color=colors.get(meth, "tab:blue"), edgecolor="black",
                linewidth=0.4)
        span = s.max() - min(0, s.min())
        for y, (m, v) in enumerate(s.items()):
            ax.text(v + 0.01 * span, y, f"{v:.3f} ({ranks.loc[m, meth]})",
                    va="center", fontsize=7.5, fontweight="bold")
        ax.set_title(METHOD_LABELS.get(meth, meth), fontweight="bold")
        ax.set_xlim(min(0, s.min()), s.max() + 0.3 * span)
        _bold_axes(ax, "MLmrpi")
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[k:]:
        ax.set_visible(False)
    fig.tight_layout()
    return _save(fig, save)


def plot_rank_heatmap(ranks: pd.DataFrame, cmap: str = "coolwarm", figsize=None,
                      save: Optional[str] = None):
    """Models x methods heat map of ranks (1 = best), sorted by mean rank."""
    plt = _plt()
    order = ranks.mean(axis=1).sort_values().index
    R = ranks.loc[order]
    fig, ax = plt.subplots(figsize=figsize or (0.9 * R.shape[1] + 2.8, 0.38 * R.shape[0] + 1.2))
    _heatmap(ax, fig, R, cmap, 1, R.values.max(), ".0f", "Rank (1 = best)")
    ax.set_aspect("auto")
    fig.tight_layout()
    return _save(fig, save)


def plot_radar(normalized: pd.DataFrame, models=None, figsize=(6, 6), save: Optional[str] = None):
    """Radar chart of the normalized metrics (1 = best) for selected models."""
    plt = _plt()
    models = list(models) if models is not None else list(
        normalized.mean(axis=1).sort_values(ascending=False).index[:5])
    labels = list(normalized.columns)
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
    for j, m in enumerate(models):
        v = normalized.loc[m].tolist()
        v += v[:1]
        ax.plot(ang, v, lw=1.5, label=m, color=GROUP_COLORS[j % 8])
        ax.fill(ang, v, alpha=0.08, color=GROUP_COLORS[j % 8])
    ax.set_xticks(ang[:-1], labels, fontsize=8, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    return _save(fig, save)
