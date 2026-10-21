"""Matplotlib figures for MLmrpi results (no seaborn dependency)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .core import rank_scores


def _plt():
    import matplotlib.pyplot as plt
    return plt


def plot_scores(scores: pd.DataFrame, ncols: Optional[int] = None, color: str = "#4C72B0",
                highlight: str = "#DD8452", figsize=None, save: Optional[str] = None):
    """One horizontal bar chart per method: MLmrpi value and rank written in each bar,
    models sorted from best (top) to worst. The best model is highlighted."""
    plt = _plt()
    k = scores.shape[1]
    ncols = ncols or min(k, 3)
    nrows = int(np.ceil(k / ncols))
    n = scores.shape[0]
    figsize = figsize or (4.6 * ncols, 0.38 * n * nrows + 1.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    ranks = rank_scores(scores)
    for ax, meth in zip(axes.flat, scores.columns):
        s = scores[meth].sort_values()
        cols = [highlight if ranks.loc[m, meth] == 1 else color for m in s.index]
        ax.barh(s.index, s.values, color=cols, edgecolor="none")
        span = s.max() - min(0, s.min())
        for y, (m, v) in enumerate(s.items()):
            ax.text(v + 0.01 * span, y, f"{v:.3f} ({ranks.loc[m, meth]})",
                    va="center", fontsize=7.5)
        ax.set_title(meth, fontweight="bold")
        ax.set_xlabel("$MLmrpi$")
        ax.set_xlim(min(0, s.min()), s.max() + 0.28 * span)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[k:]:
        ax.set_visible(False)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig


def _heatmap(ax, mat: pd.DataFrame, cmap, vmin, vmax, fmt, cbar_label, fig):
    im = ax.imshow(mat.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=8)
    mid = (vmin + vmax) / 2
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=7,
                        color="white" if (v > mid) == (cmap.endswith("_r")) else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)


def plot_rank_heatmap(ranks: pd.DataFrame, cmap: str = "viridis_r", figsize=None,
                      save: Optional[str] = None):
    """Models x methods heatmap of ranks (1 = best), sorted by mean rank."""
    plt = _plt()
    order = ranks.mean(axis=1).sort_values().index
    R = ranks.loc[order]
    fig, ax = plt.subplots(figsize=figsize or (1.0 * R.shape[1] + 2.5, 0.38 * R.shape[0] + 1.2))
    _heatmap(ax, R, cmap, 1, R.values.max(), "d", "Rank (1 = best)", fig)
    ax.set_title("$MLmrpi$ rankings")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig


def plot_kappa(kappa: pd.DataFrame, cmap: str = "RdYlGn", lower: bool = True, figsize=None,
               save: Optional[str] = None):
    """Cohen's kappa matrix as an annotated heatmap (lower triangle by default)."""
    plt = _plt()
    K = kappa.copy().astype(float)
    if lower:
        K = K.where(np.tril(np.ones(K.shape, dtype=bool)))
    fig, ax = plt.subplots(figsize=figsize or (0.6 * K.shape[1] + 2.5, 0.55 * K.shape[0] + 1.2))
    _heatmap(ax, K, cmap, -0.2, 1.0, ".2f", "Cohen's $\\kappa$", fig)
    ax.set_title("Pairwise ranking agreement")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig


def plot_radar(normalized: pd.DataFrame, models=None, figsize=(6, 6), save: Optional[str] = None):
    """Radar chart of the normalized metrics (1 = best) for selected models."""
    plt = _plt()
    models = list(models) if models is not None else list(
        normalized.mean(axis=1).sort_values(ascending=False).index[:5])
    labels = list(normalized.columns)
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
    for m in models:
        v = normalized.loc[m].tolist()
        v += v[:1]
        ax.plot(ang, v, lw=1.5, label=m)
        ax.fill(ang, v, alpha=0.08)
    ax.set_xticks(ang[:-1], labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig
