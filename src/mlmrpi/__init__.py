"""MLmrpi - Machine Learning Multi-Response Performance Index.

Rank machine-learning models from their precomputed performance metrics
(R2, RMSE, MAE, ... on training / testing sets) with a single composite index,
computed by five multi-response methods: AW, DEAR, GRA, FA and MRNV.

Quick start
-----------
>>> from mlmrpi import MLmrpi, datasets
>>> r = MLmrpi().fit(datasets.load_rac())
>>> r.summary()          # scores, ranks and consensus rank
>>> r.agreement()        # Cohen's kappa, Fleiss' kappa, Kendall's W
"""

__version__ = "1.0.0"

from . import agreement, datasets, robustness
from .agreement import cohen_kappa, cohen_kappa_matrix, fleiss_kappa, kendall_w
from .core import (METHODS, compute_mlmrpi, normalize, prepare_input, rank_by_metrics,
                   rank_scores, select_phase)
from .objectives import infer_objective, resolve_objectives
from .ranker import MLmrpi

__all__ = [
    "MLmrpi", "compute_mlmrpi", "rank_by_metrics", "rank_scores", "normalize",
    "prepare_input", "select_phase", "METHODS", "cohen_kappa", "cohen_kappa_matrix",
    "fleiss_kappa", "kendall_w", "infer_objective", "resolve_objectives",
    "agreement", "robustness", "datasets", "__version__",
]
