"""Metric objectives: decide whether a metric is larger-the-better or smaller-the-better.

The direction of each metric is inferred from its column name (e.g. ``Train_R2``
-> larger-the-better, ``Test_RMSE`` -> smaller-the-better). Any inference can be
overridden explicitly with the ``objectives`` argument used throughout the package.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Mapping, Optional

LARGER = "larger"
SMALLER = "smaller"

# Keywords matched against the lower-cased metric name (without phase prefix).
_LARGER_KEYS = (
    "r2", "r^2", "r²", "rsquared", "r_squared", "adj_r2", "explained_variance",
    "accuracy", "acc", "f1", "precision", "recall", "auc", "roc_auc", "kappa",
    "mcc", "nse", "kge", "ia", "willmott", "pearson", "spearman", "a20", "a10",
    "score", "specificity", "sensitivity", "jaccard", "iou",
)
_SMALLER_KEYS = (
    "mse", "rmse", "mae", "mape", "smape", "msle", "rmsle", "medae", "maxerror",
    "max_error", "error", "loss", "time", "duration", "seconds", "sec", "memory",
    "bias", "std", "sd", "aic", "bic", "rsr", "cv_rmse",
)

_PHASE_PREFIX = re.compile(
    r"^(train(ing)?|test(ing)?|val(id(ation)?)?|total|all|full|cv)[\s_\-\.:/()]+",
    re.IGNORECASE,
)


def _clean(name: str) -> str:
    base = _PHASE_PREFIX.sub("", str(name).strip())
    base = re.sub(r"[\s\-\.()\[\]]+", "_", base.lower()).strip("_")
    return base


def infer_objective(metric_name: str) -> Optional[str]:
    """Return ``"larger"`` / ``"smaller"`` for a metric name, or ``None`` if unknown."""
    base = _clean(metric_name)
    tokens = set(re.split(r"_+", base)) | {base}
    # Exact-token match first (avoids 'mae' matching inside 'mape' etc.)
    for key in _SMALLER_KEYS:
        if key in tokens:
            return SMALLER
    for key in _LARGER_KEYS:
        if key in tokens:
            return LARGER
    # Fallback: substring match
    for key in _SMALLER_KEYS:
        if key in base:
            return SMALLER
    for key in _LARGER_KEYS:
        if key in base:
            return LARGER
    return None


def resolve_objectives(
    columns: Iterable[str], objectives: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    """Build the full ``{column: "larger"|"smaller"}`` mapping.

    Parameters
    ----------
    columns : iterable of str
        Metric column names.
    objectives : mapping, optional
        Explicit directions. Keys may be full column names (``"Test_MAE"``) or
        bare metric names (``"MAE"``) that apply to every phase. Values accept
        ``"larger"``, ``"max"``, ``"+"``, ``"smaller"``, ``"min"``, ``"-"``.
    """
    aliases = {
        "larger": LARGER, "max": LARGER, "maximize": LARGER, "+": LARGER,
        "larger-the-better": LARGER, "ltb": LARGER, "higher": LARGER,
        "smaller": SMALLER, "min": SMALLER, "minimize": SMALLER, "-": SMALLER,
        "smaller-the-better": SMALLER, "stb": SMALLER, "lower": SMALLER,
    }
    user = {}
    for k, v in (objectives or {}).items():
        key = str(v).strip().lower()
        if key not in aliases:
            raise ValueError(f"Unknown objective {v!r} for metric {k!r}")
        user[str(k)] = aliases[key]
    user_lower = {_clean(k): v for k, v in user.items()}

    out: Dict[str, str] = {}
    unknown = []
    for col in columns:
        if col in user:
            out[col] = user[col]
        elif _clean(col) in user_lower:
            out[col] = user_lower[_clean(col)]
        else:
            obj = infer_objective(col)
            if obj is None:
                unknown.append(col)
            else:
                out[col] = obj
    if unknown:
        raise ValueError(
            "Cannot infer whether these metrics are larger- or smaller-the-better: "
            f"{unknown}. Pass them explicitly, e.g. objectives={{'{unknown[0]}': 'larger'}}."
        )
    return out
