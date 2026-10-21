"""Command-line interface.

Examples
--------
    mlmrpi rank metrics.csv
    mlmrpi rank metrics.csv --phase Test --methods GRA MRNV --lam 0.5
    mlmrpi rank metrics.csv --objective Train_Time=smaller --out results.xlsx --plot ranks.png
    mlmrpi demo
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import __version__
from .core import METHODS, prepare_input, select_phase
from .ranker import MLmrpi


def _parse_obj(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--objective expects METRIC=larger|smaller, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _report(r: MLmrpi, agreement: bool):
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    print("\nMetric directions:", ", ".join(f"{k} ({v})" for k, v in r.objectives_.items()))
    print("\nMLmrpi scores and ranks (1 = best):")
    print(r.summary().round(4).to_string())
    print(f"\nBest model (consensus): {r.best_model()}")
    if agreement and len(r.methods) > 1:
        a = r.agreement()
        print("\nCohen's kappa between methods:")
        print(a["cohen_kappa"].round(2).to_string())
        print(f"\nFleiss' kappa = {a['fleiss_kappa']:.3f} ({a['fleiss_interpretation']})")
        print(f"Kendall's W   = {a['kendall_w']:.3f} (p = {a['kendall_p_value']:.2e})")


def main(argv=None):
    p = argparse.ArgumentParser(prog="mlmrpi",
                                description="Rank ML models with the ML Multi-Response Performance Index.")
    p.add_argument("--version", action="version", version=f"mlmrpi {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rank", help="rank the models of a CSV/Excel metric table")
    r.add_argument("file", help="CSV or Excel file: one row per model, one column per metric")
    r.add_argument("--model-col", default=None, help="name of the model column (default: first column)")
    r.add_argument("--phase", nargs="+", default=None,
                   help="keep only columns starting with these prefixes, e.g. Train Test")
    r.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    r.add_argument("--objective", nargs="+", metavar="METRIC=DIR",
                   help="override metric direction, e.g. Train_Time=smaller")
    r.add_argument("--lam", type=float, default=0.5, help="GRA distinguishing coefficient (default 0.5)")
    r.add_argument("--out", default=None, help="save results (.csv = summary, .xlsx = all tables)")
    r.add_argument("--plot", default=None, help="save a bar chart of the MLmrpi scores (png/pdf/svg)")
    r.add_argument("--no-agreement", action="store_true", help="skip kappa / Kendall statistics")

    d = sub.add_parser("demo", help="run on the paper's RAC dataset")
    d.add_argument("--dataset", default="rac", choices=["rac", "case_study_1", "case_study_2"])

    a = p.parse_args(argv)

    if a.cmd == "demo":
        from . import datasets
        df = {"rac": datasets.load_rac, "case_study_1": datasets.load_case_study_1,
              "case_study_2": datasets.load_case_study_2}[a.dataset]()
        res = MLmrpi().fit(df)
        _report(res, True)
        return 0

    df = prepare_input(a.file, model_col=a.model_col)
    if a.phase:
        df = select_phase(df, a.phase)
    res = MLmrpi(methods=a.methods, objectives=_parse_obj(a.objective), lam=a.lam).fit(df)
    _report(res, not a.no_agreement)
    if a.out:
        if a.out.lower().endswith((".xlsx", ".xls")):
            res.to_excel(a.out)
        else:
            res.summary().to_csv(a.out)
        print(f"\nResults written to {a.out}")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        res.plot("scores", save=a.plot)
        print(f"Figure written to {a.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
