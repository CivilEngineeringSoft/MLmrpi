"""Command-line interface.

Examples
--------
    mlmrpi rank metrics.csv
    mlmrpi rank metrics.csv --phase Test --methods GRA MRNV --lam 0.5
    mlmrpi rank metrics.csv --objective Train_Time=smaller --out results.xlsx --plot ranks.png
    mlmrpi analyze metrics.csv --figures figs --fmt pdf
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

    pa = sub.add_parser("analyze", help="full multi-partition analysis of the paper "
                                        "(Train / Test / All data / Train+Test), tables and figures")
    pa.add_argument("file", help="CSV/Excel with columns Train_*, Test_*, Total_* (one row per model)")
    pa.add_argument("--model-col", default=None)
    pa.add_argument("--objective", nargs="+", metavar="METRIC=DIR")
    pa.add_argument("--lam", type=float, default=0.5)
    pa.add_argument("--no-joint", action="store_true", help="skip the joint Train+Test ranking")
    pa.add_argument("--out", default="mlmrpi_results.xlsx", help="Excel workbook with every table")
    pa.add_argument("--figures", default="mlmrpi_figures", help="folder for the figures")
    pa.add_argument("--fmt", default="pdf", choices=["pdf", "png", "svg"])

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

    if a.cmd == "analyze":
        import matplotlib
        matplotlib.use("Agg")
        from .partitions import PartitionAnalysis
        df = prepare_input(a.file, model_col=a.model_col)
        an = PartitionAnalysis(df, joint=None if a.no_joint else ("Train", "Test"),
                               objectives=_parse_obj(a.objective), lam=a.lam).fit()
        pd.set_option("display.width", 160)
        for lab in an.labels_ + ([an.joint_label_] if an.joint_label_ else []):
            print(f"\n=== MLmrpi ranks - {lab}")
            print(an.ranks(lab).sort_values(an.methods[0]).T.to_string())
        print("\nFleiss' kappa / Kendall's W among methods per data split:")
        print(an.split_agreement("methods").round(2).to_string())
        an.to_excel(a.out)
        files = an.plot_all(a.figures, fmt=a.fmt)
        print(f"\nTables written to {a.out}; {len(files)} figures written to {a.figures}/")
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
        res.plot("bars", save=a.plot)
        print(f"Figure written to {a.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
