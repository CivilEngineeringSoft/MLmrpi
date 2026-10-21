"""Reproduce the MLmrpi analysis of the paper on the bundled datasets.

    python examples/reproduce_paper.py

Writes tables and figures to examples/output/.
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from mlmrpi import MLmrpi, datasets, select_phase
from mlmrpi.agreement import agreement_report
from mlmrpi.plotting import plot_kappa

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 160)

df = datasets.load_rac()
phases = {"Train": "Train", "Test": "Test", "Total": "Total", "Train-Test": ["Train", "Test"]}

# 1. Rank-sum of the individual metrics (classical approach)
for name, ph in phases.items():
    if isinstance(ph, str):
        print(f"\n--- Rank-sum of individual metrics: {name}")
        print(MLmrpi().fit(select_phase(df, ph)).rank_by_metrics().T.to_string())

# 2. MLmrpi for each data partition
all_ranks = {}
for name, ph in phases.items():
    r = MLmrpi().fit(select_phase(df, ph))
    print(f"\n=== MLmrpi ranking: {name}")
    print(r.summary().round(4).to_string())
    a = r.agreement()
    print(f"Fleiss' kappa = {a['fleiss_kappa']:.2f} | Kendall's W = {a['kendall_w']:.2f}")
    r.plot("scores", save=OUT / f"mlmrpi_scores_{name}.png")
    r.to_excel(OUT / f"mlmrpi_{name}.xlsx")
    all_ranks.update({f"{m}({name})": r.ranks_[m] for m in r.methods})

    if name in ("Train", "Test", "Total"):
        print("Spearman correlation with the metrics:")
        print(r.correlations("spearman").round(2).to_string())
        print("Performance tiers:")
        print(r.tiers().round(2).to_string())
        print("Wilcoxon signed-rank (MLmrpi rank vs metric rank) - any significant?",
              r.significance()["Significant"].any())

# 3. Agreement of methods across partitions (Train, Test, Total)
R = pd.DataFrame(all_ranks)[[c for c in all_ranks if "Train-Test" not in c]]
rep = agreement_report(R)
plot_kappa(rep["cohen_kappa"], save=OUT / "cohen_kappa_methods_partitions.png")
print(f"\nAll partitions: Fleiss' kappa = {rep['fleiss_kappa']:.2f}, Kendall's W = {rep['kendall_w']:.2f}")

# 4. Independent case studies
for name, loader in [("Case study 1", datasets.load_case_study_1),
                     ("Case study 2", datasets.load_case_study_2)]:
    r = MLmrpi().fit(loader())
    a = r.agreement()
    print(f"\n=== {name}: best model = {r.best_model()}, "
          f"Fleiss' kappa = {a['fleiss_kappa']:.2f}, Kendall's W = {a['kendall_w']:.2f}")
    print(a["mean_cohen_kappa"].round(2).to_string())

print(f"\nFigures and workbooks written to {OUT}")
