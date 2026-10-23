"""Reproduce the MLmrpi analysis of the paper on the bundled datasets.

    python examples/reproduce_paper.py

Writes every table (Excel) and figure (PDF) to examples/output/.
Equivalent command line:  mlmrpi analyze <file.csv> --figures <folder>
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from mlmrpi import MLmrpi, PartitionAnalysis, datasets

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 30)

# ------------------------------------------------------------------ main dataset
pa = PartitionAnalysis(datasets.load_rac()).fit()        # Train / Test / All data + Train+Test

print("Rank-sum of the individual metrics (Table 5)")
print(pa.rank_sum_table().to_string())

for lab in pa.labels_ + [pa.joint_label_]:
    print(f"\nMLmrpi ranks - {lab} (Fig. 11)")
    print(pa.ranks(lab).sort_values("GRA").T.to_string())

print("\nFleiss' kappa / Kendall's W of the metrics (Fig. 10)")
print(pa.split_agreement("metrics").round(2).to_string())
print("\nFleiss' kappa / Kendall's W of the methods (Fig. 18 a, c)")
print(pa.split_agreement("methods").round(2).to_string())
print("\nTiered mean metrics, Top 25 % / Middle 50 % / Bottom 25 % (Table 7)")
print(pa.tiers().round(2).to_string())
print("\nAverage Cohen's kappa (Fig. 17)")
print(pa.mean_kappa().round(2).to_string())

pa.to_excel(OUT / "mlmrpi_RAC_all_tables.xlsx")
files = pa.plot_all(OUT / "figures_RAC", fmt="pdf")
print(f"\n{len(files)} figures written to {OUT / 'figures_RAC'}")

# -------------------------------------------------------- independent case studies
for name, loader in [("Case study 1 (Huang et al. 2023)", datasets.load_case_study_1),
                     ("Case study 2 (Cho et al. 2024)", datasets.load_case_study_2)]:
    r = MLmrpi().fit(loader())                 # all metrics of the table together
    a = r.agreement()
    print(f"\n=== {name}: best model = {r.best_model()}, "
          f"Fleiss' kappa = {a['fleiss_kappa']:.2f}, Kendall's W = {a['kendall_w']:.2f}")
    print("Average Cohen's kappa per method:")
    print(a["mean_cohen_kappa"].round(2).to_string())
    r.plot("bars", save=OUT / f"{name.split(' (')[0].replace(' ', '_')}_ranking.pdf")
