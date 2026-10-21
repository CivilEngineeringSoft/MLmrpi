# Changelog

## 1.1.0 (2026-10-21)

Aligned with the paper (September 2025 version).

- **New `PartitionAnalysis` class** and `mlmrpi analyze` command. They rank the
  Train / Test / All-data partitions and the joint Train+Test set, then produce
  every table and figure of the paper: Tables 5 and 7, and Figs. 9–18.
- **Performance tiers** now follow Algorithm 2.2: Top 25 %, Middle 50 %, Bottom 25 %,
  with ⌈n/4⌉ models in the Top and Bottom tiers. This reproduces Table 7 exactly.
- **Kendall's W** is tie-corrected by default (`tie_correction=False` gives the
  plain formula).
- **Fleiss' κ for combined data splits** stacks the splits, and **Kendall's W**
  pools their rankings. These rules reproduce Fig. 10.
- **Figures** use the paper's styles and colours:
  - grouped MLmrpi bars (×100) with ranks on top: AW green, DEAR blue, GRA orange,
    FA red, MRNV gray;
  - `coolwarm` heat maps for Cohen's κ, correlations and Wilcoxon, and `YlGnBu`
    for Shapiro;
  - Fleiss' κ bars in skyblue / lightsteelblue and Kendall's W bars in lightcoral;
  - the average-κ chart (Fig. 17) in `tab10`.
- `MLmrpi.plot()` now defaults to the paper's grouped bar chart (`kind="bars"`).
- New tests check the results against the paper (Table 7, Fig. 9, Fig. 10).

## 1.0.0 (2026-10-21)

- First release: AW, DEAR, GRA, FA and MRNV methods.
- Cohen's κ, Fleiss' κ and Kendall's W.
- Correlations, performance tiers, Wilcoxon test and rank stability.
- Plots, command-line interface and example datasets.
