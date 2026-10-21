# MLmrpi — Machine Learning Multi-Response Performance Index

[![tests](https://github.com/CivilEngineeringSoft/MLmrpi/actions/workflows/tests.yml/badge.svg)](https://github.com/CivilEngineeringSoft/MLmrpi/actions)
![python](https://img.shields.io/badge/python-%E2%89%A53.9-blue)
![license](https://img.shields.io/badge/license-MIT-green)

**MLmrpi** ranks machine-learning models from their **precomputed performance metrics**
(R², MSE, RMSE, MAE, training time, … on the training, testing and/or full datasets)
with a **single composite index**. Instead of comparing a dozen numbers by eye,
you obtain one score per model — higher is better — computed by five
multi-response methods:

| Code   | Method                                   | Normalization | MLmrpi                                   |
|--------|------------------------------------------|---------------|------------------------------------------|
| `AW`   | Assignment of Weights                    | no            | Σⱼ Wᵢⱼ·Rᵢⱼ                               |
| `DEAR` | Data Envelopment Analysis-based Ranking  | no            | Σ larger-the-better Wᵢⱼ·Rᵢⱼ / Σ smaller-the-better Wᵢⱼ·Rᵢⱼ |
| `GRA`  | Grey Relational Analysis                 | min–max       | grey relational grade (1/m) Σⱼ GCᵢⱼ       |
| `FA`   | Factor Analysis (principal components)   | min–max       | Σⱼ ηᵢⱼ·PCⱼ                               |
| `MRNV` | Multi-Response Normalized Value          | min–max       | (1/m) Σⱼ ηᵢⱼ                             |

The package also provides the statistics used in the paper to check that a ranking
is trustworthy: **Cohen's κ**, **Fleiss' κ**, **Kendall's W**, Pearson/Spearman
correlation with the individual metrics, Top/Middle/Bottom **performance tiers**,
Shapiro–Wilk + **Wilcoxon** signed-rank tests, and a Monte-Carlo **rank-stability** test.

It works for regression and classification alike: any table with one row per model
and one numeric column per metric.

---

## Installation

```bash
pip install git+https://github.com/CivilEngineeringSoft/MLmrpi.git
# or, from a local clone
git clone https://github.com/CivilEngineeringSoft/MLmrpi.git
cd MLmrpi
pip install -e ".[excel]"
```

Requirements: Python ≥ 3.9, numpy, pandas, scipy, matplotlib (openpyxl for Excel export).

## Quick start

### 1. Prepare the input table (Step 1 of the method)

One row per model, one column per metric. Phase prefixes (`Train_`, `Test_`, `Total_`)
are optional but let you select a phase later.

| Model    | Train_R2 | Train_RMSE | Train_MAE | Test_R2 | Test_RMSE | Test_MAE |
|----------|---------:|-----------:|----------:|--------:|----------:|---------:|
| CatBoost |     0.99 |       1.46 |      1.00 |    0.83 |      5.02 |     3.20 |
| XGBoost  |     0.98 |       1.62 |      1.15 |    0.84 |      4.89 |     3.25 |
| SVR      |     0.64 |       7.63 |      5.30 |    0.49 |      8.59 |     6.15 |
| …        |        … |          … |         … |       … |         … |        … |

### 2. Rank the models (Python)

```python
import pandas as pd
from mlmrpi import MLmrpi

table = pd.read_csv("my_metrics.csv", index_col="Model")

r = MLmrpi().fit(table)          # all five methods
print(r.summary())               # MLmrpi scores, ranks, consensus rank
print(r.best_model())            # best model by consensus rank

agr = r.agreement()              # Cohen's κ matrix, Fleiss' κ, Kendall's W
print(agr["fleiss_kappa"], agr["kendall_w"])

r.plot("scores", save="mlmrpi_scores.png")   # bar charts, one per method
r.to_excel("mlmrpi_results.xlsx")            # every table in one workbook
```

Single method, functional style:

```python
from mlmrpi import compute_mlmrpi, rank_scores
grade = compute_mlmrpi(table, method="GRA", lam=0.5)
ranks = rank_scores(grade)       # 1 = best
```

### 3. Or from the command line

```bash
mlmrpi rank my_metrics.csv
mlmrpi rank my_metrics.csv --phase Test --methods GRA MRNV
mlmrpi rank my_metrics.csv --objective Train_Time=smaller --out results.xlsx --plot scores.png
mlmrpi demo                      # runs on the paper's dataset
```

## Metric directions

Whether a metric is *larger-the-better* or *smaller-the-better* is inferred from its name
(R², accuracy, F1, AUC, NSE … → larger; MSE, RMSE, MAE, MAPE, loss, time … → smaller).
Anything else must be declared:

```python
MLmrpi(objectives={"Train_Time": "smaller", "A20_index": "larger"}).fit(table)
# a bare name applies to every phase: {"MAE": "smaller"}
```

The resolved directions are stored in `r.objectives_`.

## Options

| Argument          | Default        | Meaning |
|-------------------|----------------|---------|
| `methods`         | all five       | subset of `("AW","DEAR","GRA","FA","MRNV")` |
| `objectives`      | inferred       | metric directions |
| `lam`             | `0.5`          | GRA distinguishing coefficient λ ∈ [0, 1] |
| `weights`         | equal          | optional metric weights (GRA and MRNV only) |
| `eigen_threshold` | `1.0`          | Kaiser criterion for retaining components in FA |

## Results available after `fit`

| Attribute / method        | Content |
|---------------------------|---------|
| `scores_`, `ranks_`       | MLmrpi and rank of every model for every method |
| `normalized_`             | min–max normalized metrics (1 = best) |
| `fa_weights_`             | PCⱼ weights used by FA |
| `metric_ranks_`           | rank of every model for each metric separately |
| `summary()`               | scores + ranks + mean and consensus rank |
| `rank_by_metrics()`       | classical rank-sum aggregation of the individual metrics |
| `agreement(include_metrics=False)` | Cohen's κ matrix, mean κ, Fleiss' κ, Kendall's W (+ p-value) |
| `correlations("pearson"/"spearman")` | correlation of each MLmrpi with each metric |
| `tiers(n_tiers=3)`        | mean metrics of the Top / Middle / Bottom models |
| `significance()`          | Shapiro–Wilk and Wilcoxon p-values, MLmrpi rank vs. metric rank |
| `stability(method, noise=0.02)` | rank stability under ±2 % random perturbation of the metrics |
| `plot(kind)`              | `"scores"`, `"ranks"`, `"kappa"`, `"radar"` |
| `to_excel(path)`          | all of the above in one workbook |

## Example datasets

```python
from mlmrpi import datasets, select_phase
df = datasets.load_rac()            # 14 regressors, RAC compressive strength (paper)
df1 = datasets.load_case_study_1()  # Huang et al. (2023), 12 models
df2 = datasets.load_case_study_2()  # Cho et al. (2024), 7 models, incl. training time

MLmrpi().fit(select_phase(df, "Test")).summary()
MLmrpi().fit(select_phase(df, ["Train", "Test"])).summary()   # joint train–test ranking
```

See `examples/reproduce_paper.py` for the full analysis of the paper.

## Method details

Let Rᵢⱼ be metric *j* of model *i* (n models, m metrics).

**Normalization** (GRA, FA, MRNV): ηᵢⱼ = (Rᵢⱼ − minᵢ Rᵢⱼ)/(maxᵢ Rᵢⱼ − minᵢ Rᵢⱼ) for
larger-the-better, (maxᵢ Rᵢⱼ − Rᵢⱼ)/(maxᵢ Rᵢⱼ − minᵢ Rᵢⱼ) for smaller-the-better.
A metric that is identical for all models carries no ranking information (η = 1).

**Weights** (AW, DEAR): Wᵢⱼ = Rᵢⱼ / Σᵢ Rᵢⱼ (larger-the-better),
Wᵢⱼ = (1/Rᵢⱼ) / Σᵢ (1/Rᵢⱼ) (smaller-the-better).

**Grey coefficient**: GCᵢⱼ = (Δmin + λΔmax)/(Δᵢⱼ + λΔmax), Δᵢⱼ = maxᵢ ηᵢⱼ − ηᵢⱼ.

**FA weights**: eigen-decomposition of the correlation matrix of η; components with
eigenvalue > 1 are retained; PCⱼ = Σₖ (λₖ/Σλ) |Lⱼₖ| with loadings Lⱼₖ = vⱼₖ√λₖ,
rescaled to Σⱼ PCⱼ = 1.

**Ranking**: descending MLmrpi; tied models share the best rank.

## Citation

If you use MLmrpi, please cite the paper (see `CITATION.cff`):


> Cirhuza Badesire, P., *et al.* (2026). Ranking Machine Learning Models for Predicting 
> Recycled Aggregate Concrete Properties: Application of Multi-Response Methods.
> 


## License

MIT — © 2026 Université Claude Bernard Lyon 1 (LMC², EA 7427). See `LICENSE`.
