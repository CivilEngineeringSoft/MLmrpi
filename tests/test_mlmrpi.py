import numpy as np
import pandas as pd
import pytest

from mlmrpi import (MLmrpi, cohen_kappa, compute_mlmrpi, datasets, fleiss_kappa,
                    infer_objective, kendall_w, normalize, rank_by_metrics, select_phase)
from mlmrpi.core import grey_coefficients
from mlmrpi.objectives import resolve_objectives


@pytest.fixture
def toy():
    return pd.DataFrame({"R2": [0.9, 0.8, 0.6], "RMSE": [2.0, 3.0, 5.0]},
                        index=["A", "B", "C"])


OBJ = {"R2": "larger", "RMSE": "smaller"}


# --------------------------------------------------------------- objectives
@pytest.mark.parametrize("name,exp", [("Train_R2", "larger"), ("Test_RMSE", "smaller"),
                                      ("Total_MAE", "smaller"), ("Train_Time", "smaller"),
                                      ("MAPE", "smaller"), ("accuracy", "larger"), ("F1", "larger")])
def test_infer(name, exp):
    assert infer_objective(name) == exp


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        resolve_objectives(["Foo"])
    assert resolve_objectives(["Foo"], {"Foo": "max"}) == {"Foo": "larger"}


# ------------------------------------------------------------------ methods
def test_normalize(toy):
    eta = normalize(toy, OBJ)
    assert np.allclose(eta["R2"], [1, 2 / 3, 0])
    assert np.allclose(eta["RMSE"], [1, 2 / 3, 0])


def test_aw_by_hand(toy):
    s = compute_mlmrpi(toy, "AW", OBJ)
    r2 = toy["R2"]
    stb = 1 / (1 / toy["RMSE"]).sum()          # (1/R)/sum(1/R) * R
    assert np.allclose(s, r2 ** 2 / r2.sum() + stb)


def test_dear_by_hand(toy):
    s = compute_mlmrpi(toy, "DEAR", OBJ)
    num = toy["R2"] ** 2 / toy["R2"].sum()
    den = 1 / (1 / toy["RMSE"]).sum()
    assert np.allclose(s, num / den)


def test_gra_by_hand(toy):
    eta = normalize(toy, OBJ)
    delta = 1 - eta                      # dmin = 0, dmax = 1
    gc = 0.5 / (delta + 0.5)
    assert np.allclose(grey_coefficients(eta, 0.5), gc)
    assert np.allclose(compute_mlmrpi(toy, "GRA", OBJ), gc.mean(axis=1))


def test_mrnv_by_hand(toy):
    assert np.allclose(compute_mlmrpi(toy, "MRNV", OBJ), normalize(toy, OBJ).mean(axis=1))


def test_fa_weights_sum_to_one():
    r = MLmrpi(methods=["FA"]).fit(datasets.load_rac())
    assert np.isclose(r.fa_weights_.sum(), 1.0)
    assert (r.fa_weights_ >= 0).all()


def test_constant_metric_does_not_break(toy):
    t = toy.assign(MAE=1.0)
    r = MLmrpi(objectives={**OBJ, "MAE": "smaller"}).fit(t)
    assert list(r.ranks_.index[r.ranks_["GRA"] == 1]) == ["A"]


def test_dominant_model_ranks_first_everywhere(toy):
    r = MLmrpi(objectives=OBJ).fit(toy)
    assert (r.ranks_.loc["A"] == 1).all() and (r.ranks_.loc["C"] == 3).all()
    assert r.best_model() == "A"


# ------------------------------------------------------------ paper dataset
def test_paper_table_train_mse_ranks():
    """Reproduces the Train-MSE row of the paper's rank-aggregation table."""
    ranks = rank_by_metrics(select_phase(datasets.load_rac(), "Train"))
    expected = dict(AB=12, BR=8, CatBoost=4, DT=11, ET=2, GBoost=6, HistGB=3, KNN=1,
                    LGBM=5, RF=9, SGD=14, SVR=13, Voting=10, XGBoost=7)
    assert ranks["Train_MSE"].to_dict() == expected


def test_paper_best_and_worst():
    df = datasets.load_rac()
    test = MLmrpi().fit(select_phase(df, "Test"))
    assert test.ranks_.loc["XGBoost", "GRA"] == 1
    assert (test.ranks_.loc["SGD"] == 14).all()
    total = MLmrpi().fit(select_phase(df, "Total"))
    assert total.ranks_.loc["ET", "GRA"] == 1


def test_case_studies_run():
    for loader in (datasets.load_case_study_1, datasets.load_case_study_2):
        r = MLmrpi().fit(loader())
        assert r.scores_.shape[1] == 5
        assert 0 <= r.agreement()["kendall_w"] <= 1


# ---------------------------------------------------------------- agreement
def test_kappa_identity_and_known_value():
    assert cohen_kappa([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    # sklearn.metrics.cohen_kappa_score([1,2,3,4],[1,2,4,3]) == 0.3333
    assert cohen_kappa([1, 2, 3, 4], [1, 2, 4, 3]) == pytest.approx(1 / 3)


def test_fleiss_perfect():
    R = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
    assert fleiss_kappa(R) == pytest.approx(1.0)


def test_kendall_w_extremes():
    R = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})
    assert kendall_w(R) == pytest.approx(1.0)
    R2 = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    assert kendall_w(R2) == pytest.approx(0.0)


# --------------------------------------------------------------- robustness
def test_robustness_outputs():
    r = MLmrpi().fit(select_phase(datasets.load_rac(), "Test"))
    sp = r.correlations("spearman")
    assert (sp["Test_R2"] > 0.9).all() and (sp["Test_RMSE"] < -0.9).all()
    tiers = r.tiers()
    for m in r.methods:
        assert tiers.loc[(m, "Top"), "Test_R2"] > tiers.loc[(m, "Bottom"), "Test_R2"]
    assert {"Shapiro_p", "Wilcoxon_p"} <= set(r.significance().columns)
    st = r.stability("GRA", n_iter=50)
    assert st["Same rank (%)"].between(0, 100).all()


def test_excel_and_plots(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    r = MLmrpi().fit(datasets.load_case_study_1())
    assert (tmp_path / "out.xlsx").exists() is False
    r.to_excel(tmp_path / "out.xlsx")
    assert (tmp_path / "out.xlsx").exists()
    for kind in ("scores", "ranks", "kappa", "radar"):
        r.plot(kind)


def test_cli(tmp_path, capsys):
    from mlmrpi.cli import main
    f = tmp_path / "m.csv"
    datasets.load_case_study_2().to_csv(f)
    assert main(["rank", str(f), "--phase", "Test", "--out", str(tmp_path / "r.csv")]) == 0
    assert "Best model" in capsys.readouterr().out
    assert (tmp_path / "r.csv").exists()
