"""Tests for SHAP explainer — uses a tiny LightGBM model, no real data needed."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from home_credit.explain import shap_explainer

# ---------------------------------------------------------------------------
# Fixture: train a tiny LightGBM model so we have a real booster
# ---------------------------------------------------------------------------

def _tiny_booster() -> lgb.Booster:
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({
        "age": rng.uniform(18, 70, n),
        "income": rng.uniform(10_000, 200_000, n),
        "loan_amount": rng.uniform(5_000, 500_000, n),
    })
    y = (X["loan_amount"] / X["income"] > 3).astype(int)
    ds = lgb.Dataset(X, label=y)
    params = {"objective": "binary", "n_estimators": 20, "num_leaves": 8, "verbose": -1}
    return lgb.train(params, ds, num_boost_round=20)


@pytest.fixture(scope="module")
def booster():
    return _tiny_booster()


@pytest.fixture(scope="module")
def X_sample():
    rng = np.random.default_rng(7)
    n = 100
    return pd.DataFrame({
        "age": rng.uniform(18, 70, n),
        "income": rng.uniform(10_000, 200_000, n),
        "loan_amount": rng.uniform(5_000, 500_000, n),
    })


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

class TestCompute:
    def test_returns_array(self, booster, X_sample):
        sv = shap_explainer.compute([booster], X_sample, n_sample=50)
        assert isinstance(sv, np.ndarray)

    def test_shape_is_rows_by_features(self, booster, X_sample):
        n_sample = 50
        sv = shap_explainer.compute([booster], X_sample, n_sample=n_sample)
        assert sv.shape == (n_sample, X_sample.shape[1])

    def test_respects_n_sample_cap(self, booster, X_sample):
        sv = shap_explainer.compute([booster], X_sample, n_sample=30)
        assert sv.shape[0] == 30

    def test_n_sample_larger_than_data_uses_all_rows(self, booster, X_sample):
        sv = shap_explainer.compute([booster], X_sample, n_sample=10_000)
        assert sv.shape[0] == len(X_sample)

    def test_averages_across_multiple_models(self, booster, X_sample):
        sv_single = shap_explainer.compute([booster], X_sample, n_sample=50)
        sv_double = shap_explainer.compute([booster, booster], X_sample, n_sample=50)
        # Averaging identical models gives the same result
        np.testing.assert_allclose(sv_single, sv_double, rtol=1e-5)

    def test_shap_values_are_finite(self, booster, X_sample):
        sv = shap_explainer.compute([booster], X_sample, n_sample=50)
        assert np.isfinite(sv).all()


# ---------------------------------------------------------------------------
# importance_df()
# ---------------------------------------------------------------------------

class TestImportanceDf:
    def test_returns_dataframe(self, booster, X_sample):
        sv = shap_explainer.compute([booster], X_sample, n_sample=50)
        imp = shap_explainer.importance_df(sv, X_sample.columns.tolist())
        assert isinstance(imp, pd.DataFrame)

    def test_has_feature_and_mean_abs_shap_columns(self, booster, X_sample):
        sv = shap_explainer.compute([booster], X_sample, n_sample=50)
        imp = shap_explainer.importance_df(sv, X_sample.columns.tolist())
        assert "feature" in imp.columns
        assert "mean_abs_shap" in imp.columns

    def test_sorted_descending(self, booster, X_sample):
        sv = shap_explainer.compute([booster], X_sample, n_sample=50)
        imp = shap_explainer.importance_df(sv, X_sample.columns.tolist())
        assert imp["mean_abs_shap"].is_monotonic_decreasing

    def test_all_features_present(self, booster, X_sample):
        sv = shap_explainer.compute([booster], X_sample, n_sample=50)
        imp = shap_explainer.importance_df(sv, X_sample.columns.tolist())
        assert set(imp["feature"]) == set(X_sample.columns)

    def test_loan_amount_ratio_is_top_feature(self, booster, X_sample):
        sv = shap_explainer.compute([booster], X_sample, n_sample=len(X_sample))
        imp = shap_explainer.importance_df(sv, X_sample.columns.tolist())
        # loan_amount and income drive the label, so both should outrank age
        top2 = set(imp.head(2)["feature"])
        assert "age" not in top2

    def test_mean_abs_shap_nonnegative(self, booster, X_sample):
        sv = shap_explainer.compute([booster], X_sample, n_sample=50)
        imp = shap_explainer.importance_df(sv, X_sample.columns.tolist())
        assert (imp["mean_abs_shap"] >= 0).all()


# ---------------------------------------------------------------------------
# waterfall_data()
# ---------------------------------------------------------------------------

class TestWaterfallData:
    def test_returns_dataframe(self, booster, X_sample):
        sv = shap_explainer.compute([booster], X_sample, n_sample=50)
        wf = shap_explainer.waterfall_data(sv, X_sample.iloc[:50], idx=0)
        assert isinstance(wf, pd.DataFrame)

    def test_has_required_columns(self, booster, X_sample):
        sv = shap_explainer.compute([booster], X_sample, n_sample=50)
        wf = shap_explainer.waterfall_data(sv, X_sample.iloc[:50], idx=0)
        for col in ["feature", "value", "shap_value", "abs_shap"]:
            assert col in wf.columns

    def test_top_n_limits_rows(self, booster, X_sample):
        sv = shap_explainer.compute([booster], X_sample, n_sample=50)
        wf = shap_explainer.waterfall_data(sv, X_sample.iloc[:50], idx=0, top_n=2)
        # top_n rows + at most 1 "other features" aggregation row
        assert len(wf) <= 3

    def test_shap_values_sum_approximately_preserved(self, booster, X_sample):
        sv = shap_explainer.compute([booster], X_sample, n_sample=50)
        wf = shap_explainer.waterfall_data(sv, X_sample.iloc[:50], idx=0, top_n=3)
        total_in_wf = wf["shap_value"].sum()
        total_actual = sv[0].sum()
        assert total_in_wf == pytest.approx(total_actual, rel=1e-4)
