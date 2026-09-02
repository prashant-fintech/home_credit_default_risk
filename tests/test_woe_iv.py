"""Tests for WoE/IV encoding — no data files or GPU required."""

import numpy as np
import pandas as pd
import pytest

from home_credit.features.woe_iv import WoEEncoder, _iv_label


# ---------------------------------------------------------------------------
# _iv_label
# ---------------------------------------------------------------------------

class TestIVLabel:
    def test_unpredictive(self):
        assert _iv_label(0.01) == "unpredictive"

    def test_weak(self):
        assert _iv_label(0.05) == "weak"

    def test_medium(self):
        assert _iv_label(0.20) == "medium"

    def test_strong(self):
        assert _iv_label(0.40) == "strong"

    def test_suspicious(self):
        assert "suspicious" in _iv_label(0.99)

    def test_boundary_at_0_02_is_weak(self):
        assert _iv_label(0.02) == "weak"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(n: int = 500, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic dataset: one highly predictive feature, one noise feature."""
    rng = np.random.default_rng(seed)
    # strong feature: higher value → higher default probability
    strong = rng.uniform(0, 1, n)
    y = (strong > 0.6).astype(int)
    # noise feature: random, uncorrelated with y
    noise = rng.standard_normal(n)
    X = pd.DataFrame({"strong": strong, "noise": noise})
    return X, pd.Series(y, name="TARGET")


# ---------------------------------------------------------------------------
# WoEEncoder.fit / iv_summary
# ---------------------------------------------------------------------------

class TestWoEEncoderFit:
    def test_fit_returns_self(self):
        X, y = _make_df()
        enc = WoEEncoder()
        assert enc.fit(X, y) is enc

    def test_iv_summary_has_all_features(self):
        X, y = _make_df()
        enc = WoEEncoder().fit(X, y)
        summary = enc.iv_summary()
        assert set(summary["feature"]) == {"strong", "noise"}

    def test_iv_summary_sorted_descending(self):
        X, y = _make_df()
        summary = WoEEncoder().fit(X, y).iv_summary()
        assert summary["iv"].is_monotonic_decreasing

    def test_strong_feature_higher_iv_than_noise(self):
        X, y = _make_df()
        summary = WoEEncoder().fit(X, y).iv_summary().set_index("feature")
        assert summary.loc["strong", "iv"] > summary.loc["noise", "iv"]

    def test_strong_feature_iv_above_medium_threshold(self):
        X, y = _make_df()
        summary = WoEEncoder().fit(X, y).iv_summary().set_index("feature")
        assert summary.loc["strong", "iv"] > 0.10  # at least medium

    def test_noise_feature_iv_near_zero(self):
        X, y = _make_df()
        summary = WoEEncoder().fit(X, y).iv_summary().set_index("feature")
        assert summary.loc["noise", "iv"] < 0.10

    def test_label_column_present(self):
        X, y = _make_df()
        summary = WoEEncoder().fit(X, y).iv_summary()
        assert "label" in summary.columns

    def test_iv_values_nonnegative(self):
        X, y = _make_df()
        summary = WoEEncoder().fit(X, y).iv_summary()
        assert (summary["iv"] >= 0).all()


# ---------------------------------------------------------------------------
# WoEEncoder.transform
# ---------------------------------------------------------------------------

class TestWoEEncoderTransform:
    def test_transform_returns_dataframe(self):
        X, y = _make_df()
        enc = WoEEncoder().fit(X, y)
        out = enc.transform(X)
        assert isinstance(out, pd.DataFrame)

    def test_transform_preserves_shape(self):
        X, y = _make_df()
        enc = WoEEncoder().fit(X, y)
        out = enc.transform(X)
        assert out.shape == X.shape

    def test_transform_preserves_column_names(self):
        X, y = _make_df()
        enc = WoEEncoder().fit(X, y)
        assert list(enc.transform(X).columns) == list(X.columns)

    def test_woe_values_are_numeric(self):
        X, y = _make_df()
        enc = WoEEncoder().fit(X, y)
        out = enc.transform(X)
        assert out.dtypes.apply(lambda d: np.issubdtype(d, np.floating)).all()

    def test_missing_values_replaced_with_missing_woe(self):
        X, y = _make_df()
        X_with_nan = X.copy()
        X_with_nan.loc[0, "strong"] = np.nan
        enc = WoEEncoder().fit(X_with_nan, y)
        out = enc.transform(X_with_nan)
        assert not pd.isna(out.loc[0, "strong"])  # NaN replaced, not propagated


# ---------------------------------------------------------------------------
# WoEEncoder.bin_table
# ---------------------------------------------------------------------------

class TestBinTable:
    def test_bin_table_has_expected_columns(self):
        X, y = _make_df()
        enc = WoEEncoder().fit(X, y)
        tbl = enc.bin_table("strong")
        for col in ["bin", "events", "non_events", "woe", "iv_bin"]:
            assert col in tbl.columns

    def test_bin_table_iv_sums_to_feature_iv(self):
        X, y = _make_df()
        enc = WoEEncoder().fit(X, y)
        tbl = enc.bin_table("strong")
        summary_iv = enc.iv_summary().set_index("feature").loc["strong", "iv"]
        assert tbl["iv_bin"].sum() == pytest.approx(summary_iv, rel=1e-5)

    def test_bin_table_raises_for_unknown_feature(self):
        X, y = _make_df()
        enc = WoEEncoder().fit(X, y)
        with pytest.raises(KeyError):
            enc.bin_table("nonexistent_feature")


# ---------------------------------------------------------------------------
# Categorical features
# ---------------------------------------------------------------------------

class TestCategoricalWoE:
    def test_categorical_feature_handled(self):
        rng = np.random.default_rng(1)
        n = 300
        cat = rng.choice(["A", "B", "C"], n)
        y = pd.Series((cat == "A").astype(int))
        X = pd.DataFrame({"cat_feat": cat})
        enc = WoEEncoder().fit(X, y)
        out = enc.transform(X)
        assert out.shape == X.shape
        # "A" should have positive WoE (more defaults than average)
        a_woe = out.loc[X["cat_feat"] == "A", "cat_feat"].iloc[0]
        c_woe = out.loc[X["cat_feat"] == "C", "cat_feat"].iloc[0]
        assert a_woe > c_woe
