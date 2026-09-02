"""Tests for feature engineering — synthetic DataFrames, no CSV files needed."""

import numpy as np
import pandas as pd
import pytest

from home_credit.features.pipeline import (
    _application_features,
    _bureau_features,
    _installments_features,
    _pos_cash_features,
    _previous_application_features,
)


def _app_row(**overrides) -> dict:
    base = {
        "SK_ID_CURR": 1,
        "TARGET": 0,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": 0,
        "AMT_INCOME_TOTAL": 100_000.0,
        "AMT_CREDIT": 200_000.0,
        "AMT_ANNUITY": 10_000.0,
        "AMT_GOODS_PRICE": 180_000.0,
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Single / not married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "DAYS_BIRTH": -10000,
        "DAYS_EMPLOYED": -2000,
        "DAYS_REGISTRATION": -3000,
        "DAYS_ID_PUBLISH": -1000,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.6,
        "EXT_SOURCE_3": 0.7,
    }
    base.update(overrides)
    return base


class TestApplicationFeatures:
    def _df(self, **overrides) -> pd.DataFrame:
        return pd.DataFrame([_app_row(**overrides)])

    def test_credit_income_ratio(self):
        df = self._df(AMT_INCOME_TOTAL=100_000, AMT_CREDIT=200_000)
        out = _application_features(df)
        assert out.loc[1, "CREDIT_INCOME_RATIO"] == pytest.approx(2.0)

    def test_annuity_income_ratio(self):
        df = self._df(AMT_INCOME_TOTAL=100_000, AMT_ANNUITY=10_000)
        out = _application_features(df)
        assert out.loc[1, "ANNUITY_INCOME_RATIO"] == pytest.approx(0.1)

    def test_not_employed_sentinel_becomes_nan(self):
        df = self._df(DAYS_EMPLOYED=365243)
        out = _application_features(df)
        assert np.isnan(out.loc[1, "EMPLOYED_YEARS"])

    def test_age_years_positive(self):
        df = self._df(DAYS_BIRTH=-10000)
        out = _application_features(df)
        assert out.loc[1, "AGE_YEARS"] == pytest.approx(10000 / 365.25, rel=1e-4)

    def test_ext_source_mean(self):
        df = self._df(EXT_SOURCE_1=0.4, EXT_SOURCE_2=0.6, EXT_SOURCE_3=0.8)
        out = _application_features(df)
        assert out.loc[1, "EXT_SOURCE_MEAN"] == pytest.approx(0.6)

    def test_index_is_sk_id_curr(self):
        df = self._df(SK_ID_CURR=42)
        out = _application_features(df)
        assert out.index.name == "SK_ID_CURR"
        assert 42 in out.index

    def test_zero_income_gives_nan_ratio(self):
        df = self._df(AMT_INCOME_TOTAL=0)
        out = _application_features(df)
        assert np.isnan(out.loc[1, "CREDIT_INCOME_RATIO"])


class TestBureauFeatures:
    def _bureau(self) -> pd.DataFrame:
        return pd.DataFrame({
            "SK_ID_CURR": [1, 1, 2],
            "SK_ID_BUREAU": [10, 11, 12],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
            "CREDIT_CURRENCY": ["currency 1"] * 3,
            "DAYS_CREDIT": [-100, -200, -50],
            "CREDIT_DAY_OVERDUE": [0, 5, 10],
            "DAYS_CREDIT_ENDDATE": [100, 0, 200],
            "AMT_CREDIT_MAX_OVERDUE": [0.0, 100.0, 0.0],
            "CNT_CREDIT_PROLONG": [0, 1, 0],
            "AMT_CREDIT_SUM": [50_000.0, 30_000.0, 20_000.0],
            "AMT_CREDIT_SUM_DEBT": [10_000.0, 0.0, 5_000.0],
            "AMT_CREDIT_SUM_LIMIT": [0.0, 0.0, 0.0],
            "AMT_CREDIT_SUM_OVERDUE": [0.0, 500.0, 0.0],
            "CREDIT_TYPE": ["Consumer credit"] * 3,
            "DAYS_CREDIT_UPDATE": [-10, -20, -5],
            "AMT_ANNUITY": [1000.0, 800.0, 500.0],
        })

    def _bureau_balance(self) -> pd.DataFrame:
        return pd.DataFrame({
            "SK_ID_BUREAU": [10, 10, 11],
            "MONTHS_BALANCE": [-1, -2, -1],
            "STATUS": ["C", "0", "C"],
        })

    def test_loan_count_per_applicant(self):
        out = _bureau_features(self._bureau(), self._bureau_balance())
        assert out.loc[1, "BUREAU_LOAN_COUNT"] == 2
        assert out.loc[2, "BUREAU_LOAN_COUNT"] == 1

    def test_active_count(self):
        out = _bureau_features(self._bureau(), self._bureau_balance())
        assert out.loc[1, "BUREAU_ACTIVE_COUNT"] == 1

    def test_debt_credit_ratio(self):
        out = _bureau_features(self._bureau(), self._bureau_balance())
        assert out.loc[1, "BUREAU_DEBT_CREDIT_RATIO"] == pytest.approx(10_000 / 80_000)


class TestInstallmentsFeatures:
    def _inst(self) -> pd.DataFrame:
        return pd.DataFrame({
            "SK_ID_PREV": [1, 1, 2],
            "SK_ID_CURR": [10, 10, 10],
            "NUM_INSTALMENT_VERSION": [1, 1, 1],
            "NUM_INSTALMENT_NUMBER": [1, 2, 1],
            "DAYS_INSTALMENT": [-30, -60, -30],
            "DAYS_ENTRY_PAYMENT": [-28, -62, -35],
            "AMT_INSTALMENT": [1000.0, 1000.0, 500.0],
            "AMT_PAYMENT": [1000.0, 950.0, 500.0],
        })

    def test_installment_count(self):
        out = _installments_features(self._inst())
        assert out.loc[10, "INST_COUNT"] == 3

    def test_late_payment_count(self):
        # DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT:
        #   row 1: -28 - (-30) = +2  → late
        #   row 2: -62 - (-60) = -2  → early
        #   row 3: -35 - (-30) = -5  → early
        out = _installments_features(self._inst())
        assert out.loc[10, "INST_LATE_PAYMENT_COUNT"] == 1

    def test_payment_diff_mean(self):
        out = _installments_features(self._inst())
        # diffs: 0, 50, 0  → mean = 50/3
        assert out.loc[10, "INST_PAYMENT_DIFF_MEAN"] == pytest.approx(50 / 3, rel=1e-4)


class TestEvaluateMetrics:
    def test_gini_perfect_classifier(self):
        from home_credit.models.evaluate import gini
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        assert gini(y, scores) == pytest.approx(1.0)

    def test_gini_random_classifier(self):
        from home_credit.models.evaluate import gini
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 1000)
        scores = rng.random(1000)
        assert abs(gini(y, scores)) < 0.1

    def test_ks_perfect_separation(self):
        from home_credit.models.evaluate import ks_statistic
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        assert ks_statistic(y, scores) == pytest.approx(1.0)
