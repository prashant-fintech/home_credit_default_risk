"""Tests for schema validation — no data files required."""

import pandas as pd
import pytest

from home_credit.data.schema import APPLICATION, BUREAU, TableSchema, validate


def test_validate_passes_when_all_columns_present():
    df = pd.DataFrame(columns=APPLICATION.required_columns)
    validate(df, APPLICATION)  # must not raise


def test_validate_raises_on_missing_column():
    cols = [c for c in APPLICATION.required_columns if c != "SK_ID_CURR"]
    df = pd.DataFrame(columns=cols)
    with pytest.raises(ValueError, match="SK_ID_CURR"):
        validate(df, APPLICATION)


def test_validate_raises_listing_all_missing():
    df = pd.DataFrame(columns=["SK_ID_CURR"])
    with pytest.raises(ValueError) as exc_info:
        validate(df, BUREAU)
    assert "SK_ID_BUREAU" in str(exc_info.value)


def test_application_target_column():
    assert APPLICATION.target_column == "TARGET"


def test_application_test_has_no_target():
    from home_credit.data.schema import APPLICATION_TEST
    assert APPLICATION_TEST.target_column is None
    assert "TARGET" not in APPLICATION_TEST.required_columns
