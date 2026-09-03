"""Tests for remote-URI handling in the loader — no network or files needed."""

import pytest

from home_credit.data.loader import storage_options


def test_s3_uses_ambient_credentials():
    assert storage_options("s3://home-credit-default-risk-405894863747/raw") == {"anon": False}


@pytest.mark.parametrize("scheme", ["abfss", "abfs", "az"])
def test_azure_extracts_account_name(scheme):
    uri = f"{scheme}://home-credit@sthomecreditb2a695a0.dfs.core.windows.net/raw"
    assert storage_options(uri) == {"account_name": "sthomecreditb2a695a0", "anon": False}


def test_azure_without_account_raises():
    with pytest.raises(ValueError, match="storage account"):
        storage_options("abfss://home-credit@/raw")


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="unsupported"):
        storage_options("gs://bucket/raw")
