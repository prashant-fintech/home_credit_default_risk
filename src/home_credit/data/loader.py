"""Load and validate all Home Credit CSV files from local disk, S3, or ADLS Gen2."""

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from home_credit.config import settings
from home_credit.data.schema import (
    APPLICATION,
    APPLICATION_TEST,
    BUREAU,
    BUREAU_BALANCE,
    CREDIT_CARD,
    INSTALLMENTS,
    POS_CASH,
    PREVIOUS_APPLICATION,
    TableSchema,
    validate,
)

AZURE_SCHEMES = {"abfs", "abfss", "az"}


def storage_options(uri: str) -> dict[str, object]:
    """fsspec options for a remote data URI.

    - s3://bucket/prefix                                   -> ambient AWS credentials (s3fs)
    - abfss://container@account.dfs.core.windows.net/path  -> ambient Azure login via
      DefaultAzureCredential (adlfs); run `az login` first
    """
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return {"anon": False}
    if parsed.scheme in AZURE_SCHEMES:
        # netloc is "<container>@<account>.dfs.core.windows.net"
        account = parsed.netloc.split("@")[-1].split(".")[0]
        if not account:
            raise ValueError(f"cannot parse storage account from {uri}")
        return {"account_name": account, "anon": False}
    raise ValueError(f"unsupported remote URI scheme: {uri}")


def _read(
    filename: str, schema: TableSchema, data_dir: Path, remote_uri: str | None
) -> pd.DataFrame:
    if remote_uri:
        uri = f"{remote_uri.rstrip('/')}/{filename}"
        df = pd.read_csv(uri, storage_options=storage_options(remote_uri))
    else:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — set AZURE_DATA_URI / S3_DATA_URI or download the "
                "dataset from Kaggle"
            )
        df = pd.read_csv(path)
    validate(df, schema)
    return df


@dataclass
class RawDataset:
    application_train: pd.DataFrame
    application_test: pd.DataFrame
    bureau: pd.DataFrame
    bureau_balance: pd.DataFrame
    previous_application: pd.DataFrame
    pos_cash: pd.DataFrame
    installments: pd.DataFrame
    credit_card: pd.DataFrame

    @property
    def target(self) -> pd.Series:
        return self.application_train["TARGET"]

    @property
    def train_ids(self) -> pd.Series:
        return self.application_train["SK_ID_CURR"]

    @property
    def test_ids(self) -> pd.Series:
        return self.application_test["SK_ID_CURR"]


def load(
    data_dir: Path | None = None,
    s3_uri: str | None = None,
    remote_uri: str | None = None,
) -> RawDataset:
    """Load all tables from local disk, S3, or ADLS Gen2.

    Precedence: explicit remote_uri / s3_uri argument, then AZURE_DATA_URI, then
    S3_DATA_URI from the environment, then local data_dir.
    Example URIs:
        s3://home-credit-default-risk-405894863747/raw
        abfss://home-credit@sthomecreditXXXX.dfs.core.windows.net/raw
    """
    d = data_dir or settings.data_dir
    remote = remote_uri or s3_uri or settings.azure_data_uri or settings.s3_data_uri

    def r(filename: str, schema: TableSchema) -> pd.DataFrame:
        return _read(filename, schema, d, remote)

    return RawDataset(
        application_train=r("application_train.csv", APPLICATION),
        application_test=r("application_test.csv", APPLICATION_TEST),
        bureau=r("bureau.csv", BUREAU),
        bureau_balance=r("bureau_balance.csv", BUREAU_BALANCE),
        previous_application=r("previous_application.csv", PREVIOUS_APPLICATION),
        pos_cash=r("POS_CASH_balance.csv", POS_CASH),
        installments=r("installments_payments.csv", INSTALLMENTS),
        credit_card=r("credit_card_balance.csv", CREDIT_CARD),
    )
