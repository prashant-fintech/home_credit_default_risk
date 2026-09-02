"""Load and validate all Home Credit CSV files from data/raw/."""

from dataclasses import dataclass
from pathlib import Path

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


def _read(filename: str, schema: TableSchema, data_dir: Path, s3_uri: str | None) -> pd.DataFrame:
    if s3_uri:
        uri = f"{s3_uri.rstrip('/')}/{filename}"
        df = pd.read_csv(uri, storage_options={"anon": False})
    else:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — set S3_DATA_URI or download the dataset from Kaggle"
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


def load(data_dir: Path | None = None, s3_uri: str | None = None) -> RawDataset:
    """Load all tables from local disk or S3.

    S3 takes precedence when s3_uri is given or S3_DATA_URI is set in the environment.
    Example S3 URI: s3://home-credit-default-risk-405894863747/raw
    """
    d = data_dir or settings.data_dir
    s = s3_uri or settings.s3_data_uri

    def r(filename: str, schema: TableSchema) -> pd.DataFrame:
        return _read(filename, schema, d, s)

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
