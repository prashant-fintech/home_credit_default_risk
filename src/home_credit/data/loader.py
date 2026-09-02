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


def _read(filename: str, schema: TableSchema, data_dir: Path) -> pd.DataFrame:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — download the dataset from Kaggle first")
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


def load(data_dir: Path | None = None) -> RawDataset:
    d = data_dir or settings.data_dir
    return RawDataset(
        application_train=_read("application_train.csv", APPLICATION, d),
        application_test=_read("application_test.csv", APPLICATION_TEST, d),
        bureau=_read("bureau.csv", BUREAU, d),
        bureau_balance=_read("bureau_balance.csv", BUREAU_BALANCE, d),
        previous_application=_read("previous_application.csv", PREVIOUS_APPLICATION, d),
        pos_cash=_read("POS_CASH_balance.csv", POS_CASH, d),
        installments=_read("installments_payments.csv", INSTALLMENTS, d),
        credit_card=_read("credit_card_balance.csv", CREDIT_CARD, d),
    )
