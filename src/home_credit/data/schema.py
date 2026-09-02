"""Expected columns and dtypes for every Home Credit CSV file.

Used for early validation so a corrupt or truncated download fails loudly
at load time rather than silently producing wrong features.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class TableSchema:
    name: str
    required_columns: list[str]
    target_column: str | None = None


APPLICATION = TableSchema(
    name="application",
    required_columns=[
        "SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER",
        "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN", "AMT_INCOME_TOTAL",
        "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
        "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    ],
    target_column="TARGET",
)

APPLICATION_TEST = TableSchema(
    name="application_test",
    required_columns=[c for c in APPLICATION.required_columns if c != "TARGET"],
)

BUREAU = TableSchema(
    name="bureau",
    required_columns=[
        "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_CURRENCY",
        "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT_ENDDATE",
        "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE",
        "CREDIT_TYPE", "DAYS_CREDIT_UPDATE", "AMT_ANNUITY",
    ],
)

BUREAU_BALANCE = TableSchema(
    name="bureau_balance",
    required_columns=["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
)

PREVIOUS_APPLICATION = TableSchema(
    name="previous_application",
    required_columns=[
        "SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_TYPE", "AMT_ANNUITY",
        "AMT_APPLICATION", "AMT_CREDIT", "AMT_DOWN_PAYMENT", "AMT_GOODS_PRICE",
        "NAME_CONTRACT_STATUS", "DAYS_DECISION", "NAME_PAYMENT_TYPE",
        "CODE_REJECT_REASON", "NAME_CLIENT_TYPE", "NAME_GOODS_CATEGORY",
        "NAME_PORTFOLIO", "NAME_PRODUCT_TYPE", "CHANNEL_TYPE",
        "SELLERPLACE_AREA", "NAME_SELLER_INDUSTRY", "CNT_PAYMENT",
        "NAME_YIELD_GROUP", "PRODUCT_COMBINATION", "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE",
        "DAYS_TERMINATION", "NFLAG_INSURED_ON_APPROVAL",
    ],
)

POS_CASH = TableSchema(
    name="POS_CASH_balance",
    required_columns=[
        "SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "CNT_INSTALMENT",
        "CNT_INSTALMENT_FUTURE", "NAME_CONTRACT_STATUS", "SK_DPD", "SK_DPD_DEF",
    ],
)

INSTALLMENTS = TableSchema(
    name="installments_payments",
    required_columns=[
        "SK_ID_PREV", "SK_ID_CURR", "NUM_INSTALMENT_VERSION",
        "NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT", "AMT_PAYMENT",
    ],
)

CREDIT_CARD = TableSchema(
    name="credit_card_balance",
    required_columns=[
        "SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_ATM_CURRENT",
        "AMT_DRAWINGS_CURRENT", "AMT_DRAWINGS_OTHER_CURRENT",
        "AMT_DRAWINGS_POS_CURRENT", "AMT_INST_MIN_REGULARITY",
        "AMT_PAYMENT_CURRENT", "AMT_PAYMENT_TOTAL_CURRENT",
        "AMT_RECEIVABLE_PRINCIPAL", "AMT_RECIVABLE", "AMT_TOTAL_RECEIVABLE",
        "CNT_DRAWINGS_ATM_CURRENT", "CNT_DRAWINGS_CURRENT",
        "CNT_DRAWINGS_OTHER_CURRENT", "CNT_DRAWINGS_POS_CURRENT",
        "CNT_INSTALMENT_MATURE_CUM", "NAME_CONTRACT_STATUS", "SK_DPD",
        "SK_DPD_DEF",
    ],
)

ALL_SCHEMAS = [
    APPLICATION, APPLICATION_TEST, BUREAU, BUREAU_BALANCE,
    PREVIOUS_APPLICATION, POS_CASH, INSTALLMENTS, CREDIT_CARD,
]


def validate(df: pd.DataFrame, schema: TableSchema) -> None:
    missing = [c for c in schema.required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"[{schema.name}] missing columns: {missing}")
