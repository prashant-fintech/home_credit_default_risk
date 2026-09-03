"""Feature engineering pipeline.

Aggregates all satellite tables onto SK_ID_CURR and concatenates with
application features. Returns a single wide DataFrame ready for LightGBM.

Design principles:
- Every aggregation is a pure function: (DataFrame) -> DataFrame with SK_ID_CURR index.
- The pipeline function combines them deterministically and caches to parquet.
- Categorical columns are left as object dtype; LightGBM handles them natively
  when passed as the `categorical_feature` argument.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from home_credit.config import settings
from home_credit.data.loader import RawDataset

# ---------------------------------------------------------------------------
# Application table — main features
# ---------------------------------------------------------------------------

def _application_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Domain-informed ratios
    out["CREDIT_INCOME_RATIO"] = out["AMT_CREDIT"] / out["AMT_INCOME_TOTAL"].replace(0, np.nan)
    out["ANNUITY_INCOME_RATIO"] = out["AMT_ANNUITY"] / out["AMT_INCOME_TOTAL"].replace(0, np.nan)
    out["CREDIT_GOODS_RATIO"] = out["AMT_CREDIT"] / out["AMT_GOODS_PRICE"].replace(0, np.nan)

    # Age in years (DAYS_BIRTH is negative)
    out["AGE_YEARS"] = -out["DAYS_BIRTH"] / 365.25

    # Employment ratio — DAYS_EMPLOYED = 365243 encodes "not employed"
    out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].replace(365243, np.nan)
    out["EMPLOYED_YEARS"] = -out["DAYS_EMPLOYED"] / 365.25
    out["EMPLOYMENT_AGE_RATIO"] = out["EMPLOYED_YEARS"] / out["AGE_YEARS"].replace(0, np.nan)

    # External sources composite
    ext = out[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    out["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    out["EXT_SOURCE_MIN"] = ext.min(axis=1)
    out["EXT_SOURCE_PROD"] = ext.prod(axis=1)

    # Document flags count
    doc_cols = [c for c in out.columns if c.startswith("FLAG_DOCUMENT_")]
    out["FLAG_DOCUMENT_SUM"] = out[doc_cols].sum(axis=1)

    return out.set_index("SK_ID_CURR")


# ---------------------------------------------------------------------------
# Bureau — credit history from external agencies
# ---------------------------------------------------------------------------

def _bureau_features(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    # Aggregate bureau_balance per bureau entry
    bb_agg = bureau_balance.groupby("SK_ID_BUREAU")["STATUS"].agg(
        BB_STATUS_C_COUNT=lambda x: (x == "C").sum(),
        BB_STATUS_X_COUNT=lambda x: (x == "X").sum(),
        BB_MONTHS_COUNT="count",
    )

    df = bureau.join(bb_agg, on="SK_ID_BUREAU", how="left")

    agg = df.groupby("SK_ID_CURR").agg(
        BUREAU_LOAN_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
        BUREAU_CLOSED_COUNT=("CREDIT_ACTIVE", lambda x: (x == "Closed").sum()),
        BUREAU_AMT_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
        BUREAU_AMT_CREDIT_SUM_DEBT=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_AMT_CREDIT_SUM_OVERDUE=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
        BUREAU_CREDIT_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_CREDIT_DAY_OVERDUE_MEAN=("CREDIT_DAY_OVERDUE", "mean"),
        BUREAU_BB_STATUS_C_SUM=("BB_STATUS_C_COUNT", "sum"),
        BUREAU_BB_MONTHS_MEAN=("BB_MONTHS_COUNT", "mean"),
    )
    agg["BUREAU_DEBT_CREDIT_RATIO"] = (
        agg["BUREAU_AMT_CREDIT_SUM_DEBT"] / agg["BUREAU_AMT_CREDIT_SUM"].replace(0, np.nan)
    )
    return agg


# ---------------------------------------------------------------------------
# Previous applications
# ---------------------------------------------------------------------------

def _previous_application_features(prev: pd.DataFrame) -> pd.DataFrame:
    agg = prev.groupby("SK_ID_CURR").agg(
        PREV_COUNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_COUNT=("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
        PREV_REFUSED_COUNT=("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_AMT_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
        PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PREV_AMT_DOWN_PAYMENT_MEAN=("AMT_DOWN_PAYMENT", "mean"),
        PREV_DAYS_DECISION_MEAN=("DAYS_DECISION", "mean"),
        PREV_CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
        PREV_RATE_DOWN_PAYMENT_MEAN=("RATE_DOWN_PAYMENT", "mean"),
    ).rename(columns={"RATE_DOWN_PAYMENT": "PREV_RATE_DOWN_PAYMENT_MEAN"})

    agg["PREV_APPROVAL_RATE"] = (
        agg["PREV_APPROVED_COUNT"] / agg["PREV_COUNT"].replace(0, np.nan)
    )
    agg["PREV_CREDIT_APP_RATIO"] = (
        agg["PREV_AMT_CREDIT_MEAN"] / agg["PREV_AMT_APPLICATION_MEAN"].replace(0, np.nan)
    )
    return agg


# ---------------------------------------------------------------------------
# POS CASH balances
# ---------------------------------------------------------------------------

def _pos_cash_features(pos: pd.DataFrame) -> pd.DataFrame:
    return pos.groupby("SK_ID_CURR").agg(
        POS_COUNT=("SK_ID_PREV", "count"),
        POS_MONTHS_BALANCE_MEAN=("MONTHS_BALANCE", "mean"),
        POS_SK_DPD_MEAN=("SK_DPD", "mean"),
        POS_SK_DPD_MAX=("SK_DPD", "max"),
        POS_SK_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
        POS_SK_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
        POS_NAME_CONTRACT_STATUS_ACTIVE=(
            "NAME_CONTRACT_STATUS", lambda x: (x == "Active").sum()
        ),
    )


# ---------------------------------------------------------------------------
# Installment payments
# ---------------------------------------------------------------------------

def _installments_features(inst: pd.DataFrame) -> pd.DataFrame:
    inst = inst.copy()
    inst["PAYMENT_DIFF"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]
    inst["PAYMENT_RATIO"] = inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"].replace(0, np.nan)
    inst["DAYS_PAYMENT_DIFF"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]

    return inst.groupby("SK_ID_CURR").agg(
        INST_COUNT=("SK_ID_PREV", "count"),
        INST_AMT_PAYMENT_SUM=("AMT_PAYMENT", "sum"),
        INST_AMT_INSTALMENT_SUM=("AMT_INSTALMENT", "sum"),
        INST_PAYMENT_DIFF_MEAN=("PAYMENT_DIFF", "mean"),
        INST_PAYMENT_DIFF_MAX=("PAYMENT_DIFF", "max"),
        INST_PAYMENT_RATIO_MEAN=("PAYMENT_RATIO", "mean"),
        INST_DAYS_PAYMENT_DIFF_MEAN=("DAYS_PAYMENT_DIFF", "mean"),
        INST_DAYS_PAYMENT_DIFF_MAX=("DAYS_PAYMENT_DIFF", "max"),
        INST_LATE_PAYMENT_COUNT=("DAYS_PAYMENT_DIFF", lambda x: (x > 0).sum()),
    )


# ---------------------------------------------------------------------------
# Credit card balances
# ---------------------------------------------------------------------------

def _credit_card_features(cc: pd.DataFrame) -> pd.DataFrame:
    cc = cc.copy()
    cc["UTILIZATION"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)

    return cc.groupby("SK_ID_CURR").agg(
        CC_COUNT=("SK_ID_PREV", "count"),
        CC_AMT_BALANCE_MEAN=("AMT_BALANCE", "mean"),
        CC_AMT_BALANCE_MAX=("AMT_BALANCE", "max"),
        CC_UTILIZATION_MEAN=("UTILIZATION", "mean"),
        CC_UTILIZATION_MAX=("UTILIZATION", "max"),
        CC_SK_DPD_MEAN=("SK_DPD", "mean"),
        CC_SK_DPD_MAX=("SK_DPD", "max"),
        CC_AMT_DRAWINGS_CURRENT_MEAN=("AMT_DRAWINGS_CURRENT", "mean"),
        CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN=("AMT_PAYMENT_TOTAL_CURRENT", "mean"),
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build(dataset: RawDataset, processed_dir: Path | None = None) -> pd.DataFrame:
    """Combine all feature tables into a single wide DataFrame indexed by SK_ID_CURR.

    Writes the result to parquet for fast re-use. Pass the same processed_dir
    to load instead of recomputing.
    """
    out_dir = processed_dir or settings.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "features.parquet"

    if cache.exists():
        return pd.read_parquet(cache)

    app = _application_features(
        pd.concat([dataset.application_train, dataset.application_test], ignore_index=True)
    )
    bureau = _bureau_features(dataset.bureau, dataset.bureau_balance)
    prev = _previous_application_features(dataset.previous_application)
    pos = _pos_cash_features(dataset.pos_cash)
    inst = _installments_features(dataset.installments)
    cc = _credit_card_features(dataset.credit_card)

    features = app.join(bureau, how="left") \
                  .join(prev, how="left") \
                  .join(pos, how="left") \
                  .join(inst, how="left") \
                  .join(cc, how="left")

    features.to_parquet(cache)
    return features
