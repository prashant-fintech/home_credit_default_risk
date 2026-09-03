# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold — feature table (PySpark port of `home_credit/features/pipeline.py`)
# MAGIC
# MAGIC Same aggregations as the local pandas pipeline, expressed in Spark so they scale out.
# MAGIC The output is one wide row per `SK_ID_CURR` (train + test), registered as a
# MAGIC **Unity Catalog feature table** by declaring a primary key.
# MAGIC
# MAGIC **Concepts (exam topics)**
# MAGIC - `groupBy().agg()` with conditional counts via `sum(when(...))`.
# MAGIC - `avg` ignores nulls (like pandas `mean`); `sum` of all-null is null in Spark but 0 in pandas,
# MAGIC   so sums are wrapped in `coalesce(..., 0)` to keep parity with the local features.
# MAGIC - A Delta table with a `PRIMARY KEY` constraint in UC is a feature table: Feature Engineering
# MAGIC   in UC can look it up by key at training and inference time.

# COMMAND ----------

from functools import reduce
from operator import add, mul

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "learningdatabricks")
catalog = dbutils.widgets.get("catalog")
silver = f"{catalog}.silver"
gold = f"{catalog}.gold"


def nz(col: str):
    """Treat 0 as null so ratios do not divide by zero (pandas: .replace(0, np.nan))."""
    return F.when(F.col(col) == 0, None).otherwise(F.col(col))


def sum0(col: str):
    return F.coalesce(F.sum(col), F.lit(0.0))


def count_eq(col: str, value):
    return F.coalesce(F.sum(F.when(F.col(col) == value, 1).otherwise(0)), F.lit(0))

# COMMAND ----------

# MAGIC %md ## Application

# COMMAND ----------


def application_features(df: DataFrame) -> DataFrame:
    ext = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    ext_nonnull = reduce(add, [F.when(F.col(c).isNotNull(), 1).otherwise(0) for c in ext])
    ext_sum = reduce(add, [F.coalesce(F.col(c), F.lit(0.0)) for c in ext])
    ext_prod = reduce(mul, [F.coalesce(F.col(c), F.lit(1.0)) for c in ext])
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]

    return (
        df.withColumn("CREDIT_INCOME_RATIO", F.col("AMT_CREDIT") / nz("AMT_INCOME_TOTAL"))
        .withColumn("ANNUITY_INCOME_RATIO", F.col("AMT_ANNUITY") / nz("AMT_INCOME_TOTAL"))
        .withColumn("CREDIT_GOODS_RATIO", F.col("AMT_CREDIT") / nz("AMT_GOODS_PRICE"))
        .withColumn("AGE_YEARS", -F.col("DAYS_BIRTH") / 365.25)
        # DAYS_EMPLOYED sentinel was already nulled in silver
        .withColumn("EMPLOYED_YEARS", -F.col("DAYS_EMPLOYED") / 365.25)
        .withColumn("EMPLOYMENT_AGE_RATIO", F.col("EMPLOYED_YEARS") / nz("AGE_YEARS"))
        .withColumn("EXT_SOURCE_MEAN", F.when(ext_nonnull > 0, ext_sum / ext_nonnull))
        .withColumn("EXT_SOURCE_MIN", F.least(*[F.col(c) for c in ext]))
        .withColumn("EXT_SOURCE_PROD", F.when(ext_nonnull > 0, ext_prod))
        .withColumn(
            "FLAG_DOCUMENT_SUM",
            reduce(add, [F.coalesce(F.col(c), F.lit(0)) for c in doc_cols]),
        )
    )

# COMMAND ----------

# MAGIC %md ## Bureau + bureau_balance

# COMMAND ----------


def bureau_features(bureau: DataFrame, bureau_balance: DataFrame) -> DataFrame:
    bb = bureau_balance.groupBy("SK_ID_BUREAU").agg(
        count_eq("STATUS", "C").alias("BB_STATUS_C_COUNT"),
        count_eq("STATUS", "X").alias("BB_STATUS_X_COUNT"),
        F.count("STATUS").alias("BB_MONTHS_COUNT"),
    )
    df = bureau.join(bb, "SK_ID_BUREAU", "left")
    agg = df.groupBy("SK_ID_CURR").agg(
        F.count("SK_ID_BUREAU").alias("BUREAU_LOAN_COUNT"),
        count_eq("CREDIT_ACTIVE", "Active").alias("BUREAU_ACTIVE_COUNT"),
        count_eq("CREDIT_ACTIVE", "Closed").alias("BUREAU_CLOSED_COUNT"),
        sum0("AMT_CREDIT_SUM").alias("BUREAU_AMT_CREDIT_SUM"),
        sum0("AMT_CREDIT_SUM_DEBT").alias("BUREAU_AMT_CREDIT_SUM_DEBT"),
        sum0("AMT_CREDIT_SUM_OVERDUE").alias("BUREAU_AMT_CREDIT_SUM_OVERDUE"),
        F.avg("DAYS_CREDIT").alias("BUREAU_DAYS_CREDIT_MEAN"),
        F.max("CREDIT_DAY_OVERDUE").alias("BUREAU_CREDIT_DAY_OVERDUE_MAX"),
        F.avg("CREDIT_DAY_OVERDUE").alias("BUREAU_CREDIT_DAY_OVERDUE_MEAN"),
        sum0("BB_STATUS_C_COUNT").alias("BUREAU_BB_STATUS_C_SUM"),
        F.avg("BB_MONTHS_COUNT").alias("BUREAU_BB_MONTHS_MEAN"),
    )
    return agg.withColumn(
        "BUREAU_DEBT_CREDIT_RATIO",
        F.col("BUREAU_AMT_CREDIT_SUM_DEBT") / nz("BUREAU_AMT_CREDIT_SUM"),
    )

# COMMAND ----------

# MAGIC %md ## Previous applications, POS cash, installments, credit card

# COMMAND ----------


def previous_application_features(prev: DataFrame) -> DataFrame:
    agg = prev.groupBy("SK_ID_CURR").agg(
        F.count("SK_ID_PREV").alias("PREV_COUNT"),
        count_eq("NAME_CONTRACT_STATUS", "Approved").alias("PREV_APPROVED_COUNT"),
        count_eq("NAME_CONTRACT_STATUS", "Refused").alias("PREV_REFUSED_COUNT"),
        F.avg("AMT_CREDIT").alias("PREV_AMT_CREDIT_MEAN"),
        F.avg("AMT_ANNUITY").alias("PREV_AMT_ANNUITY_MEAN"),
        F.avg("AMT_APPLICATION").alias("PREV_AMT_APPLICATION_MEAN"),
        F.avg("AMT_DOWN_PAYMENT").alias("PREV_AMT_DOWN_PAYMENT_MEAN"),
        F.avg("DAYS_DECISION").alias("PREV_DAYS_DECISION_MEAN"),
        F.avg("CNT_PAYMENT").alias("PREV_CNT_PAYMENT_MEAN"),
        F.avg("RATE_DOWN_PAYMENT").alias("PREV_RATE_DOWN_PAYMENT_MEAN"),
    )
    return agg.withColumn(
        "PREV_APPROVAL_RATE", F.col("PREV_APPROVED_COUNT") / nz("PREV_COUNT")
    ).withColumn(
        "PREV_CREDIT_APP_RATIO",
        F.col("PREV_AMT_CREDIT_MEAN") / nz("PREV_AMT_APPLICATION_MEAN"),
    )


def pos_cash_features(pos: DataFrame) -> DataFrame:
    return pos.groupBy("SK_ID_CURR").agg(
        F.count("SK_ID_PREV").alias("POS_COUNT"),
        F.avg("MONTHS_BALANCE").alias("POS_MONTHS_BALANCE_MEAN"),
        F.avg("SK_DPD").alias("POS_SK_DPD_MEAN"),
        F.max("SK_DPD").alias("POS_SK_DPD_MAX"),
        F.avg("SK_DPD_DEF").alias("POS_SK_DPD_DEF_MEAN"),
        F.max("SK_DPD_DEF").alias("POS_SK_DPD_DEF_MAX"),
        count_eq("NAME_CONTRACT_STATUS", "Active").alias("POS_NAME_CONTRACT_STATUS_ACTIVE"),
    )


def installments_features(inst: DataFrame) -> DataFrame:
    inst = (
        inst.withColumn("PAYMENT_DIFF", F.col("AMT_INSTALMENT") - F.col("AMT_PAYMENT"))
        .withColumn("PAYMENT_RATIO", F.col("AMT_PAYMENT") / nz("AMT_INSTALMENT"))
        .withColumn("DAYS_PAYMENT_DIFF", F.col("DAYS_ENTRY_PAYMENT") - F.col("DAYS_INSTALMENT"))
    )
    return inst.groupBy("SK_ID_CURR").agg(
        F.count("SK_ID_PREV").alias("INST_COUNT"),
        sum0("AMT_PAYMENT").alias("INST_AMT_PAYMENT_SUM"),
        sum0("AMT_INSTALMENT").alias("INST_AMT_INSTALMENT_SUM"),
        F.avg("PAYMENT_DIFF").alias("INST_PAYMENT_DIFF_MEAN"),
        F.max("PAYMENT_DIFF").alias("INST_PAYMENT_DIFF_MAX"),
        F.avg("PAYMENT_RATIO").alias("INST_PAYMENT_RATIO_MEAN"),
        F.avg("DAYS_PAYMENT_DIFF").alias("INST_DAYS_PAYMENT_DIFF_MEAN"),
        F.max("DAYS_PAYMENT_DIFF").alias("INST_DAYS_PAYMENT_DIFF_MAX"),
        F.coalesce(
            F.sum(F.when(F.col("DAYS_PAYMENT_DIFF") > 0, 1).otherwise(0)), F.lit(0)
        ).alias("INST_LATE_PAYMENT_COUNT"),
    )


def credit_card_features(cc: DataFrame) -> DataFrame:
    cc = cc.withColumn("UTILIZATION", F.col("AMT_BALANCE") / nz("AMT_CREDIT_LIMIT_ACTUAL"))
    return cc.groupBy("SK_ID_CURR").agg(
        F.count("SK_ID_PREV").alias("CC_COUNT"),
        F.avg("AMT_BALANCE").alias("CC_AMT_BALANCE_MEAN"),
        F.max("AMT_BALANCE").alias("CC_AMT_BALANCE_MAX"),
        F.avg("UTILIZATION").alias("CC_UTILIZATION_MEAN"),
        F.max("UTILIZATION").alias("CC_UTILIZATION_MAX"),
        F.avg("SK_DPD").alias("CC_SK_DPD_MEAN"),
        F.max("SK_DPD").alias("CC_SK_DPD_MAX"),
        F.avg("AMT_DRAWINGS_CURRENT").alias("CC_AMT_DRAWINGS_CURRENT_MEAN"),
        F.avg("AMT_PAYMENT_TOTAL_CURRENT").alias("CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN"),
    )

# COMMAND ----------

# MAGIC %md ## Assemble and write the gold feature table

# COMMAND ----------

app = application_features(spark.table(f"{silver}.application"))
bureau = bureau_features(spark.table(f"{silver}.bureau"), spark.table(f"{silver}.bureau_balance"))
prev = previous_application_features(spark.table(f"{silver}.previous_application"))
pos = pos_cash_features(spark.table(f"{silver}.pos_cash_balance"))
inst = installments_features(spark.table(f"{silver}.installments_payments"))
cc = credit_card_features(spark.table(f"{silver}.credit_card_balance"))

features = (
    app.join(bureau, "SK_ID_CURR", "left")
    .join(prev, "SK_ID_CURR", "left")
    .join(pos, "SK_ID_CURR", "left")
    .join(inst, "SK_ID_CURR", "left")
    .join(cc, "SK_ID_CURR", "left")
)

target_table = f"{gold}.features"
(
    features.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(target_table)
)

# Primary key => Unity Catalog feature table (informational constraint, not enforced)
spark.sql(f"ALTER TABLE {target_table} ALTER COLUMN SK_ID_CURR SET NOT NULL")
spark.sql(f"ALTER TABLE {target_table} DROP CONSTRAINT IF EXISTS features_pk")
spark.sql(f"ALTER TABLE {target_table} ADD CONSTRAINT features_pk PRIMARY KEY (SK_ID_CURR)")
spark.sql(f"OPTIMIZE {target_table} ZORDER BY (SK_ID_CURR)")

df = spark.table(target_table)
print(f"{target_table}: {df.count():,} rows x {len(df.columns)} columns")
display(df.groupBy("IS_TRAIN").count())
