# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver — typed, deduplicated, constrained
# MAGIC
# MAGIC **Concepts (exam topics)**
# MAGIC - `MERGE INTO` (upsert) makes a load idempotent: re-running updates existing keys instead of
# MAGIC   duplicating rows. Used for the `application` table which has a natural key (`SK_ID_CURR`).
# MAGIC - Satellite tables have no clean single-column key, so they are rebuilt with
# MAGIC   `mode("overwrite")` — Delta makes that atomic and keeps the old version for time travel.
# MAGIC - **Constraints**: `NOT NULL` and `CHECK` are enforced on write; a violating write fails the job.
# MAGIC - **OPTIMIZE / ZORDER**: compacts small files and co-locates rows by the join key so the gold
# MAGIC   joins read fewer files. (`VACUUM` would then delete files older than the retention window.)
# MAGIC - `DESCRIBE HISTORY` + `VERSION AS OF` = time travel.

# COMMAND ----------

from delta.tables import DeltaTable
from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "learningdatabricks")
catalog = dbutils.widgets.get("catalog")
bronze = f"{catalog}.bronze"
silver = f"{catalog}.silver"

LINEAGE_COLS = ["_ingested_at", "_source_file", "_rescued_data"]


def drop_lineage(df):
    return df.drop(*[c for c in LINEAGE_COLS if c in df.columns])

# COMMAND ----------

# MAGIC %md ## application = train + test (TARGET is null for test rows), upserted with MERGE

# COMMAND ----------

train = drop_lineage(spark.table(f"{bronze}.application_train"))
test = drop_lineage(spark.table(f"{bronze}.application_test")).withColumn(
    "TARGET", F.lit(None).cast("int")
)

application = (
    train.withColumn("IS_TRAIN", F.lit(True))
    .unionByName(test.withColumn("IS_TRAIN", F.lit(False)), allowMissingColumns=True)
    # DAYS_EMPLOYED = 365243 is the dataset's "not employed" sentinel (~1000 years)
    .withColumn(
        "DAYS_EMPLOYED",
        F.when(F.col("DAYS_EMPLOYED") == 365243, None).otherwise(F.col("DAYS_EMPLOYED")),
    )
    .withColumn("TARGET", F.col("TARGET").cast("int"))
    .dropDuplicates(["SK_ID_CURR"])
)

target_table = f"{silver}.application"
if not spark.catalog.tableExists(target_table):
    application.limit(0).write.format("delta").saveAsTable(target_table)
    spark.sql(f"ALTER TABLE {target_table} ALTER COLUMN SK_ID_CURR SET NOT NULL")
    spark.sql(
        f"ALTER TABLE {target_table} ADD CONSTRAINT target_is_binary "
        "CHECK (TARGET IS NULL OR TARGET IN (0, 1))"
    )
    spark.sql(
        f"ALTER TABLE {target_table} ADD CONSTRAINT positive_income "
        "CHECK (AMT_INCOME_TOTAL > 0)"
    )

(
    DeltaTable.forName(spark, target_table)
    .alias("t")
    .merge(application.alias("s"), "t.SK_ID_CURR = s.SK_ID_CURR")
    .whenMatchedUpdateAll()
    .whenNotMatchedInsertAll()
    .execute()
)
print(f"{target_table}: {spark.table(target_table).count():,} rows")

# COMMAND ----------

# MAGIC %md ## Satellite tables: dedupe, drop lineage, overwrite

# COMMAND ----------

SATELLITES = [
    "bureau",
    "bureau_balance",
    "previous_application",
    "pos_cash_balance",
    "installments_payments",
    "credit_card_balance",
]

for name in SATELLITES:
    df = drop_lineage(spark.table(f"{bronze}.{name}")).dropDuplicates()
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{silver}.{name}")
    )
    n = spark.table(f"{silver}.{name}").count()
    print(f"{silver}.{name:<25} {n:>12,} rows")

# COMMAND ----------

# MAGIC %md ## Layout: compact + Z-order on the join key

# COMMAND ----------

for name in ["application", "bureau", "previous_application", "pos_cash_balance",
             "installments_payments", "credit_card_balance"]:
    spark.sql(f"OPTIMIZE {silver}.{name} ZORDER BY (SK_ID_CURR)")
spark.sql(f"OPTIMIZE {silver}.bureau_balance ZORDER BY (SK_ID_BUREAU)")

# COMMAND ----------

# MAGIC %md ## Time travel

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY {silver}.application"))
display(spark.sql(f"SELECT COUNT(*) AS rows_at_v0 FROM {silver}.application VERSION AS OF 0"))
