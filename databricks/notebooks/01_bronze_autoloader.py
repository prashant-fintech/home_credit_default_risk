# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze — ingest raw CSVs with Auto Loader
# MAGIC
# MAGIC One Delta table per CSV, loaded with **Auto Loader** (`cloudFiles`).
# MAGIC
# MAGIC **Concepts (exam topics)**
# MAGIC - Auto Loader is a Structured Streaming source. `trigger(availableNow=True)` runs it as a batch:
# MAGIC   process everything new, then stop. Re-running only picks up files not yet seen.
# MAGIC - The **checkpoint** stores which files were ingested; the **schemaLocation** stores the inferred
# MAGIC   schema and drives *schema evolution* (`addNewColumns` by default: a new column fails the stream
# MAGIC   once, records it, and succeeds on the next run).
# MAGIC - Bronze keeps the data as-is plus lineage columns (`_ingested_at`, `_source_file`). Cleaning is
# MAGIC   silver's job.
# MAGIC - `cloudFiles.inferColumnTypes` is off by default (everything would be `string`); we turn it on.

# COMMAND ----------

from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "learningdatabricks")
dbutils.widgets.text("raw_path", "/Volumes/learningdatabricks/home-credit-default-risk/home-credit-default-risk")
catalog = dbutils.widgets.get("catalog")

raw_dir = dbutils.widgets.get("raw_path").rstrip("/") + "/"
ckpt_dir = f"/Volumes/{catalog}/bronze/checkpoints/"

TABLES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "installments_payments": "installments_payments.csv",
    "credit_card_balance": "credit_card_balance.csv",
}

# COMMAND ----------


def ingest(table: str, filename: str) -> None:
    target = f"{catalog}.bronze.{table}"
    checkpoint = f"{ckpt_dir}{table}"

    stream = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.schemaLocation", checkpoint)
        .option("cloudFiles.inferColumnTypes", "true")
        .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
        .option("header", "true")
        .option("pathGlobFilter", filename)  # one table per file
        .load(raw_dir)
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_file", F.col("_metadata.file_path"))
    )

    query = (
        stream.writeStream.format("delta")
        .option("checkpointLocation", checkpoint)
        .option("mergeSchema", "true")
        .trigger(availableNow=True)
        .toTable(target)
    )
    query.awaitTermination()
    n = spark.table(target).count()
    print(f"{target:<45} {n:>12,} rows")


for table, filename in TABLES.items():
    ingest(table, filename)

# COMMAND ----------

# MAGIC %md ## Inspect: schema inferred by Auto Loader and Delta history for one table

# COMMAND ----------

spark.table(f"{catalog}.bronze.application_train").printSchema()
display(spark.sql(f"DESCRIBE HISTORY {catalog}.bronze.application_train"))

# COMMAND ----------

# MAGIC %md
# MAGIC **Try it:** upload a second CSV that matches the glob (or re-upload the same file with a new name)
# MAGIC and re-run this notebook — only the new file is ingested. Drop a column from the new file and
# MAGIC watch schema evolution add `_rescued_data`.
