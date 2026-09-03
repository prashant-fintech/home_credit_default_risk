# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Batch scoring with the champion model
# MAGIC
# MAGIC Loads `models:/<catalog>.gold.lgbm_default_risk@champion` from the Unity Catalog registry and
# MAGIC scores every row without a `TARGET` (the Kaggle test set), writing a gold predictions table.
# MAGIC
# MAGIC **Concepts (exam topics)**
# MAGIC - Model URIs with `@alias` resolve to whichever version currently carries the alias, so promoting
# MAGIC   a new champion changes what this job scores with, with no code change.
# MAGIC - `mlflow.pyfunc.load_model` gives a flavour-agnostic `predict(pandas)`; the model signature is
# MAGIC   **enforced** on every call, so inputs must match the declared types exactly.
# MAGIC - `mlflow.pyfunc.spark_udf` is the distributed variant for large tables. On this serverless
# MAGIC   runtime the UDF hands MLflow pandas batches with nullable `Int64` columns, which schema
# MAGIC   enforcement refuses to widen to `double`, so for the 48k-row test set we score on the driver
# MAGIC   and normalise dtypes ourselves. (Exercise: wrap the model in your own `pandas_udf` that does
# MAGIC   the same casts, to get distributed scoring back.)
# MAGIC - `dbutils.notebook.exit(json)` returns a value from a task; visible in the run output and
# MAGIC   usable by downstream tasks via `{{tasks.<key>.values}}` / task values.

# COMMAND ----------

# MAGIC %pip install --quiet lightgbm scikit-learn

# COMMAND ----------

import json

import mlflow
import numpy as np
import pandas as pd
from mlflow.types import DataType
from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "learningdatabricks")
catalog = dbutils.widgets.get("catalog")

MODEL_NAME = f"{catalog}.gold.lgbm_default_risk"
ALIAS = "champion"
model_uri = f"models:/{MODEL_NAME}@{ALIAS}"

mlflow.set_registry_uri("databricks-uc")
version = int(mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS).version)
print(f"scoring with {model_uri} (version {version})")

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri)
input_schema = model.metadata.get_input_schema()
feature_cols = [c.name for c in input_schema.inputs]
double_cols = [c.name for c in input_schema.inputs if c.type == DataType.double]
string_cols = [c.name for c in input_schema.inputs if c.type == DataType.string]
assert double_cols and string_cols, "unexpected signature"

to_score = spark.table(f"{catalog}.gold.features").filter("NOT IS_TRAIN").select(
    "SK_ID_CURR", *feature_cols
)
pdf = to_score.toPandas()

diagnostics = {
    "spark_type_BUREAU_LOAN_COUNT": dict(to_score.dtypes)["BUREAU_LOAN_COUNT"],
    "pandas_dtype_before": str(pdf["BUREAU_LOAN_COUNT"].dtype),
}

# Match the signature exactly: numerics -> float64 (nullable Int64 / int64 both widen cleanly),
# strings -> plain object with None for missing.
pdf[double_cols] = pdf[double_cols].astype("float64")
pdf[string_cols] = pdf[string_cols].astype(object).where(pdf[string_cols].notna(), None)

probs = np.asarray(model.predict(pdf[feature_cols]), dtype="float64")

scored_pdf = pd.DataFrame(
    {"SK_ID_CURR": pdf["SK_ID_CURR"].astype("int64"), "TARGET_PROB": probs}
)
scored = (
    spark.createDataFrame(scored_pdf)
    .withColumn("MODEL_VERSION", F.lit(version))
    .withColumn("SCORED_AT", F.current_timestamp())
)
scored.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.gold.predictions")

# COMMAND ----------

df = spark.table(f"{catalog}.gold.predictions")
stats = df.select(
    F.count("*").alias("scored"),
    F.mean("TARGET_PROB").alias("mean_prob"),
    F.expr("percentile(TARGET_PROB, 0.5)").alias("median_prob"),
    F.max("TARGET_PROB").alias("max_prob"),
).first().asDict()
display(df.orderBy(F.desc("TARGET_PROB")).limit(10))

dbutils.notebook.exit(json.dumps({"model_version": version, **stats, **diagnostics}))
