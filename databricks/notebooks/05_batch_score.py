# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Batch scoring with the champion model
# MAGIC
# MAGIC Loads `models:/<catalog>.gold.lgbm_default_risk@champion` from the Unity Catalog registry as a
# MAGIC **Spark UDF** and scores every row without a `TARGET` (the Kaggle test set) in parallel.
# MAGIC
# MAGIC **Concepts (exam topics)**
# MAGIC - `mlflow.pyfunc.spark_udf` wraps any logged model so Spark can call it per partition.
# MAGIC - Model URIs with `@alias` resolve to whichever version currently carries the alias, so promoting
# MAGIC   a new champion changes what this job scores with, with no code change.
# MAGIC - Predictions are written to a gold Delta table with the model version for auditability.

# COMMAND ----------

# MAGIC %pip install --quiet lightgbm scikit-learn

# COMMAND ----------

import mlflow
from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "learningdatabricks")
catalog = dbutils.widgets.get("catalog")

MODEL_NAME = f"{catalog}.gold.lgbm_default_risk"
ALIAS = "champion"
model_uri = f"models:/{MODEL_NAME}@{ALIAS}"

mlflow.set_registry_uri("databricks-uc")
version = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS).version
print(f"scoring with {model_uri} (version {version})")

# COMMAND ----------

predict = mlflow.pyfunc.spark_udf(spark, model_uri, result_type="double")
feature_cols = [c.name for c in predict.metadata.get_input_schema().inputs]

to_score = spark.table(f"{catalog}.gold.features").filter("NOT IS_TRAIN")

scored = (
    to_score.withColumn("TARGET_PROB", predict(F.struct(*feature_cols)))
    .select("SK_ID_CURR", "TARGET_PROB")
    .withColumn("MODEL_VERSION", F.lit(int(version)))
    .withColumn("SCORED_AT", F.current_timestamp())
)

(
    scored.write.format("delta")
    .mode("overwrite")
    .saveAsTable(f"{catalog}.gold.predictions")
)

# COMMAND ----------

df = spark.table(f"{catalog}.gold.predictions")
print(f"{df.count():,} applicants scored")
display(
    df.select(
        F.mean("TARGET_PROB").alias("mean_prob"),
        F.expr("percentile(TARGET_PROB, 0.5)").alias("median_prob"),
        F.max("TARGET_PROB").alias("max_prob"),
    )
)
display(df.orderBy(F.desc("TARGET_PROB")).limit(10))
