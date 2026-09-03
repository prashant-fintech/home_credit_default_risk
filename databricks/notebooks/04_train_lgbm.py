# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Train LightGBM with MLflow, register in Unity Catalog
# MAGIC
# MAGIC Mirrors `home_credit/models/train.py`: stratified K-fold, early stopping, OOF AUC / Gini / KS,
# MAGIC best fold registered and aliased `champion`.
# MAGIC
# MAGIC **Concepts (exam topics)**
# MAGIC - MLflow tracking on Databricks is built in: runs go to an *experiment* in the workspace tree.
# MAGIC - `mlflow.set_registry_uri("databricks-uc")` sends `log_model(registered_model_name=...)` to the
# MAGIC   **Unity Catalog model registry** (`catalog.schema.model`). UC models **require a signature**.
# MAGIC - **Aliases** (`@champion`) replace the legacy Production/Staging stages.
# MAGIC - The model is wrapped as a custom `pyfunc` so string columns are cast to pandas `category` inside
# MAGIC   `predict()`. That makes the same artifact work from `mlflow.pyfunc.spark_udf` (batch scoring)
# MAGIC   and Model Serving without callers knowing about LightGBM categorical handling.

# COMMAND ----------

# MAGIC %pip install --quiet lightgbm scikit-learn

# COMMAND ----------

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, DataType, Schema
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

dbutils.widgets.text("catalog", "learningdatabricks")
dbutils.widgets.text("n_folds", "5")
catalog = dbutils.widgets.get("catalog")
n_folds = int(dbutils.widgets.get("n_folds"))

MODEL_NAME = f"{catalog}.gold.lgbm_default_risk"
ALIAS = "champion"
SEED = 42

# Same hyper-parameters as configs/model.yaml in the repo
LGBM_PARAMS = dict(
    objective="binary", metric="auc", boosting_type="gbdt",
    learning_rate=0.02, num_leaves=63, max_depth=-1, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1, verbose=-1,
)
N_ESTIMATORS = 5000
EARLY_STOPPING = 100

# COMMAND ----------

# MAGIC %md ## Load the gold feature table (training rows only) into pandas

# COMMAND ----------

NON_FEATURES = ["SK_ID_CURR", "TARGET", "IS_TRAIN"]

pdf = spark.table(f"{catalog}.gold.features").filter("IS_TRAIN").toPandas()
y = pdf["TARGET"].astype(int)
X_raw = pdf.drop(columns=NON_FEATURES)

obj_cols = [c for c in X_raw.columns if X_raw[c].dtype == "object"]
num_cols = [c for c in X_raw.columns if c not in obj_cols]
X = X_raw.copy()
for c in obj_cols:
    X[c] = X[c].astype("category")
print(f"{len(X):,} rows, {X.shape[1]} features ({len(obj_cols)} categorical)")

# COMMAND ----------

# MAGIC %md ## Metrics (same definitions as `home_credit/models/evaluate.py`)

# COMMAND ----------


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    df = pd.DataFrame({"score": y_score, "target": y_true}).sort_values("score", ascending=False)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    cum_pos = df["target"].cumsum() / n_pos
    cum_neg = (1 - df["target"]).cumsum() / n_neg
    return float((cum_pos - cum_neg).abs().max())


def report(y_true, y_score, prefix: str) -> dict[str, float]:
    auc = float(roc_auc_score(y_true, y_score))
    metrics = {
        f"{prefix}_auc": auc,
        f"{prefix}_gini": 2 * auc - 1,
        f"{prefix}_ks": ks_statistic(np.asarray(y_true), np.asarray(y_score)),
    }
    mlflow.log_metrics(metrics)
    return metrics

# COMMAND ----------

# MAGIC %md ## pyfunc wrapper: string -> category inside predict()

# COMMAND ----------


class LGBMCategoricalModel(mlflow.pyfunc.PythonModel):
    def __init__(self, booster: lgb.Booster, num_cols: list[str], obj_cols: list[str]):
        self.booster = booster
        self.num_cols = num_cols
        self.obj_cols = obj_cols

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        df = model_input[self.num_cols + self.obj_cols].copy()
        df[self.num_cols] = df[self.num_cols].astype("float64")
        for c in self.obj_cols:
            # LightGBM re-aligns categories to the training categories via pandas_categorical
            df[c] = df[c].astype("category")
        return self.booster.predict(df)


signature = ModelSignature(
    inputs=Schema(
        [ColSpec(DataType.double, c) for c in num_cols]
        + [ColSpec(DataType.string, c) for c in obj_cols]
    ),
    outputs=Schema([ColSpec(DataType.double)]),
)

# COMMAND ----------

# MAGIC %md ## K-fold training

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/home-credit-lgbm")

with mlflow.start_run(run_name=f"lgbm_{n_folds}fold") as run:
    mlflow.log_params({
        **LGBM_PARAMS,
        "n_folds": n_folds,
        "seed": SEED,
        "n_estimators": N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING,
        "feature_table": f"{catalog}.gold.features",
    })

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    boosters: list[lgb.Booster] = []
    fold_auc: list[float] = []
    importances: list[pd.DataFrame] = []

    for fold, (trn_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        model = lgb.LGBMClassifier(**LGBM_PARAMS, n_estimators=N_ESTIMATORS, random_state=SEED)
        # eval_set works on every LightGBM 4.x (the ML runtime may predate eval_X/eval_y)
        model.fit(
            X.iloc[trn_idx], y.iloc[trn_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(200)],
        )
        preds = np.asarray(model.predict_proba(X.iloc[val_idx]))[:, 1]
        oof[val_idx] = preds
        m = report(y.iloc[val_idx].values, preds, prefix=f"fold{fold}")
        fold_auc.append(m[f"fold{fold}_auc"])
        boosters.append(model.booster_)
        importances.append(
            pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        )
        print(f"fold {fold}: AUC {m[f'fold{fold}_auc']:.4f}  best_iter {model.best_iteration_}")

    oof_metrics = report(y.values, oof, prefix="oof")
    print(
        f"\nOOF AUC {oof_metrics['oof_auc']:.4f} | Gini {oof_metrics['oof_gini']:.4f} "
        f"| KS {oof_metrics['oof_ks']:.4f}"
    )

    importance = (
        pd.concat(importances).groupby("feature")["importance"].mean()
        .sort_values(ascending=False).reset_index()
    )
    mlflow.log_table(importance, "feature_importance.json")

    best_fold = int(np.argmax(fold_auc))
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=LGBMCategoricalModel(boosters[best_fold], num_cols, obj_cols),
        signature=signature,
        input_example=X_raw.head(5),
        registered_model_name=MODEL_NAME,
        pip_requirements=[f"lightgbm=={lgb.__version__}", f"pandas=={pd.__version__}"],
    )

client = mlflow.MlflowClient()
client.set_registered_model_alias(MODEL_NAME, ALIAS, info.registered_model_version)
print(
    f"Registered {MODEL_NAME} v{info.registered_model_version} as @{ALIAS} "
    f"(fold {best_fold + 1}, AUC {fold_auc[best_fold]:.4f})"
)

# COMMAND ----------

display(importance.head(25))
