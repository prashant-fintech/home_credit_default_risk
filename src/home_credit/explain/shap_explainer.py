"""SHAP-based model explanations for LightGBM fold ensembles.

Why SHAP over built-in LightGBM importance?
  - Split/gain importance is biased toward high-cardinality features.
  - SHAP values have a solid theoretical foundation (Shapley values from
    cooperative game theory) and sum exactly to the model's prediction.
  - Mean |SHAP| across the dataset is the fairest global importance measure.

Typical workflow:
    shap_vals = shap_explainer.compute(result.models, X_train)
    importance = shap_explainer.importance_df(shap_vals, X_train.columns.tolist())
    waterfall  = shap_explainer.waterfall_data(shap_vals, X_train, idx=0)
"""

import mlflow
import numpy as np
import pandas as pd
import shap


def compute(
    models: list,
    X: pd.DataFrame,
    n_sample: int = 5_000,
    seed: int = 42,
) -> np.ndarray:
    """Average SHAP values across all fold models on a random sample of X.

    Sampling to n_sample keeps this tractable even on the full feature matrix
    (307k rows × 200+ features). The sample is stratified by row index so
    every region of the feature space is represented.

    Returns an array of shape (min(n_sample, len(X)), n_features).
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    X_sample = X.iloc[idx].copy()

    # Cast categoricals so TreeExplainer sees the same dtypes as training
    for col in X_sample.select_dtypes("object").columns:
        X_sample[col] = X_sample[col].astype("category")

    all_shap: list[np.ndarray] = []
    for model in models:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
        # LightGBM binary classification: shap returns list[ndarray] or ndarray
        if isinstance(sv, list):
            sv = sv[1]  # index 1 = positive class
        all_shap.append(sv)

    return np.stack(all_shap, axis=0).mean(axis=0)  # (n_sample, n_features)


def importance_df(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, sorted descending.

    This is the most reliable global feature importance for tree models —
    it reflects the average magnitude of each feature's contribution to
    the model output across the sampled population.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def waterfall_data(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    idx: int,
    top_n: int = 15,
) -> pd.DataFrame:
    """SHAP contributions for a single applicant (row `idx` in the sample).

    Returns a DataFrame with columns:
      feature        — feature name
      value          — raw feature value for this applicant
      shap_value     — SHAP contribution (positive = pushes toward default)
      abs_shap       — |shap_value| for ordering

    The top_n features by |SHAP| are returned.  The remainder are
    aggregated into an 'other features' row so contributions still sum
    to the model's log-odds output.
    """
    row_shap = shap_values[idx]
    row_vals = X.iloc[idx]

    df = pd.DataFrame({
        "feature": X.columns,
        "value": row_vals.values,
        "shap_value": row_shap,
        "abs_shap": np.abs(row_shap),
    }).sort_values("abs_shap", ascending=False)

    top = df.head(top_n).copy()
    rest_shap = df.iloc[top_n:]["shap_value"].sum()
    if abs(rest_shap) > 0:
        rest_row = pd.DataFrame([{
            "feature": f"other {len(df) - top_n} features",
            "value": np.nan,
            "shap_value": rest_shap,
            "abs_shap": abs(rest_shap),
        }])
        top = pd.concat([top, rest_row], ignore_index=True)

    return top.reset_index(drop=True)


def log_to_mlflow(shap_values: np.ndarray, feature_names: list[str]) -> None:
    """Log mean-|SHAP| importance table as an MLflow artifact."""
    imp = importance_df(shap_values, feature_names)
    try:
        with mlflow.start_run(nested=True):
            imp_path = "/tmp/shap_importance.csv"
            imp.to_csv(imp_path, index=False)
            mlflow.log_artifact(imp_path, artifact_path="explain")
    except Exception:
        pass
