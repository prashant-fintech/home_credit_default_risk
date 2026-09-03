"""Stratified K-fold LightGBM training with MLflow tracking.

Usage:
    python scripts/train.py
"""

from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from home_credit.config import settings
from home_credit.models.evaluate import report


@dataclass
class TrainResult:
    oof_predictions: np.ndarray
    models: list[lgb.Booster] = field(default_factory=list)
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    fold_metrics: list[dict] = field(default_factory=list)


def _load_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text())


def train(
    X: pd.DataFrame,
    y: pd.Series,
    config_path: Path = Path("configs/model.yaml"),
    experiment_name: str = "home-credit-lgbm",
) -> TrainResult:
    """Stratified K-fold training. Returns OOF predictions and fold models."""
    cfg = _load_config(config_path)
    lgbm_params = cfg["lightgbm"]
    train_cfg = cfg["training"]

    n_folds = train_cfg["n_folds"]
    seed = train_cfg["seed"]
    early_stopping = lgbm_params.pop("early_stopping_rounds", 100)
    n_estimators = lgbm_params.pop("n_estimators", 5000)

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    for col in cat_cols:
        X[col] = X[col].astype("category")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    result = TrainResult(oof_predictions=np.zeros(len(y)))

    with mlflow.start_run():
        mlflow.log_params({**lgbm_params, "n_folds": n_folds, "seed": seed})
        mlflow.log_artifact(str(config_path))

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        importances: list[pd.DataFrame] = []

        for fold, (trn_idx, val_idx) in enumerate(cv.split(X, y)):
            X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
            y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]

            model = lgb.LGBMClassifier(
                **lgbm_params,
                n_estimators=n_estimators,
                random_state=seed,
                categorical_feature=cat_cols or "auto",
            )
            model.fit(
                X_trn, y_trn,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(early_stopping, verbose=False),
                    lgb.log_evaluation(100),
                ],
            )

            val_preds = np.asarray(model.predict_proba(X_val))[:, 1]
            result.oof_predictions[val_idx] = val_preds

            fold_metrics = report(y_val.values, val_preds, prefix=f"fold{fold+1}")
            result.fold_metrics.append(fold_metrics)
            result.models.append(model.booster_)
            print(f"Fold {fold+1} | AUC {fold_metrics[f'fold{fold+1}_auc']:.4f}")

            importances.append(
                pd.DataFrame(
                    {"feature": X.columns, "importance": model.feature_importances_}
                )
            )

        oof_metrics = report(y.values, result.oof_predictions, prefix="oof")
        print(f"\nOOF AUC {oof_metrics['oof_auc']:.4f}  |  Gini {oof_metrics['oof_gini']:.4f}"
              f"  |  KS {oof_metrics['oof_ks']:.4f}")

        result.feature_importance = (
            pd.concat(importances)
            .groupby("feature")["importance"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        # Register best model (fold with highest val AUC)
        best_fold = max(range(n_folds), key=lambda i: result.fold_metrics[i][f"fold{i+1}_auc"])
        mlflow.lightgbm.log_model(
            result.models[best_fold],
            artifact_path="model",
            registered_model_name=settings.model_registry_name,
        )

    return result


def predict(models: list[lgb.Booster], X: pd.DataFrame) -> np.ndarray:
    """Average predictions across all fold models."""
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    for col in cat_cols:
        X[col] = X[col].astype("category")
    preds = np.stack([m.predict(X) for m in models], axis=0)
    return preds.mean(axis=0)
