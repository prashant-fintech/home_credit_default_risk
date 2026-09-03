"""Generate predictions for the test set using all fold models.

    python scripts/predict.py [--output data/processed/submission.csv]
"""

import argparse
from pathlib import Path

import mlflow
import pandas as pd

from home_credit.config import settings
from home_credit.data.loader import load
from home_credit.features.pipeline import build

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/submission.csv", type=Path)
    parser.add_argument("--run-id", default=None, help="MLflow run ID (latest if omitted)")
    args = parser.parse_args()

    print("Loading data and features...")
    dataset = load()
    features = build(dataset)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_uri = (
        f"runs:/{args.run_id}/model"
        if args.run_id
        else f"models:/{settings.model_registry_name}@{settings.model_alias}"
    )
    print(f"Loading model from {model_uri}...")
    model = mlflow.pyfunc.load_model(model_uri)

    test_ids = dataset.test_ids
    X_test = features.loc[test_ids].drop(columns=["TARGET"], errors="ignore")

    print(f"Scoring {len(X_test):,} test rows...")
    preds = model.predict(X_test)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"SK_ID_CURR": test_ids.values, "TARGET": preds}).to_csv(
        args.output, index=False
    )
    print(f"Saved to {args.output}")
