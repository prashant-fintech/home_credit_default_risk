"""End-to-end training pipeline.

    python scripts/train.py [--config configs/model.yaml] [--no-cache]
"""

import argparse
from pathlib import Path

from home_credit.data.loader import load
from home_credit.features.pipeline import build
from home_credit.models.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.yaml", type=Path)
    parser.add_argument("--no-cache", action="store_true", help="recompute features even if cached")
    args = parser.parse_args()

    print("Loading raw data...")
    dataset = load()

    if args.no_cache:
        cache = Path("data/processed/features.parquet")
        cache.unlink(missing_ok=True)

    print("Building features...")
    features = build(dataset)

    train_ids = dataset.train_ids
    X_train = features.loc[train_ids].drop(columns=["TARGET"], errors="ignore")
    y_train = dataset.target

    print(f"Training on {len(X_train):,} rows, {X_train.shape[1]} features...")
    result = train(X_train, y_train, config_path=args.config)

    print("\nTop 20 features by importance:")
    print(result.feature_importance.head(20).to_string(index=False))
