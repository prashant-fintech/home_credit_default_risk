"""Post-training explanation script.

Loads the latest trained models from MLflow, computes SHAP values and
WoE/IV analysis, and saves the results as artifacts.

Usage:
    python scripts/explain.py [--run-id RUN_ID] [--top-n 20] [--n-sample 5000]
"""

import argparse
from pathlib import Path

import mlflow

from home_credit.config import settings
from home_credit.data.loader import load
from home_credit.explain import shap_explainer
from home_credit.features.pipeline import build
from home_credit.features.woe_iv import WoEEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--n-sample", type=int, default=5_000)
    args = parser.parse_args()

    print("Loading data and features...")
    dataset = load()
    features = build(dataset)

    train_ids = dataset.train_ids
    X_train = features.loc[train_ids].drop(columns=["TARGET"], errors="ignore")
    y_train = dataset.target

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    # --- load fold models from MLflow ---
    if args.run_id:
        run = mlflow.get_run(args.run_id)
    else:
        runs = mlflow.search_runs(
            experiment_names=["home-credit-lgbm"],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            raise SystemExit("No MLflow runs found — train a model first: python scripts/train.py")
        run = mlflow.get_run(runs.iloc[0]["run_id"])

    run_id = run.info.run_id
    print(f"Using MLflow run {run_id}")

    model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")

    # --- SHAP ---
    print(f"\nComputing SHAP values on {args.n_sample:,} samples...")
    shap_vals = shap_explainer.compute([model], X_train, n_sample=args.n_sample)

    imp = shap_explainer.importance_df(shap_vals, X_train.columns.tolist())
    print(f"\nTop {args.top_n} features by mean |SHAP|:")
    print(imp.head(args.top_n).to_string(index=False))

    out_dir = Path("data/processed")
    out_dir.mkdir(exist_ok=True)
    imp.to_csv(out_dir / "shap_importance.csv", index=False)
    print(f"\nSaved SHAP importance → {out_dir}/shap_importance.csv")

    # Waterfall for the first applicant in the sample
    wf = shap_explainer.waterfall_data(shap_vals, X_train, idx=0, top_n=15)
    wf.to_csv(out_dir / "shap_waterfall_sample0.csv", index=False)
    print(f"Saved waterfall (applicant 0) → {out_dir}/shap_waterfall_sample0.csv")

    # --- WoE / IV ---
    print("\nFitting WoE encoder and computing Information Values...")
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    enc = WoEEncoder(n_bins=10).fit(X_train[num_cols], y_train)
    iv = enc.iv_summary()

    print(f"\nTop {args.top_n} features by IV:")
    print(iv.head(args.top_n).to_string(index=False))

    iv.to_csv(out_dir / "iv_summary.csv", index=False)
    print(f"\nSaved IV summary → {out_dir}/iv_summary.csv")

    # Features worth keeping (IV > 0.02 = at least weakly predictive)
    good = iv[iv["iv"] > 0.02]
    suspect = iv[iv["iv"] > 0.50]
    print(f"\n{len(good)} features with IV > 0.02 (worth keeping)")
    if not suspect.empty:
        print(f"WARNING: {len(suspect)} features with IV > 0.50 — check for target leakage:")
        print(suspect[["feature", "iv"]].to_string(index=False))


if __name__ == "__main__":
    main()
