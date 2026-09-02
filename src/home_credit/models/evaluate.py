"""Credit risk evaluation metrics.

ROC-AUC is the competition metric. Gini (= 2*AUC - 1) is the industry
standard. KS statistic separates the score distributions. All metrics
are logged to MLflow when an active run exists.
"""

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return 2 * roc_auc_score(y_true, y_score) - 1


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max separation between default / non-default CDFs."""
    df = pd.DataFrame({"score": y_score, "target": y_true}).sort_values("score", ascending=False)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    df["cum_pos"] = df["target"].cumsum() / n_pos
    df["cum_neg"] = (1 - df["target"]).cumsum() / n_neg
    return (df["cum_pos"] - df["cum_neg"]).abs().max()


def report(y_true: np.ndarray, y_score: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Compute and log AUC, Gini, KS. Returns the metric dict."""
    p = f"{prefix}_" if prefix else ""
    auc = float(roc_auc_score(y_true, y_score))
    metrics = {
        f"{p}auc": auc,
        f"{p}gini": 2 * auc - 1,
        f"{p}ks": float(ks_statistic(y_true, y_score)),
    }
    try:
        mlflow.log_metrics(metrics)
    except Exception:
        pass
    return metrics
