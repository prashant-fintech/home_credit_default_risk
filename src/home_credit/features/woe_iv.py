"""Weight of Evidence (WoE) encoding and Information Value (IV) analysis.

Background
----------
WoE and IV are workhorses of traditional credit scorecard development and
are still required by many regulators who want interpretable models.

  WoE_i  = ln( (Events_i   / Total_Events)
              / (NonEvents_i / Total_NonEvents) )

  IV_i   = (Events_i/Total_Events - NonEvents_i/Total_NonEvents) * WoE_i

  IV     = Σ IV_i  over all bins of a feature

IV interpretation (Siddiqi, 2006):
  < 0.02          unpredictive   — drop the feature
  0.02 – 0.10     weak
  0.10 – 0.30     medium
  0.30 – 0.50     strong
  > 0.50          suspicious    — check for target leakage

WoE is monotonic in predictive power per bin, which makes it easy to
spot non-linearities and build scorecards that regulators can read.

Usage
-----
    enc = WoEEncoder(n_bins=10).fit(X_train, y_train)
    X_woe = enc.transform(X_train)           # all features replaced by WoE
    summary = enc.iv_summary()               # IV table for feature selection
    good_features = summary[summary["iv"] > 0.02]["feature"].tolist()
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


IV_LABELS = [
    (0.02,  "unpredictive"),
    (0.10,  "weak"),
    (0.30,  "medium"),
    (0.50,  "strong"),
    (float("inf"), "suspicious — check for leakage"),
]

_CLIP = 1e-10  # prevents ln(0) when a bin has 0 events or 0 non-events


def _iv_label(iv: float) -> str:
    for threshold, label in IV_LABELS:
        if iv < threshold:
            return label
    return IV_LABELS[-1][1]


@dataclass
class _BinStats:
    """WoE and IV data for one feature."""
    feature: str
    woe_map: dict        # bin_label -> WoE value
    missing_woe: float   # WoE for NaN bin (may be 0 if no NaNs in training)
    iv: float
    bin_table: pd.DataFrame  # full diagnostics table


def _compute_bins(
    series: pd.Series,
    y: pd.Series,
    n_bins: int,
    is_categorical: bool,
) -> _BinStats:
    """Fit WoE/IV for a single feature."""
    total_events = float(y.sum())
    total_nonevents = float((1 - y).sum())

    rows = []

    # --- missing values bin ---
    mask_nan = series.isna()
    if mask_nan.any():
        e = float(y[mask_nan].sum())
        ne = float((1 - y[mask_nan]).sum())
        rows.append({"bin": "__MISSING__", "events": e, "non_events": ne})

    s_known = series[~mask_nan]
    y_known = y[~mask_nan]

    if is_categorical:
        for cat, grp in y_known.groupby(s_known):
            e = float(grp.sum())
            ne = float((1 - grp).sum())
            rows.append({"bin": str(cat), "events": e, "non_events": ne})
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                labels = pd.qcut(s_known, q=n_bins, duplicates="drop")
            except ValueError:
                labels = pd.cut(s_known, bins=n_bins)
        for interval, grp_y in y_known.groupby(labels, observed=False):
            e = float(grp_y.sum())
            ne = float((1 - grp_y).sum())
            rows.append({"bin": str(interval), "events": e, "non_events": ne})

    tbl = pd.DataFrame(rows)
    tbl["pct_events"] = (tbl["events"] + _CLIP) / (total_events + _CLIP)
    tbl["pct_non_events"] = (tbl["non_events"] + _CLIP) / (total_nonevents + _CLIP)
    tbl["woe"] = np.log(tbl["pct_events"] / tbl["pct_non_events"])
    tbl["iv_bin"] = (tbl["pct_events"] - tbl["pct_non_events"]) * tbl["woe"]
    tbl["feature"] = series.name

    iv = float(tbl["iv_bin"].sum())
    woe_map = dict(zip(tbl["bin"], tbl["woe"]))
    missing_woe = woe_map.pop("__MISSING__", 0.0)

    return _BinStats(
        feature=str(series.name),
        woe_map=woe_map,
        missing_woe=missing_woe,
        iv=iv,
        bin_table=tbl,
    )


def _apply_woe(series: pd.Series, stats: _BinStats, n_bins: int, is_categorical: bool) -> pd.Series:
    """Map a series of raw values to their WoE scores."""
    out = pd.Series(np.nan, index=series.index, name=series.name)

    nan_mask = series.isna()
    out[nan_mask] = stats.missing_woe

    s_known = series[~nan_mask]
    if is_categorical:
        out[~nan_mask] = s_known.astype(str).map(stats.woe_map).fillna(0.0)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                cut = pd.qcut(s_known, q=n_bins, duplicates="drop")
            except ValueError:
                cut = pd.cut(s_known, bins=n_bins)
        out[~nan_mask] = cut.astype(str).map(stats.woe_map).fillna(0.0)

    return out


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Replace each feature with its Weight of Evidence score.

    Parameters
    ----------
    n_bins : int
        Number of quantile bins for continuous features.
    cat_threshold : int
        Columns with at most this many unique values are treated as
        categorical regardless of dtype.

    After fit(), call iv_summary() to get the Information Value table
    and decide which features to keep.
    """

    def __init__(self, n_bins: int = 10, cat_threshold: int = 10):
        self.n_bins = n_bins
        self.cat_threshold = cat_threshold
        self._stats: dict[str, _BinStats] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        y = y.reset_index(drop=True)
        X = X.reset_index(drop=True)
        self._stats = {}
        for col in X.columns:
            is_cat = (
                X[col].dtype == "object"
                or X[col].dtype.name == "category"
                or X[col].nunique() <= self.cat_threshold
            )
            self._stats[col] = _compute_bins(X[col], y, self.n_bins, is_cat)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.reset_index(drop=True)
        out = {}
        for col in X.columns:
            if col not in self._stats:
                out[col] = X[col]
                continue
            stats = self._stats[col]
            is_cat = (
                X[col].dtype == "object"
                or X[col].dtype.name == "category"
                or X[col].nunique() <= self.cat_threshold
            )
            out[col] = _apply_woe(X[col], stats, self.n_bins, is_cat)
        return pd.DataFrame(out)

    def iv_summary(self) -> pd.DataFrame:
        """Return a DataFrame of features ranked by Information Value.

        Columns:
          feature    — column name
          iv         — Information Value (higher = more predictive)
          label      — interpretability label (unpredictive / weak / medium / strong / suspicious)
          n_bins     — number of bins used (after duplicate-edge collapsing)
        """
        rows = [
            {
                "feature": name,
                "iv": stats.iv,
                "label": _iv_label(stats.iv),
                "n_bins": len(stats.woe_map),
            }
            for name, stats in self._stats.items()
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )

    def bin_table(self, feature: str) -> pd.DataFrame:
        """Full WoE/IV diagnostics for one feature — useful for scorecard review."""
        if feature not in self._stats:
            raise KeyError(f"Feature '{feature}' was not seen during fit()")
        cols = ["bin", "events", "non_events", "pct_events", "pct_non_events", "woe", "iv_bin"]
        return self._stats[feature].bin_table[cols].copy()
