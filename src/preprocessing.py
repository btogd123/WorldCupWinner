"""
Data preparation: GaussRank + team encoding → tensor tuples.

Depends on features.FEATURE_COLS for the canonical feature list.
"""

import numpy as np
import pandas as pd
from scipy.special import erfinv

from features import FEATURE_COLS


class GaussRankTransformer:
    """Rank-based Gaussian transformation — non-parametric, handles any scale.

    For each continuous feature, maps values → empirical percentiles →
    standard normal quantiles. Binary features are passed through unchanged.

    Fit on train, transform val/test. Out-of-range values are clipped.
    """

    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
        self.bins = {}       # col_idx -> (sorted_values, percentiles)
        self.binary_cols = []

    def fit(self, X, binary_cols=None):
        self.binary_cols = binary_cols or []
        self.n_cols = X.shape[1]
        for col in range(self.n_cols):
            if col in self.binary_cols:
                continue
            col_data = X[:, col]
            valid = col_data[~np.isnan(col_data)]
            if len(valid) == 0:
                continue
            sorted_vals = np.sort(valid)
            n = len(sorted_vals)
            # Midpoint percentiles: (i + 0.5) / n handles ties gracefully
            percentiles = (np.arange(n) + 0.5) / n
            self.bins[col] = (sorted_vals.astype(np.float64), percentiles.astype(np.float64))

    def transform(self, X):
        X_out = X.copy().astype(np.float64)
        for col in range(self.n_cols):
            if col in self.binary_cols:
                continue
            if col not in self.bins:
                continue
            sorted_vals, percentiles = self.bins[col]
            col_data = X[:, col]
            # Interpolate rank percentile, clip out-of-range
            ranks = np.interp(col_data, sorted_vals, percentiles,
                              left=self.epsilon, right=1.0 - self.epsilon)
            X_out[:, col] = np.sqrt(2) * erfinv(2.0 * ranks - 1.0)
        return X_out.astype(np.float32)

    def fit_transform(self, X, binary_cols=None):
        self.fit(X, binary_cols=binary_cols)
        return self.transform(X)


# Columns that are binary and should not be GaussRank-ed
BINARY_FEATURE_COLS = {
    "is_neutral", "is_wc", "is_wcq", "is_continental", "is_friendly",
    "has_h2h", "low_scoring_tendency", "mismatch_flag",
}


def prepare_enhanced_data(df, scaler, team_encoder, fit_scaler=False, feature_cols=None):
    """Prepare enhanced feature tensors.

    Args:
        df: DataFrame with all feature columns
        scaler: GaussRankTransformer (or None if fit_scaler=True)
        team_encoder: LabelEncoder for team names
        fit_scaler: if True, create and fit a new GaussRankTransformer
        feature_cols: list of feature column names (defaults to FEATURE_COLS)

    Returns:
        (X, home_ids, away_ids, y, home_goals, away_goals, feature_cols, scaler)
        where X has is_neutral preserved as raw binary (0/1).
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    X = df[feature_cols].fillna(0).values.astype(np.float32)

    if fit_scaler:
        binary_indices = [i for i, c in enumerate(feature_cols) if c in BINARY_FEATURE_COLS]
        scaler = GaussRankTransformer()
        X = scaler.fit_transform(X, binary_cols=binary_indices)
    else:
        X = scaler.transform(X)

    # Keep is_neutral as raw binary (0/1) — scaling breaks neutral-venue gating
    is_neutral_idx = feature_cols.index("is_neutral")
    X[:, is_neutral_idx] = df["is_neutral"].fillna(0).values.astype(np.float32)

    home_ids = df["home_team_id"].values.astype(np.int64) + 1
    away_ids = df["away_team_id"].values.astype(np.int64) + 1
    y = df["result"].values.astype(np.int64)
    home_goals = df["home_score"].values.astype(np.float32)
    away_goals = df["away_score"].values.astype(np.float32)

    if fit_scaler:
        return X, home_ids, away_ids, y, home_goals, away_goals, feature_cols, scaler
    return X, home_ids, away_ids, y, home_goals, away_goals, feature_cols, None


def split_data_improved(df, train_end="2021-12-31", val_end="2023-12-31", val_start=None):
    """Split data chronologically with optional gap between train and val.

    Standard split:           train = [2000, train_end), val = [train_end, val_end)
    Gap split (val_start):    train = [2000, train_end) U [val_start, val_end)
                              val   = [train_end, val_start)
    Test is always [val_end, ...).

    Plan C (deployment): train_end="2025-07-01", val_start="2026-01-01", val_end="2026-06-17"
    """
    df = df[df["date"] >= pd.to_datetime("2000-01-01")]
    df = df[df["date"] <= pd.to_datetime("2026-06-16")]
    df = df.dropna(subset=["home_score", "away_score"])

    if val_start is not None:
        # Gap split: train has two blocks with a val gap in between
        train_before = df[df["date"] < pd.to_datetime(train_end)]
        train_after = df[
            (df["date"] >= pd.to_datetime(val_start))
            & (df["date"] < pd.to_datetime(val_end))
        ]
        train = pd.concat([train_before, train_after])
        val = df[
            (df["date"] >= pd.to_datetime(train_end))
            & (df["date"] < pd.to_datetime(val_start))
        ]
    else:
        train = df[df["date"] < pd.to_datetime(train_end)]
        val = df[
            (df["date"] >= pd.to_datetime(train_end))
            & (df["date"] < pd.to_datetime(val_end))
        ]

    test = df[df["date"] >= pd.to_datetime(val_end)]

    print(f"\nSplit: Train={len(train)} ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"       Val={len(val)}   ({val['date'].min().date()} to {val['date'].max().date()})")
    print(f"       Test={len(test)}  ({test['date'].min().date()} to {test['date'].max().date()})")

    wcq = test[test["tournament"].str.contains("qualification", na=False)]
    print(f"       WCQ in test: {len(wcq)}")

    return train, val, test
