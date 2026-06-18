"""
Data preparation: StandardScaler + team encoding → tensor tuples.

Depends on features.FEATURE_COLS for the canonical feature list.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLS


def prepare_enhanced_data(df, scaler, team_encoder, fit_scaler=False, feature_cols=None):
    """Prepare enhanced feature tensors.

    Args:
        df: DataFrame with all feature columns
        scaler: StandardScaler (or None if fit_scaler=True)
        team_encoder: LabelEncoder for team names
        fit_scaler: if True, create and fit a new StandardScaler
        feature_cols: list of feature column names (defaults to FEATURE_COLS)

    Returns:
        (X, home_ids, away_ids, y, home_goals, away_goals, feature_cols, scaler)
        where X has is_neutral preserved as raw binary (0/1).
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    X = df[feature_cols].fillna(0).values.astype(np.float32)

    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
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
