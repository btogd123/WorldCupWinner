"""
Feature engineering for match prediction.
Pure functions, no PyTorch dependency.

Canonical 21-feature list (5 groups) — single source of truth for all scripts.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    # Elo-based (4)
    "elo_diff",
    "elo_quality",
    "elo_ratio",
    "abs_elo_diff",
    # Form-based (3)
    "form_advantage",
    "form_quality",
    "wr_advantage",
    # Elo-similarity (3)
    "sim_wr_advantage",
    "sim_gs_advantage",
    "sim_dr_quality",
    # Draw-specific (6) — Hvattum 2017
    "draw_rate_home",
    "draw_rate_away",
    "both_draw_prone",
    "defensive_similarity",
    "strength_parity",
    "low_scoring_tendency",
    # Context (5)
    "is_neutral",
    "is_wc",
    "is_wcq",
    "is_continental",
    "is_friendly",
]

# Columns that are binary (0/1) and should not be scaled
BINARY_FEATURE_COLS = {
    "is_neutral", "is_wc", "is_wcq", "is_continental", "is_friendly",
    "low_scoring_tendency",
}


def feature_engineering_v2(df):
    """Enhanced feature engineering."""
    print("Engineering enhanced features...")
    df = df.copy()

    df["elo_diff"] = df["elo_diff"]
    df["elo_quality"] = (df["home_elo"] + df["away_elo"]) / 2
    df["elo_ratio"] = df["home_elo"] / df["away_elo"]
    df["abs_elo_diff"] = abs(df["elo_diff"])

    # Form-based features
    df["form_advantage"] = df["home_form"] - df["away_form"]
    df["form_quality"] = (df["home_form"] + df["away_form"]) / 2
    df["wr_advantage"] = df["home_win_rate"] - df["away_win_rate"]

    # Draw-specific features (TheDrawCode / Hvattum 2017 original formulas)
    df["draw_rate_home"] = df["home_draw_rate"]
    df["draw_rate_away"] = df["away_draw_rate"]
    df["both_draw_prone"] = np.minimum(df["home_draw_rate"], df["away_draw_rate"])
    df["defensive_similarity"] = 1.0 / (1.0 + abs(df["home_goals_conceded_avg"] - df["away_goals_conceded_avg"]))
    df["strength_parity"] = 1.0 / (1.0 + abs(df["elo_diff"]) / 100.0)
    df["low_scoring_tendency"] = (
        (df["home_goals_scored_avg"] + df["home_goals_conceded_avg"] < 2.5)
        & (df["away_goals_scored_avg"] + df["away_goals_conceded_avg"] < 2.5)
    ).astype(float)

    # Elo-similarity advantage features
    df["sim_wr_advantage"] = df["home_sim_win_rate"] - df["away_sim_win_rate"]
    df["sim_gs_advantage"] = df["home_sim_gs"] - df["away_sim_gs"]
    df["sim_dr_quality"] = np.minimum(df["home_sim_draw_rate"], df["away_sim_draw_rate"])

    # Tournament type encoding
    df["is_wc"] = df["tournament"].str.contains("FIFA World Cup", na=False).astype(int)
    df["is_wcq"] = df["tournament"].str.contains("qualification", na=False).astype(int)
    df["is_friendly"] = df["tournament"].str.contains("Friendly", na=False).astype(int)
    df["is_continental"] = (
        df["tournament"].str.contains(
            "UEFA Euro|Copa Am|African Cup|Asian Cup|Gold Cup|Nations League",
            na=False,
        )
    ).astype(int)

    # Neutral venue
    df["is_neutral"] = df["neutral"].astype(int)

    # Target
    df["result"] = np.where(
        df["home_score"] > df["away_score"],
        2,
        np.where(df["home_score"] == df["away_score"], 1, 0),
    )

    return df


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
    binary_indices = [i for i, c in enumerate(feature_cols) if c in BINARY_FEATURE_COLS]

    if fit_scaler:
        scaler = StandardScaler()
        # Zero out binary cols before fitting — they'll be restored as-is
        X[:, binary_indices] = 0
        X = scaler.fit_transform(X).astype(np.float32)
        for i in binary_indices:
            X[:, i] = df[feature_cols[i]].fillna(0).values.astype(np.float32)
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
