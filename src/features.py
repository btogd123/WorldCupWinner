"""
Feature engineering for match prediction.
Pure functions, no PyTorch dependency.

Canonical 26-feature list (7 groups) — single source of truth for all scripts.
"""

import numpy as np
import pandas as pd

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
    # Goal-based (3)
    "gs_advantage",
    "gc_advantage",
    "goal_diff_advantage",
    # H2H (2)
    "h2h_dominance",
    "has_h2h",
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

def feature_engineering_v2(df):
    """Enhanced feature engineering."""
    print("Engineering enhanced features...")
    df = df.copy()

    df["elo_diff"] = df["elo_diff"]
    df["elo_quality"] = (df["home_elo"] + df["away_elo"]) / 2
    df["elo_ratio"] = df["home_elo"] / df["away_elo"]
    df["abs_elo_diff"] = abs(df["elo_diff"])

    # Goal difference-based features
    df["home_goal_diff_avg"] = df["home_goals_scored_avg"] - df["home_goals_conceded_avg"]
    df["away_goal_diff_avg"] = df["away_goals_scored_avg"] - df["away_goals_conceded_avg"]
    df["goal_diff_advantage"] = df["home_goal_diff_avg"] - df["away_goal_diff_avg"]

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

    # H2H enhanced
    df["h2h_dominance"] = np.where(
        df["h2h_count"] >= 3,
        (df["h2h_home_wins"] - df["h2h_away_wins"]) / df["h2h_count"],
        0,
    )

    # Elo-similarity advantage features
    df["sim_wr_advantage"] = df["home_sim_win_rate"] - df["away_sim_win_rate"]
    df["sim_gs_advantage"] = df["home_sim_gs"] - df["away_sim_gs"]
    df["sim_dr_quality"] = (df["home_sim_draw_rate"] + df["away_sim_draw_rate"]) / 2

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
