"""
Feature engineering for match prediction.
Pure functions, no PyTorch dependency.

Canonical 35-feature list — single source of truth for all scripts.
"""

import numpy as np
import pandas as pd

# Canonical 35-feature list (10 groups)
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
    # Elo-similarity (4)
    "sim_wr_advantage",
    "sim_gs_advantage",
    "sim_wr_quality",
    "sim_dr_quality",
    # Draw-specific (5) — Hvattum 2017
    "draw_rate_home",
    "draw_rate_away",
    "both_draw_prone",
    "defensive_similarity",
    "low_scoring_tendency",
    # Positional strength — attack/defense (6)
    "home_att_vs_away_def",
    "away_att_vs_home_def",
    "attack_balance",
    "scoring_potential",
    "defensive_strength",
    "mismatch_flag",
    # Context (5)
    "is_neutral",
    "is_wc",
    "is_wcq",
    "is_continental",
    "is_friendly",
]

# Subset of FEATURE_COLS that are positional attack/defense matchup features
POSITIONAL_FEATURE_COLS = [
    "home_att_vs_away_def",
    "away_att_vs_home_def",
    "attack_balance",
    "scoring_potential",
    "defensive_strength",
    "mismatch_flag",
]

# FEATURE_COLS without positional features (29 features — baseline)
NON_POSITIONAL_FEATURE_COLS = [
    c for c in FEATURE_COLS if c not in POSITIONAL_FEATURE_COLS
]


def compute_positional_features(df):
    """
    Compute iterative opponent-corrected attack/defense ratings for all teams.

    Processes matches chronologically. For each match, stores pre-match ratings,
    then updates based on actual vs expected goals.

    Returns df with columns: home_att, home_def, away_att, away_def,
    and 6 matchup features derived from them.
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    total_goals = df["home_score"].sum() + df["away_score"].sum()
    n_matches = len(df)
    league_avg = total_goals / (2 * n_matches)

    att = {}
    def_rat = {}

    lr = 0.02
    regress = 0.995  # per-match mean reversion toward 1.0

    home_att_vals = [None] * len(df)
    home_def_vals = [None] * len(df)
    away_att_vals = [None] * len(df)
    away_def_vals = [None] * len(df)

    for idx, row in df.iterrows():
        h = row["home_team"]
        a = row["away_team"]
        hg = row["home_score"]
        ag = row["away_score"]

        for t in [h, a]:
            if t not in att:
                att[t] = 1.0
                def_rat[t] = 1.0

        home_att_vals[idx] = att[h]
        home_def_vals[idx] = def_rat[h]
        away_att_vals[idx] = att[a]
        away_def_vals[idx] = def_rat[a]

        if pd.isna(hg) or pd.isna(ag):
            continue

        exp_home = league_avg * att[h] / max(def_rat[a], 0.3)
        exp_away = league_avg * att[a] / max(def_rat[h], 0.3)

        att[h] *= 1.0 + lr * (hg - exp_home) / max(exp_home, 0.5)
        att[a] *= 1.0 + lr * (ag - exp_away) / max(exp_away, 0.5)

        def_rat[h] *= 1.0 + lr * (exp_away - ag) / max(exp_away, 0.5)
        def_rat[a] *= 1.0 + lr * (exp_home - hg) / max(exp_home, 0.5)

        for t in [h, a]:
            att[t] = max(0.3, min(3.0, att[t]))
            def_rat[t] = max(0.3, min(3.0, def_rat[t]))
            att[t] = 1.0 + regress * (att[t] - 1.0)
            def_rat[t] = 1.0 + regress * (def_rat[t] - 1.0)

    df["home_att"] = home_att_vals
    df["home_def"] = home_def_vals
    df["away_att"] = away_att_vals
    df["away_def"] = away_def_vals

    home_threat = df["home_att"] / df["away_def"].clip(0.3)
    away_threat = df["away_att"] / df["home_def"].clip(0.3)

    df["home_att_vs_away_def"] = home_threat
    df["away_att_vs_home_def"] = away_threat
    df["attack_balance"] = np.log(home_threat.clip(0.1)) - np.log(away_threat.clip(0.1))
    df["scoring_potential"] = df["home_att"] * df["away_att"]
    df["defensive_strength"] = df["home_def"] * df["away_def"]
    df["mismatch_flag"] = (abs(df["attack_balance"]) > 0.5).astype(float)

    print(f"Positional features computed. Teams rated: {len(att)}")
    print(f"  att stats: mean={np.mean(list(att.values())):.3f}, "
          f"std={np.std(list(att.values())):.3f}, "
          f"range=[{min(att.values()):.3f}, {max(att.values()):.3f}]")
    print(f"  def stats: mean={np.mean(list(def_rat.values())):.3f}, "
          f"std={np.std(list(def_rat.values())):.3f}, "
          f"range=[{min(def_rat.values()):.3f}, {max(def_rat.values()):.3f}]")

    return df


def feature_engineering_v2(df):
    """Enhanced feature engineering — raw natural units for GaussRank."""
    print("Engineering enhanced features (raw units for GaussRank)...")
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
    df["defensive_similarity"] = abs(df["home_goals_conceded_avg"] - df["away_goals_conceded_avg"])
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
    df["sim_wr_quality"] = (df["home_sim_win_rate"] + df["away_sim_win_rate"]) / 2
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
