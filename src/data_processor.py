"""
Data processing pipeline: Elo ratings, feature engineering, and dataset preparation.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import warnings

warnings.filterwarnings("ignore")

from config import (
    RAW_RESULTS_PATH,
    PROCESSED_DATA_PATH,
    ELO_RATINGS_PATH,
    SCALER_PATH,
    TEAM_ENCODER_PATH,
    ELO_K_FACTOR,
    ELO_HOME_ADVANTAGE,
    ELO_INITIAL,
    RECENT_FORM_WINDOW,
    IMPORTANT_TOURNAMENTS,
)


def load_raw_data(path=None):
    """Load raw match results CSV."""
    path = path or RAW_RESULTS_PATH
    df = pd.read_csv(path, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df)} matches from {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def calculate_elo_ratings(df):
    """
    Calculate Elo ratings for all teams over time.
    Returns a DataFrame with Elo ratings before each match.
    """
    print("Calculating Elo ratings...")
    matches = df.copy()

    # Initialize Elo dictionary
    elo = {}
    home_elo_list = []
    away_elo_list = []
    home_elo_after = []
    away_elo_after = []

    for idx, row in matches.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        neutral = row["neutral"]

        # Get current Elo (or initialize)
        home_elo_before = elo.get(home_team, ELO_INITIAL)
        away_elo_before = elo.get(away_team, ELO_INITIAL)

        home_elo_list.append(home_elo_before)
        away_elo_list.append(away_elo_before)

        # Calculate expected scores
        home_advantage = 0 if neutral else ELO_HOME_ADVANTAGE
        elo_diff_home = home_elo_before - away_elo_before + home_advantage
        elo_diff_away = away_elo_before - home_elo_before - home_advantage

        expected_home = 1 / (1 + 10 ** (-elo_diff_home / 400))
        expected_away = 1 / (1 + 10 ** (-elo_diff_away / 400))

        # Determine match importance
        tournament = str(row.get("tournament", ""))
        importance = 1.0
        for imp_tournament in IMPORTANT_TOURNAMENTS:
            if imp_tournament.lower() in tournament.lower():
                importance = 1.5
                if "World Cup" in tournament and "qualification" not in tournament.lower():
                    importance = 2.0
                break

        # Actual result - skip Elo update if scores are NaN (future matches)
        home_score = row["home_score"]
        away_score = row["away_score"]
        has_result = pd.notna(home_score) and pd.notna(away_score)

        if has_result:
            if home_score > away_score:
                actual_home = 1
                actual_away = 0
            elif home_score < away_score:
                actual_home = 0
                actual_away = 1
            else:
                actual_home = 0.5
                actual_away = 0.5

            # Goal difference multiplier
            goal_diff = abs(home_score - away_score)
            if goal_diff == 0:
                goal_factor = 1.0
            elif goal_diff == 1:
                goal_factor = 1.0
            elif goal_diff == 2:
                goal_factor = 1.5
            else:
                goal_factor = (11 + goal_diff) / 8

            # Update Elo
            k = ELO_K_FACTOR * importance * goal_factor
            new_home_elo = home_elo_before + k * (actual_home - expected_home)
            new_away_elo = away_elo_before + k * (actual_away - expected_away)
        else:
            # No result yet - keep Elo unchanged
            new_home_elo = home_elo_before
            new_away_elo = away_elo_before

        home_elo_after.append(new_home_elo)
        away_elo_after.append(new_away_elo)

        elo[home_team] = new_home_elo
        elo[away_team] = new_away_elo

    matches["home_elo"] = home_elo_list
    matches["away_elo"] = away_elo_list
    matches["home_elo_after"] = home_elo_after
    matches["away_elo_after"] = away_elo_after
    matches["elo_diff"] = matches["home_elo"] - matches["away_elo"]

    # Save final Elo ratings
    final_elo = pd.DataFrame(
        {"team": list(elo.keys()), "elo_rating": list(elo.values())}
    ).sort_values("elo_rating", ascending=False)
    final_elo.to_csv(ELO_RATINGS_PATH, index=False)

    print(f"Elo ratings calculated for {len(elo)} teams")
    print(f"Top 10 teams by Elo:\n{final_elo.head(10)}")

    return matches


def calculate_recent_form(df, window=RECENT_FORM_WINDOW):
    """
    Calculate recent form metrics for each team before each match.
    Uses rolling window of last N matches.
    """
    print("Calculating recent form...")
    matches = df.copy()

    # Store all matches for lookup
    home_form = []
    away_form = []
    home_goals_scored = []
    away_goals_scored = []
    home_goals_conceded = []
    away_goals_conceded = []
    home_win_rate = []
    away_win_rate = []

    # For each team, track their match history
    team_history = {}

    for idx, row in matches.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        # Calculate form for home team
        h_hist = team_history.get(home_team, [])
        h_form = _calc_form_from_history(h_hist, window)
        h_gs, h_gc = _calc_goals_from_history(h_hist, window)
        h_wr = _calc_win_rate(h_hist, window)

        home_form.append(h_form)
        home_goals_scored.append(h_gs)
        home_goals_conceded.append(h_gc)
        home_win_rate.append(h_wr)

        # Calculate form for away team
        a_hist = team_history.get(away_team, [])
        a_form = _calc_form_from_history(a_hist, window)
        a_gs, a_gc = _calc_goals_from_history(a_hist, window)
        a_wr = _calc_win_rate(a_hist, window)

        away_form.append(a_form)
        away_goals_scored.append(a_gs)
        away_goals_conceded.append(a_gc)
        away_win_rate.append(a_wr)

        # Update history after match
        if home_team not in team_history:
            team_history[home_team] = []
        if away_team not in team_history:
            team_history[away_team] = []

        team_history[home_team].append({
            "goals_for": row["home_score"],
            "goals_against": row["away_score"],
            "home": True,
            "date": row["date"],
        })
        team_history[away_team].append({
            "goals_for": row["away_score"],
            "goals_against": row["home_score"],
            "home": False,
            "date": row["date"],
        })

    matches["home_form"] = home_form
    matches["away_form"] = away_form
    matches["home_goals_scored_avg"] = home_goals_scored
    matches["away_goals_scored_avg"] = away_goals_scored
    matches["home_goals_conceded_avg"] = home_goals_conceded
    matches["away_goals_conceded_avg"] = away_goals_conceded
    matches["home_win_rate"] = home_win_rate
    matches["away_win_rate"] = away_win_rate

    return matches


def _calc_form_from_history(history, window):
    """Calculate form score from history (weighted: recent matches more important)."""
    if not history:
        return 0.0
    recent = history[-window:]
    score = 0
    total_weight = 0
    for i, match in enumerate(recent):
        weight = (i + 1) / len(recent)  # Later matches weighted more
        if match["goals_for"] > match["goals_against"]:
            score += 3 * weight
        elif match["goals_for"] == match["goals_against"]:
            score += 1 * weight
        total_weight += weight
    return score / max(total_weight, 1)


def _calc_goals_from_history(history, window):
    """Calculate average goals scored and conceded."""
    if not history:
        return 0.0, 0.0
    recent = history[-window:]
    if not recent:
        return 0.0, 0.0
    gs = np.mean([m["goals_for"] for m in recent])
    gc = np.mean([m["goals_against"] for m in recent])
    return gs, gc


def _calc_win_rate(history, window):
    """Calculate win rate from history."""
    if not history:
        return 0.0
    recent = history[-window:]
    if not recent:
        return 0.0
    wins = sum(1 for m in recent if m["goals_for"] > m["goals_against"])
    return wins / len(recent)


def calculate_h2h_features(df):
    """Calculate head-to-head features from past encounters."""
    print("Calculating head-to-head features...")
    matches = df.copy()

    # Store all past matches for H2H lookup
    past_matches = []
    h2h_home_wins = []
    h2h_away_wins = []
    h2h_draws = []
    h2h_count = []

    for idx, row in matches.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        current_date = row["date"]

        # Find past matches between these teams
        h2h_matches = [
            m
            for m in past_matches
            if (m["home_team"] == home_team and m["away_team"] == away_team)
            or (m["home_team"] == away_team and m["away_team"] == home_team)
        ]

        if h2h_matches:
            h_count = len(h2h_matches)
            h_wins = sum(
                1
                for m in h2h_matches
                if (m["home_team"] == home_team and m["home_score"] > m["away_score"])
                or (m["home_team"] == away_team and m["away_score"] > m["home_score"])
            )
            a_wins = sum(
                1
                for m in h2h_matches
                if (m["home_team"] == away_team and m["home_score"] > m["away_score"])
                or (m["home_team"] == home_team and m["away_score"] > m["home_score"])
            )
            draws = h_count - h_wins - a_wins
        else:
            h_count = 0
            h_wins = 0
            a_wins = 0
            draws = 0

        h2h_count.append(h_count)
        h2h_home_wins.append(h_wins)
        h2h_away_wins.append(a_wins)
        h2h_draws.append(draws)

        past_matches.append(
            {
                "home_team": home_team,
                "away_team": away_team,
                "home_score": row["home_score"],
                "away_score": row["away_score"],
                "date": current_date,
            }
        )

    matches["h2h_count"] = h2h_count
    matches["h2h_home_wins"] = h2h_home_wins
    matches["h2h_away_wins"] = h2h_away_wins
    matches["h2h_draws"] = h2h_draws

    return matches


def engineer_features(df):
    """
    Engineer all features for the model.
    """
    print("Engineering features...")
    df = df.copy()

    # Days since first match (temporal feature)
    df["days_since_first"] = (df["date"] - df["date"].min()).dt.days

    # Year and month
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Neutral venue
    df["is_neutral"] = df["neutral"].astype(int)

    # Elo-based features
    df["elo_advantage_home"] = df["elo_diff"] / 400.0  # Normalized elo difference
    df["elo_quality"] = (df["home_elo"] + df["away_elo"]) / 2 / ELO_INITIAL  # Match quality

    # Form-based features
    df["form_advantage"] = df["home_form"] - df["away_form"]
    df["form_quality"] = (df["home_form"] + df["away_form"]) / 2

    # Goal scoring difference
    df["gs_advantage"] = df["home_goals_scored_avg"] - df["away_goals_scored_avg"]
    df["gc_advantage"] = df["home_goals_conceded_avg"] - df["away_goals_conceded_avg"]

    # Win rate advantage
    df["wr_advantage"] = df["home_win_rate"] - df["away_win_rate"]

    # H2H features
    df["h2h_home_advantage"] = np.where(
        df["h2h_count"] > 0,
        (df["h2h_home_wins"] - df["h2h_away_wins"]) / df["h2h_count"].clip(lower=1),
        0,
    )
    df["has_h2h"] = (df["h2h_count"] > 0).astype(int)

    # Tournament importance
    df["tournament_importance"] = df["tournament"].apply(_get_tournament_importance)

    # Target variable: 0 = away win, 1 = draw, 2 = home win
    df["result"] = np.where(
        df["home_score"] > df["away_score"], 2,
        np.where(df["home_score"] == df["away_score"], 1, 0)
    )

    return df


def _get_tournament_importance(tournament):
    """Rate tournament importance 0-3."""
    t = str(tournament).lower()
    if "fifa world cup" in t and "qualification" not in t:
        return 3
    elif "fifa world cup qualification" in t:
        return 2
    elif any(x in t for x in ["uefa euro", "copa américa", "african cup", "asian cup", "gold cup"]):
        return 2
    elif any(x in t for x in ["nations league", "confederations"]):
        return 1
    elif "friendly" in t:
        return 0
    else:
        return 1


def create_target_labels(df):
    """Create target labels for classification."""
    # result column already created in engineer_features
    return df


def prepare_dataset(df, min_date=None, max_date=None):
    """
    Prepare the final dataset for model training.
    Returns feature matrix X and target vector y.
    """
    if min_date:
        df = df[df["date"] >= min_date]
    if max_date:
        df = df[df["date"] <= max_date]

    # Select feature columns
    feature_cols = [
        "elo_advantage_home",
        "elo_quality",
        "form_advantage",
        "form_quality",
        "gs_advantage",
        "gc_advantage",
        "wr_advantage",
        "h2h_home_advantage",
        "has_h2h",
        "is_neutral",
        "tournament_importance",
        "days_since_first",
        "year",
        "month",
    ]

    X = df[feature_cols].values.astype(np.float32)
    y = df["result"].values.astype(np.int64)

    return X, y, feature_cols, df


def preprocess_pipeline():
    """
    Run the full preprocessing pipeline:
    1. Load raw data
    2. Calculate Elo ratings
    3. Calculate recent form
    4. Calculate H2H features
    5. Engineer features
    6. Save processed data
    """
    print("=" * 60)
    print("Running preprocessing pipeline...")
    print("=" * 60)

    # Step 1: Load raw data
    df = load_raw_data()

    # Step 2: Calculate Elo ratings
    df = calculate_elo_ratings(df)

    # Step 3: Calculate recent form
    df = calculate_recent_form(df)

    # Step 4: Calculate H2H features
    df = calculate_h2h_features(df)

    # Step 5: Engineer features
    df = engineer_features(df)

    # Step 6: Fit and save team encoder
    all_teams = pd.concat([df["home_team"], df["away_team"]]).unique()
    team_encoder = LabelEncoder()
    team_encoder.fit(all_teams)

    # Encode team IDs
    df["home_team_id"] = team_encoder.transform(df["home_team"])
    df["away_team_id"] = team_encoder.transform(df["away_team"])

    with open(TEAM_ENCODER_PATH, "wb") as f:
        pickle.dump(team_encoder, f)

    # Step 7: Fit scaler on features
    X, y, feature_cols, _ = prepare_dataset(df)
    scaler = StandardScaler()
    scaler.fit(X)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # Step 8: Save processed data
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"\nProcessed data saved to {PROCESSED_DATA_PATH}")
    print(f"Total samples: {len(df)}")
    print(f"Features: {feature_cols}")
    print(
        f"Label distribution: Home Win={sum(y==2)} ({sum(y==2)/len(y)*100:.1f}%), "
        f"Draw={sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%), "
        f"Away Win={sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)"
    )

    return df, scaler, team_encoder


if __name__ == "__main__":
    preprocess_pipeline()
