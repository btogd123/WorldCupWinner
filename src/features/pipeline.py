"""
Data processing pipeline: Elo ratings, feature engineering, and dataset preparation.
"""
import pandas as pd
import numpy as np
import random
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings

warnings.filterwarnings("ignore")

from config import (
    RAW_RESULTS_PATH,
    PROCESSED_DATA_PATH,
    ELO_RATINGS_PATH,
    TEAM_ENCODER_PATH,
    ELO_K_FACTOR,
    ELO_HOME_ADVANTAGE,
    ELO_INITIAL,
    RECENT_FORM_WINDOW,
    IMPORTANT_TOURNAMENTS,
)
from features.builder import feature_engineering_v2


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
    home_draw_rate = []
    away_draw_rate = []

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
        h_dr = _calc_draw_rate(h_hist, window)
        a_dr = _calc_draw_rate(a_hist, window)

        away_form.append(a_form)
        away_goals_scored.append(a_gs)
        away_goals_conceded.append(a_gc)
        away_win_rate.append(a_wr)
        home_draw_rate.append(h_dr)
        away_draw_rate.append(a_dr)

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
    matches["home_draw_rate"] = home_draw_rate
    matches["away_draw_rate"] = away_draw_rate

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


def _calc_draw_rate(history, window):
    """Calculate draw rate from history."""
    if not history:
        return 0.0
    recent = history[-window:]
    if not recent:
        return 0.0
    draws = sum(1 for m in recent if m["goals_for"] == m["goals_against"])
    return draws / len(recent)


def calculate_elo_similarity_features(df, elo_delta=125, max_matches=10):
    """Calculate Elo-similarity per-team statistics (no home/away restriction).

    For each match, finds prior matches where a team with Elo ~= current home Elo
    faced a team with Elo ~= current away Elo, regardless of who was home.
    Two matching modes:
      mask1: historical home ~= current home AND historical away ~= current away
      mask2: historical home ~= current away AND historical away ~= current home (swapped)

    Stats computed from the perspective of the Elo-matched team.
    """
    print(f"Calculating Elo-similarity features (delta=+/-{elo_delta}, max={max_matches})...")
    matches = df.copy()
    N = len(matches)

    new_cols = [
        "home_sim_win_rate", "home_sim_draw_rate", "home_sim_gs", "home_sim_gc",
        "away_sim_win_rate", "away_sim_draw_rate", "away_sim_gs", "away_sim_gc",
    ]
    for col in new_cols:
        matches[col] = 0.0

    home_elo = matches["home_elo"].values.astype(np.float64)
    away_elo = matches["away_elo"].values.astype(np.float64)
    home_score = matches["home_score"].values
    away_score = matches["away_score"].values

    has_result = pd.notna(home_score) & pd.notna(away_score)

    n_matched = 0
    for i in range(1, N):
        h_elo_i = home_elo[i]
        a_elo_i = away_elo[i]

        h_elo_prior = home_elo[:i]
        a_elo_prior = away_elo[:i]
        has_res_prior = has_result[:i]

        mask1 = (
            (np.abs(h_elo_prior - h_elo_i) <= elo_delta)
            & (np.abs(a_elo_prior - a_elo_i) <= elo_delta)
            & has_res_prior
        )
        mask2 = (
            (np.abs(h_elo_prior - a_elo_i) <= elo_delta)
            & (np.abs(a_elo_prior - h_elo_i) <= elo_delta)
            & has_res_prior
        )

        mask = mask1 | mask2
        prior_idx = np.where(mask)[0]

        if len(prior_idx) == 0:
            continue

        n_matched += 1
        recent = prior_idx[-max_matches:]
        n_rec = len(recent)

        hs = home_score[recent].astype(np.float64)
        aws = away_score[recent].astype(np.float64)
        m1 = mask1[recent]
        m2 = mask2[recent]

        # Home perspective (team with Elo ~= current home Elo):
        home_wins = ((m1 & (hs > aws)) | (m2 & (aws > hs))).sum()
        home_draws = ((m1 | m2) & (hs == aws)).sum()
        home_gs = (m1 * hs + m2 * aws).sum()
        home_gc = (m1 * aws + m2 * hs).sum()

        matches.loc[matches.index[i], "home_sim_win_rate"] = home_wins / n_rec
        matches.loc[matches.index[i], "home_sim_draw_rate"] = home_draws / n_rec
        matches.loc[matches.index[i], "home_sim_gs"] = home_gs / n_rec
        matches.loc[matches.index[i], "home_sim_gc"] = home_gc / n_rec

        # Away perspective (team with Elo ~= current away Elo):
        away_wins = ((m1 & (aws > hs)) | (m2 & (hs > aws))).sum()
        away_draws = home_draws
        away_gs = (m1 * aws + m2 * hs).sum()
        away_gc = (m1 * hs + m2 * aws).sum()

        matches.loc[matches.index[i], "away_sim_win_rate"] = away_wins / n_rec
        matches.loc[matches.index[i], "away_sim_draw_rate"] = away_draws / n_rec
        matches.loc[matches.index[i], "away_sim_gs"] = away_gs / n_rec
        matches.loc[matches.index[i], "away_sim_gc"] = away_gc / n_rec

    print(f"Elo-similarity features: {n_matched}/{N} matches matched ({n_matched/N*100:.1f}%)")
    return matches


def create_wc2026_schedule(team_encoder):
    """Create WC 2026 schedule manually if not in dataset.

    WC 2026 has 48 teams in groups of 4 (12 groups).
    Top 2 from each group + 8 best 3rd place advance.
    """
    from utils import set_seed

    qualified_teams = [
        # Hosts
        "United States", "Mexico", "Canada",
        # AFC
        "Japan", "South Korea", "Saudi Arabia", "Iran", "Australia",
        "Qatar", "United Arab Emirates", "Iraq",
        # CAF
        "Morocco", "Senegal", "Egypt", "Algeria", "Nigeria",
        "Cameroon", "Ghana", "Ivory Coast", "Tunisia",
        # CONCACAF
        "Costa Rica", "Panama", "Jamaica", "Honduras",
        # CONMEBOL
        "Argentina", "Brazil", "Uruguay", "Colombia", "Ecuador",
        "Peru", "Chile", "Paraguay",
        # OFC
        "New Zealand",
        # UEFA
        "France", "Spain", "England", "Germany", "Portugal",
        "Netherlands", "Italy", "Belgium", "Croatia", "Denmark",
        "Switzerland", "Austria", "Serbia", "Ukraine", "Turkey",
        "Sweden",
    ]

    valid_teams = []
    for team in qualified_teams:
        if team in team_encoder.classes_:
            valid_teams.append(team)
        else:
            print(f"Warning: {team} not found in team encoder, skipping")

    print(f"Valid teams for WC 2026: {len(valid_teams)}/{len(qualified_teams)}")

    if len(valid_teams) < 32:
        print("Not enough valid teams to create meaningful schedule")
        return None

    set_seed(42)

    teams = valid_teams[:48] if len(valid_teams) >= 48 else valid_teams
    random.shuffle(teams)

    groups = []
    for i in range(0, min(48, len(teams)), 4):
        if i + 3 < len(teams):
            groups.append(teams[i : i + 4])

    matches = []
    match_date = pd.to_datetime("2026-06-11")

    for group_idx, group in enumerate(groups):
        for i in range(4):
            for j in range(i + 1, 4):
                if (i + j) % 2 == 0:
                    home_team, away_team = group[i], group[j]
                else:
                    home_team, away_team = group[j], group[i]

                matches.append({
                    "date": match_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                })

        match_date += pd.Timedelta(days=1)

    schedule_df = pd.DataFrame(matches)
    print(f"Created schedule with {len(schedule_df)} group matches across {len(groups)} groups")
    return schedule_df


def preprocess_pipeline():
    """
    Run the full preprocessing pipeline:
    1. Load raw data
    2. Calculate Elo ratings
    3. Calculate recent form
    4. Calculate Elo-similarity features
    5. Engineer features
    6. Fit and save team encoder
    7. Save processed data
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

    # Step 4: Calculate Elo-similarity features
    df = calculate_elo_similarity_features(df, elo_delta=125)

    # Step 5: Engineer features
    df = feature_engineering_v2(df)

    # Step 6: Fit and save team encoder
    all_teams = pd.concat([df["home_team"], df["away_team"]]).unique()
    team_encoder = LabelEncoder()
    team_encoder.fit(all_teams)

    # Encode team IDs
    df["home_team_id"] = team_encoder.transform(df["home_team"])
    df["away_team_id"] = team_encoder.transform(df["away_team"])

    with open(TEAM_ENCODER_PATH, "wb") as f:
        pickle.dump(team_encoder, f)

    # Step 7: Save processed data
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    home_wins = (df["result"] == 2).sum()
    draws = (df["result"] == 1).sum()
    away_wins = (df["result"] == 0).sum()
    n = len(df)
    print(f"\nProcessed data saved to {PROCESSED_DATA_PATH}")
    print(f"Total samples: {n}")
    print(
        f"Label distribution: Home Win={home_wins} ({home_wins/n*100:.1f}%), "
        f"Draw={draws} ({draws/n*100:.1f}%), "
        f"Away Win={away_wins} ({away_wins/n*100:.1f}%)"
    )

    return df, team_encoder


if __name__ == "__main__":
    preprocess_pipeline()
