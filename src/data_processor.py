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


def calculate_elo_similarity_features(df, elo_delta=75, max_matches=10):
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


# --- Group Context (Fighting Spirit) Features ---


def _get_edition_key(tournament, year):
    """Map tournament name + year to a unique edition key.

    Returns e.g. "WC2022", "WCQ2022", "EURO2024", "COPA2024".
    Returns None for unsupported or non-group tournaments.
    """
    t = str(tournament).lower()
    y = int(year)

    if "fifa world cup" in t and "qualification" not in t:
        return f"WC{y}"
    if "fifa world cup" in t and "qualification" in t:
        return f"WCQ{y}"
    if "uefa euro" in t and "qualification" not in t:
        return f"EURO{y}"
    if "uefa euro" in t and "qualification" in t:
        return f"EUROQ{y}"
    if "copa am" in t and "qualification" not in t:
        return f"COPA{y}"
    if "african cup" in t and "qualification" not in t:
        return f"AFCON{y}"
    if "african cup" in t and "qualification" in t:
        return f"AFCONQ{y}"
    if "asian cup" in t and "qualification" not in t:
        return f"ASIANCUP{y}"
    if "asian cup" in t and "qualification" in t:
        return f"ASIANCUPQ{y}"
    if "gold cup" in t and "qualification" not in t:
        return f"GOLDCUP{y}"

    return None


def _is_final_tournament(edition_key):
    return "Q" not in edition_key


def _detect_groups_final(edition_df):
    """Detect groups in final tournaments using first-3-opponents heuristic.

    In 4-team groups with single round-robin, each team plays exactly 3
    group matches. The first 3 unique opponents form the group.
    """
    matches_by_date = edition_df.sort_values("date")
    team_opponents = {}

    for _, row in matches_by_date.iterrows():
        home, away = row["home_team"], row["away_team"]
        team_opponents.setdefault(home, []).append(away)
        team_opponents.setdefault(away, []).append(home)

    raw_groups = {}
    for team, opponents in team_opponents.items():
        raw_groups[team] = frozenset([team] + opponents[:3])

    unique_groups = set(raw_groups.values())
    return [g for g in unique_groups if len(g) == 4]


def _detect_groups_qualifier(edition_df):
    """Detect groups in qualifiers via BFS connected components.

    Qualifiers use double round-robin: teams in the same group play
    each other home & away. Skip CONMEBOL single-league (>=10 teams).
    """
    adjacency = {}
    for _, row in edition_df.iterrows():
        home, away = row["home_team"], row["away_team"]
        adjacency.setdefault(home, set()).add(away)
        adjacency.setdefault(away, set()).add(home)

    all_teams = set(adjacency.keys())
    visited = set()
    components = []

    for team in all_teams:
        if team in visited:
            continue
        component = set()
        queue = [team]
        while queue:
            current = queue.pop(0)
            if current in component:
                continue
            component.add(current)
            visited.add(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in component:
                    queue.append(neighbor)
        components.append(component)

    return [frozenset(c) for c in components if len(c) < 10]


def _calculate_standings(group_teams, prev_matches_df):
    """Calculate group standings from completed matches.

    Returns dict: team -> {pts, gd, gf, mp}
    """
    standings = {t: {"pts": 0, "gd": 0, "gf": 0, "mp": 0} for t in group_teams}

    for _, row in prev_matches_df.iterrows():
        home, away = row["home_team"], row["away_team"]
        if home not in group_teams or away not in group_teams:
            continue
        hs, aws = row["home_score"], row["away_score"]
        if pd.isna(hs) or pd.isna(aws):
            continue
        hs, aws = int(hs), int(aws)

        standings[home]["mp"] += 1
        standings[away]["mp"] += 1
        standings[home]["gf"] += hs
        standings[away]["gf"] += aws
        standings[home]["gd"] += hs - aws
        standings[away]["gd"] += aws - hs

        if hs > aws:
            standings[home]["pts"] += 3
        elif hs < aws:
            standings[away]["pts"] += 3
        else:
            standings[home]["pts"] += 1
            standings[away]["pts"] += 1

    return standings


def _check_status(standings, team, current_opponent, remaining_matches, num_advancing):
    """Determine if a team is already qualified or must win to survive.

    is_already_qualified: fewer than num_advancing teams can surpass
        the team's CURRENT points with their max possible.
    must_win_to_survive: a draw mathematically eliminates but a win
        keeps the team alive (and not already qualified).

    current_opponent: the other team in this match, so its max possible
        includes the 3 points it could earn from this fixture.
    """
    if team not in standings:
        return 0, 0

    team_pts = standings[team]["pts"]
    n_remaining = sum(1 for h, a in remaining_matches if team in (h, a))

    # Max possible points for every other team.
    # remaining_matches only contains future matches (li > mi), so we add 1
    # to the opponent's remaining count to account for this match.
    other_max = {}
    for t, s in standings.items():
        if t == team:
            continue
        r = sum(1 for h, a in remaining_matches if t in (h, a))
        if t == current_opponent:
            r += 1  # this match is not in remaining_matches
        other_max[t] = s["pts"] + 3 * r

    if not other_max:
        return 1, 0

    sorted_max = sorted(other_max.values(), reverse=True)
    cutoff_idx = min(num_advancing - 1, len(sorted_max) - 1)
    cutoff = sorted_max[cutoff_idx]
    is_qualified = 1 if team_pts > cutoff else 0

    if is_qualified:
        return 1, 0

    # Must win: max with draw < cutoff_pts, max with win >= cutoff_pts.
    # Only check when there are remaining matches to play.
    if n_remaining == 0:
        return 0, 0

    # n_remaining doesn't include this match, so total remaining after this
    # (excluding this match) = n_remaining (all are future).
    after = n_remaining
    max_with_draw = team_pts + 1 + 3 * after
    max_with_win = team_pts + 3 + 3 * after

    # cutoff from max-possible standings (same logic as qualification check)
    cutoff_pts = sorted_max[cutoff_idx]

    must_win = 1 if (max_with_draw < cutoff_pts and max_with_win >= cutoff_pts) else 0
    return 0, must_win


def calculate_group_context(df):
    """Add fighting-spirit features: group_round, is_already_qualified, must_win_to_survive.

    Two-pass approach:
      1. Identify editions and detect groups per edition (no look-ahead)
      2. Process each group chronologically, computing standings from
         completed matches only, then determining status before each match.

    Supports: WC finals, Euro, Copa America, AFCON, Asian Cup, Gold Cup,
             WC qualifiers, and continental qualifiers (except CONMEBOL).
    """
    df = df.copy()

    # Default columns
    df["group_round"] = 1
    df["is_already_qualified_home"] = 0
    df["is_already_qualified_away"] = 0
    df["must_win_to_survive_home"] = 0
    df["must_win_to_survive_away"] = 0

    # ---- PASS 1: Detect groups per edition ----
    edition_masks = {}
    for idx, row in df.iterrows():
        key = _get_edition_key(row["tournament"], row["date"].year)
        if key is None:
            continue
        if key not in edition_masks:
            edition_masks[key] = []
        edition_masks[key].append(idx)

    edition_groups = {}
    for key, indices in edition_masks.items():
        edition_df = df.loc[indices]
        groups = _detect_groups_final(edition_df) if _is_final_tournament(key) else _detect_groups_qualifier(edition_df)
        if groups:
            edition_groups[key] = groups

    if not edition_groups:
        print("Group context: No groups detected, all features set to defaults")
        return df

    total_groups = sum(len(g) for g in edition_groups.values())
    print(f"Group context: Detected {total_groups} groups across {len(edition_groups)} editions")

    # ---- PASS 2: Process each group chronologically ----
    processed = 0

    for edition_key, groups in edition_groups.items():
        edition_indices = sorted(edition_masks[edition_key])
        num_advancing = 2 if _is_final_tournament(edition_key) else 1

        for group_teams in groups:
            group_set = set(group_teams)
            played = []  # completed matches within this group
            match_counts = {t: 0 for t in group_set}

            for mi in edition_indices:
                row = df.loc[mi]
                home, away = row["home_team"], row["away_team"]

                if home not in group_set or away not in group_set:
                    continue

                match_counts[home] += 1
                match_counts[away] += 1
                df.at[mi, "group_round"] = match_counts[home]  # same for both teams

                # Remaining matches AFTER this one (future only)
                remaining = [
                    (df.loc[li]["home_team"], df.loc[li]["away_team"])
                    for li in edition_indices
                    if li > mi and df.loc[li]["home_team"] in group_set and df.loc[li]["away_team"] in group_set
                ]

                standings = _calculate_standings(group_set, pd.DataFrame(played) if played else pd.DataFrame())

                qh, mwh = _check_status(standings, home, away, remaining, num_advancing)
                qa, mwa = _check_status(standings, away, home, remaining, num_advancing)

                df.at[mi, "is_already_qualified_home"] = qh
                df.at[mi, "must_win_to_survive_home"] = mwh
                df.at[mi, "is_already_qualified_away"] = qa
                df.at[mi, "must_win_to_survive_away"] = mwa

                played.append({
                    "home_team": home, "away_team": away,
                    "home_score": row["home_score"], "away_score": row["away_score"],
                })
                processed += 1

    print(f"Group context: Processed {processed} group matches")
    return df


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

    # Elo-similarity advantage features
    df["sim_wr_advantage"] = df["home_sim_win_rate"] - df["away_sim_win_rate"]
    df["sim_dr_advantage"] = df["home_sim_draw_rate"] - df["away_sim_draw_rate"]
    df["sim_gs_advantage"] = df["home_sim_gs"] - df["away_sim_gs"]
    df["sim_gc_advantage"] = df["home_sim_gc"] - df["away_sim_gc"]
    df["sim_wr_quality"] = (df["home_sim_win_rate"] + df["away_sim_win_rate"]) / 2
    df["sim_dr_quality"] = (df["home_sim_draw_rate"] + df["away_sim_draw_rate"]) / 2

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
        "sim_wr_advantage",
        "sim_dr_advantage",
        "sim_gs_advantage",
        "sim_gc_advantage",
        "sim_wr_quality",
        "sim_dr_quality",
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

    # Step 4.5: Calculate group context (fighting spirit features)
    df = calculate_group_context(df)

    # Step 4.6: Calculate Elo-similarity features
    df = calculate_elo_similarity_features(df)

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
