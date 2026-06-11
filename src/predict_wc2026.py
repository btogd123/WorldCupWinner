"""
Predict 2026 World Cup match outcomes.
"""
import torch
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime

from config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    ELO_RATINGS_PATH,
    RESULTS_DIR,
    WC2026_START,
    WC2026_END,
)
from improved_model import create_improved_model


def load_model_and_assets():
    """Load trained model and preprocessing assets."""
    print("Loading model and assets...")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    team_encoder = checkpoint["team_encoder"]
    scaler = checkpoint["scaler"]
    feature_cols = checkpoint["feature_cols"]
    num_teams = checkpoint["num_teams"]
    num_match_features = checkpoint["num_match_features"]

    # Create model and load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_improved_model(num_teams - 1, num_match_features=num_match_features, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded: {num_teams - 1} teams, {num_match_features} features")
    print(f"Device: {device}")

    return model, team_encoder, scaler, feature_cols, device


def get_wc2026_matches():
    """Extract 2026 World Cup matches from the dataset."""
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # Filter for 2026 World Cup matches
    wc_mask = (
        (df["date"] >= pd.to_datetime(WC2026_START))
        & (df["date"] <= pd.to_datetime(WC2026_END))
        & (df["tournament"].str.contains("FIFA World Cup", na=False))
    )

    wc_df = df[wc_mask].copy()
    print(f"Found {len(wc_df)} World Cup 2026 matches")
    return wc_df


def build_match_features(wc_df, team_encoder):
    """Build features for WC 2026 matches using the same logic as training."""
    df = wc_df.copy()

    # Use the pre-computed Elo and form features from processed data
    # These were computed using historical data up to each match date

    # Elo features
    df["elo_diff_norm"] = df["elo_diff"] / 400.0
    df["elo_ratio"] = (df["home_elo"] / df["away_elo"].clip(lower=1000)) - 1.0

    # Goal difference
    df["home_goal_diff_avg"] = df["home_goals_scored_avg"] - df["home_goals_conceded_avg"]
    df["away_goal_diff_avg"] = df["away_goals_scored_avg"] - df["away_goals_conceded_avg"]
    df["goal_diff_advantage"] = df["home_goal_diff_avg"] - df["away_goal_diff_avg"]

    # Strength
    df["home_strength"] = df["home_elo"] / 1500.0 + df["home_win_rate"] * 0.5 + df["home_form"] * 0.3
    df["away_strength"] = df["away_elo"] / 1500.0 + df["away_win_rate"] * 0.5 + df["away_form"] * 0.3
    df["strength_advantage"] = df["home_strength"] - df["away_strength"]

    # Match quality
    df["match_quality"] = (df["home_elo"] + df["away_elo"]) / 3000.0
    df["elo_gap"] = abs(df["elo_diff"]) / 400.0

    # H2H
    df["h2h_dominance"] = np.where(
        df["h2h_count"] >= 3,
        (df["h2h_home_wins"] - df["h2h_away_wins"]) / df["h2h_count"],
        0,
    )

    # Year
    df["year_norm"] = (df["year"] - 1950) / 80.0

    # Tournament indicators
    df["is_wc"] = 1
    df["is_wcq"] = 0
    df["is_friendly"] = 0
    df["is_continental"] = 0

    # Neutral venue - World Cup matches are all neutral
    # But some might be hosted by a participating team (like USA)
    df["is_neutral"] = df["neutral"].astype(int)

    # Team IDs
    df["home_team_id"] = team_encoder.transform(df["home_team"])
    df["away_team_id"] = team_encoder.transform(df["away_team"])

    return df


def prepare_wc_features(df, scaler, feature_cols):
    """Prepare feature tensors for WC matches."""
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    X = scaler.transform(X)

    home_ids = df["home_team_id"].values.astype(np.int64) + 1
    away_ids = df["away_team_id"].values.astype(np.int64) + 1

    return (
        torch.FloatTensor(X),
        torch.LongTensor(home_ids),
        torch.LongTensor(away_ids),
    )


def simulate_match(model, home_team, away_team, features, device, team_encoder, scaler, feature_cols):
    """
    Simulate a single match between two teams.
    Allows custom team pairings not in the original dataset.
    """
    # Create a minimal feature vector
    # For custom matchups, we use available features from the closest context
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(features).unsqueeze(0).to(device)
        h_t = torch.LongTensor([home_team]).to(device)
        a_t = torch.LongTensor([away_team]).to(device)

        logits = model(h_t, a_t, X_t)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    return probs  # [away_win_prob, draw_prob, home_win_prob]


def predict_all_wc_matches():
    """Run predictions for all 2026 World Cup matches."""
    print("=" * 60)
    print("2026 FIFA WORLD CUP - MATCH PREDICTIONS")
    print("=" * 60)

    # Load assets
    model, team_encoder, scaler, feature_cols, device = load_model_and_assets()

    # Get WC 2026 matches
    wc_df = get_wc2026_matches()

    if len(wc_df) == 0:
        print("No WC 2026 matches found. Creating from tournament schedule...")
        wc_df = create_wc2026_schedule(team_encoder)
        if wc_df is None:
            print("Cannot create schedule. Exiting.")
            return

    # Build features
    wc_df = build_match_features(wc_df, team_encoder)

    # Prepare tensors
    X_t, h_t, a_t = prepare_wc_features(wc_df, scaler, feature_cols)

    # Run predictions
    model.eval()
    predictions = []
    with torch.no_grad():
        X_t = X_t.to(device)
        h_t = h_t.to(device)
        a_t = a_t.to(device)

        logits = model(h_t, a_t, X_t)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    # Compile results
    results = []
    for i, (_, row) in enumerate(wc_df.iterrows()):
        result = {
            "date": row["date"].strftime("%Y-%m-%d"),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_elo": round(row.get("home_elo", 0), 1),
            "away_elo": round(row.get("away_elo", 0), 1),
            "elo_diff": round(row.get("elo_diff", 0), 1),
            "pred_away_win": round(float(probs[i][0]), 4),
            "pred_draw": round(float(probs[i][1]), 4),
            "pred_home_win": round(float(probs[i][2]), 4),
            "prediction": ["Away Win", "Draw", "Home Win"][preds[i]],
            "confidence": round(float(max(probs[i])), 4),
        }
        results.append(result)

    # Display results
    results_df = pd.DataFrame(results)

    # Group stage matches
    print("\n" + "-" * 60)
    print("PREDICTIONS (sorted by date):")
    print("-" * 60)

    for i, r in enumerate(results):
        winner_icon = "[H]" if r["prediction"] == "Home Win" else ("[D]" if r["prediction"] == "Draw" else "[A]")
        print(
            f"{r['date']} | {r['home_team']:20s} vs {r['away_team']:20s} | "
            f"Pred: {r['prediction']:8s} {winner_icon} | "
            f"Conf: {r['confidence']:.2%} | "
            f"({r['pred_away_win']:.1%}/{r['pred_draw']:.1%}/{r['pred_home_win']:.1%})"
        )

    # Save results
    results_path = os.path.join(RESULTS_DIR, "wc2026_predictions.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    results_csv = os.path.join(RESULTS_DIR, "wc2026_predictions.csv")
    results_df.to_csv(results_csv, index=False)

    print(f"\nPredictions saved to:")
    print(f"  JSON: {results_path}")
    print(f"  CSV:  {results_csv}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    pred_counts = results_df["prediction"].value_counts()
    for pred, count in pred_counts.items():
        print(f"  {pred}: {count} ({count/len(results)*100:.1f}%)")

    # Top teams by Elo
    print("\nTop 10 Teams by Current Elo:")
    elo_df = pd.read_csv(ELO_RATINGS_PATH)
    for i, row in elo_df.head(10).iterrows():
        print(f"  {i+1}. {row['team']:20s} - {row['elo_rating']:.1f}")

    return results_df


def create_wc2026_schedule(team_encoder):
    """Create WC 2026 schedule manually if not in dataset."""
    # WC 2026 has 48 teams in groups of 4 (12 groups)
    # Top 2 from each group + 8 best 3rd place advance
    # This is a simplified version with key matches

    # The 48 qualified teams (based on typical qualifications as of 2026)
    # This list should be updated with actual qualified teams
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

    # Verify all teams are in the encoder
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

    # Create a simple tournament structure
    # For now, we'll create group stage matches
    import random
    random.seed(42)

    # Shuffle and create groups
    teams = valid_teams[:48] if len(valid_teams) >= 48 else valid_teams
    random.shuffle(teams)

    # Create groups of 4
    groups = []
    for i in range(0, min(48, len(teams)), 4):
        if i + 3 < len(teams):
            groups.append(teams[i : i + 4])

    # Generate round-robin matches for each group
    matches = []
    match_date = pd.to_datetime("2026-06-11")

    for group_idx, group in enumerate(groups):
        # Round-robin within group (6 matches per group)
        for i in range(4):
            for j in range(i + 1, 4):
                # Alternate home/away
                if (i + j) % 2 == 0:
                    home_team, away_team = group[i], group[j]
                else:
                    home_team, away_team = group[j], group[i]

                matches.append(
                    {
                        "date": match_date,
                        "home_team": home_team,
                        "away_team": away_team,
                        "tournament": "FIFA World Cup",
                        "neutral": True,
                    }
                )

        match_date += pd.Timedelta(days=1)

    schedule_df = pd.DataFrame(matches)
    print(f"Created schedule with {len(schedule_df)} group matches across {len(groups)} groups")
    return schedule_df


def predict_custom_match(home_team, away_team, is_neutral=True, tournament="FIFA World Cup"):
    """Predict a single custom match."""
    model, team_encoder, scaler, feature_cols, device = load_model_and_assets()

    if home_team not in team_encoder.classes_:
        print(f"Error: {home_team} not in database")
        return
    if away_team not in team_encoder.classes_:
        print(f"Error: {away_team} not in database")
        return

    # Load processed data for feature computation
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # Get current Elo ratings
    elo_df = pd.read_csv(ELO_RATINGS_PATH)
    home_elo = elo_df[elo_df["team"] == home_team]["elo_rating"].values[0]
    away_elo = elo_df[elo_df["team"] == away_team]["elo_rating"].values[0]

    # Get latest form data for both teams
    home_matches = df[(df["home_team"] == home_team) | (df["away_team"] == home_team)]
    away_matches = df[(df["home_team"] == away_team) | (df["away_team"] == away_team)]

    # Simplified feature computation
    import random
    random.seed(42)
    np.random.seed(42)

    # Build feature vector using average values
    feature_dict = {
        "elo_advantage_home": (home_elo - away_elo) / 400.0,
        "elo_quality": (home_elo + away_elo) / 3000.0,
        "elo_diff_norm": (home_elo - away_elo) / 400.0,
        "elo_ratio": (home_elo / max(away_elo, 1000)) - 1.0,
        "elo_gap": abs(home_elo - away_elo) / 400.0,
        "form_advantage": 0.0,
        "form_quality": 0.0,
        "wr_advantage": 0.0,
        "gs_advantage": 0.0,
        "gc_advantage": 0.0,
        "goal_diff_advantage": 0.0,
        "strength_advantage": (home_elo - away_elo) / 1500.0,
        "match_quality": (home_elo + away_elo) / 3000.0,
        "h2h_dominance": 0.0,
        "has_h2h": 0,
        "is_neutral": 1 if is_neutral else 0,
        "year_norm": (2026 - 1950) / 80.0,
        "is_wc": 1,
        "is_wcq": 0,
        "is_continental": 0,
        "is_friendly": 0,
    }

    X = np.array([[feature_dict[col] for col in feature_cols]], dtype=np.float32)
    X = scaler.transform(X)

    home_id = team_encoder.transform([home_team])[0] + 1
    away_id = team_encoder.transform([away_team])[0] + 1

    # Predict
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        h_t = torch.LongTensor([home_id]).to(device)
        a_t = torch.LongTensor([away_id]).to(device)

        logits = model(h_t, a_t, X_t)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    labels = ["Away Win", "Draw", "Home Win"]
    winner = home_team if probs[2] > max(probs[0], probs[1]) else (
        away_team if probs[0] > max(probs[1], probs[2]) else "Draw"
    )

    print("\n" + "=" * 60)
    print(f"MATCH PREDICTION: {home_team} vs {away_team}")
    print("=" * 60)
    print(f"  Venue: {'Neutral' if is_neutral else 'Home for ' + home_team}")
    print(f"  {home_team} Elo: {home_elo:.1f}")
    print(f"  {away_team} Elo: {away_elo:.1f}")
    print(f"  Elo Difference: {home_elo - away_elo:+.1f}")
    print()
    print(f"  {away_team} wins: {probs[0]:.1%}")
    print(f"  Draw:           {probs[1]:.1%}")
    print(f"  {home_team} wins: {probs[2]:.1%}")
    print()
    print(f"  Predicted winner: {winner}")
    print("=" * 60)

    return probs, winner


def main():
    """Entry point for wc-predict command."""
    results = predict_all_wc_matches()

    if results is not None:
        print("\n\n" + "=" * 60)
        print("CUSTOM MATCH PREDICTIONS")
        print("=" * 60)
        predict_custom_match("France", "Brazil")
        predict_custom_match("Argentina", "Spain")
        predict_custom_match("England", "Germany")


if __name__ == "__main__":
    main()
