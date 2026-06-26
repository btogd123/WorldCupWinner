"""
Predict 2026 World Cup match outcomes — core functions.
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
from utils import set_seed
from modeling.architecture import create_model
from features.builder import feature_engineering_v2
from features.pipeline import create_wc2026_schedule


def load_model_and_assets():
    """Load trained model and preprocessing assets."""
    print("Loading model and assets...")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    team_encoder = checkpoint["team_encoder"]
    scaler = checkpoint["scaler"]
    feature_cols = checkpoint["feature_cols"]
    num_teams = checkpoint["num_teams"]
    num_match_features = checkpoint["num_match_features"]
    is_neutral_idx = checkpoint.get("is_neutral_idx", feature_cols.index("is_neutral"))
    temperature = checkpoint.get("temperature", 1.0)

    # Create model and load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_teams - 1, num_match_features=num_match_features,
                                   device=device, is_neutral_idx=is_neutral_idx)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded: {num_teams - 1} teams, {num_match_features} features")
    print(f"Device: {device}")
    if temperature != 1.0:
        print(f"Temperature scaling: T = {temperature:.4f}")

    return model, team_encoder, scaler, feature_cols, temperature, device


def get_wc2026_matches(df):
    """Extract 2026 World Cup matches from the full-featured dataset."""
    wc_mask = (
        (df["date"] >= pd.to_datetime(WC2026_START))
        & (df["date"] <= pd.to_datetime(WC2026_END))
        & (df["tournament"].str.contains("FIFA World Cup", na=False))
    )

    wc_df = df[wc_mask].copy()
    # Only predict unplayed matches (no scores yet)
    wc_df = wc_df[wc_df["home_score"].isna()].copy()
    print(f"Found {len(wc_df)} unplayed World Cup 2026 matches")
    return wc_df


def prepare_wc_features(df, scaler, feature_cols):
    """Prepare feature tensors for WC matches."""
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    X = scaler.transform(X)

    # Keep is_neutral as raw binary (0/1) — scaling breaks neutral-venue gating
    is_neutral_idx = feature_cols.index("is_neutral")
    X[:, is_neutral_idx] = df["is_neutral"].fillna(0).values.astype(np.float32)

    home_ids = df["home_team_id"].values.astype(np.int64) + 1
    away_ids = df["away_team_id"].values.astype(np.int64) + 1

    return (
        torch.FloatTensor(X),
        torch.LongTensor(home_ids),
        torch.LongTensor(away_ids),
    )


def simulate_match(model, home_team, away_team, features, device, team_encoder, scaler, feature_cols, temperature=1.0):
    """
    Simulate a single match between two teams.
    Allows custom team pairings not in the original dataset.
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(features).unsqueeze(0).to(device)
        h_t = torch.LongTensor([home_team]).to(device)
        a_t = torch.LongTensor([away_team]).to(device)

        logits = model(h_t, a_t, X_t)
        probs = torch.softmax(logits / temperature, dim=-1).cpu().numpy()[0]

    return probs  # [away_win_prob, draw_prob, home_win_prob]


def predict_custom_match(home_team, away_team, is_neutral=True, tournament="FIFA World Cup"):
    """Predict a single custom match."""
    model, team_encoder, scaler, feature_cols, temperature, device = load_model_and_assets()

    if home_team not in team_encoder.classes_:
        print(f"Error: {home_team} not in database")
        return
    if away_team not in team_encoder.classes_:
        print(f"Error: {away_team} not in database")
        return

    # Get current Elo ratings
    elo_df = pd.read_csv(ELO_RATINGS_PATH)
    home_elo = elo_df[elo_df["team"] == home_team]["elo_rating"].values[0]
    away_elo = elo_df[elo_df["team"] == away_team]["elo_rating"].values[0]
    elo_diff = home_elo - away_elo

    # Simplified feature computation
    set_seed(42)

    # Build feature vector using Elo
    feature_dict = {
        "elo_diff": elo_diff,
        "elo_quality": (home_elo + away_elo) / 2,
        "elo_ratio": home_elo / away_elo,
        "abs_elo_diff": abs(elo_diff),
        "form_advantage": 0.0,
        "form_quality": 0.0,
        "wr_advantage": 0.0,
        "sim_wr_advantage": 0.0,
        "sim_gs_advantage": 0.0,
        "sim_dr_quality": 0.0,
        "draw_rate_home": 0.25,
        "draw_rate_away": 0.25,
        "both_draw_prone": 0.25,
        "defensive_similarity": 0.5,
        "strength_parity": 0.5,
        "low_scoring_tendency": 0.0,
        # Context
        "is_neutral": 1 if is_neutral else 0,
        "is_wc": 1,
        "is_wcq": 0,
        "is_continental": 0,
        "is_friendly": 0,
    }

    X = np.array([[feature_dict[col] for col in feature_cols]], dtype=np.float32)
    X = scaler.transform(X)

    # Keep is_neutral as raw binary (0/1) — scaling breaks neutral-venue gating
    is_neutral_idx = feature_cols.index("is_neutral")
    X[:, is_neutral_idx] = 1.0 if is_neutral else 0.0

    home_id = team_encoder.transform([home_team])[0] + 1
    away_id = team_encoder.transform([away_team])[0] + 1

    # Predict
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        h_t = torch.LongTensor([home_id]).to(device)
        a_t = torch.LongTensor([away_id]).to(device)

        logits = model(h_t, a_t, X_t)
        cal_logits = logits / temperature
        probs = torch.softmax(cal_logits, dim=-1).cpu().numpy()[0]

    labels = ["Away Win", "Draw", "Home Win"]
    winner = home_team if probs[2] > max(probs[0], probs[1]) else (
        away_team if probs[0] > max(probs[1], probs[2]) else "Draw"
    )

    print("\n" + "=" * 60)
    print(f"MATCH PREDICTION: {home_team} vs {away_team}")
    print("=" * 60)
    print(f"  Venue: {'Neutral' if is_neutral else 'Home for ' + home_team}")
    print(f"  {home_team} Elo: {home_elo:.0f}")
    print(f"  {away_team} Elo: {away_elo:.0f}")
    print(f"  Elo Difference: {elo_diff:+.0f}")
    print()
    print(f"  {away_team} wins: {probs[0]:.1%}")
    print(f"  Draw:           {probs[1]:.1%}")
    print(f"  {home_team} wins: {probs[2]:.1%}")
    print()
    print(f"  Predicted winner: {winner}")
    print("=" * 60)

    return probs, winner
