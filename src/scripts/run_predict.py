"""
Predict 2026 World Cup match outcomes — CLI and batch prediction.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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
from inference.predictor import (
    load_model_and_assets,
    get_wc2026_matches,
    prepare_wc_features,
    predict_custom_match,
)
from features.builder import feature_engineering_v2
from features.pipeline import create_wc2026_schedule


def predict_all_wc_matches():
    """Run predictions for all 2026 World Cup matches."""
    print("=" * 60)
    print("2026 FIFA WORLD CUP - MATCH PREDICTIONS")
    print("=" * 60)

    # Load assets
    model, team_encoder, scaler, feature_cols, temperature, device = load_model_and_assets()
    print(f"Model expects {len(feature_cols)} features")

    # Load full dataset and compute positional + engineered features
    print("\nLoading full dataset for feature computation...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    print("Engineering features...")
    df = feature_engineering_v2(df)

    # Encode team IDs for WC matches
    df["home_team_id"] = team_encoder.transform(df["home_team"])
    df["away_team_id"] = team_encoder.transform(df["away_team"])

    # Get WC 2026 matches
    wc_df = get_wc2026_matches(df)

    if len(wc_df) == 0:
        print("No WC 2026 matches found. Creating from tournament schedule...")
        wc_df = create_wc2026_schedule(team_encoder)
        if wc_df is None:
            print("Cannot create schedule. Exiting.")
            return
        # Compute minimal features for synthetic schedule
        wc_df["home_team_id"] = team_encoder.transform(wc_df["home_team"])
        wc_df["away_team_id"] = team_encoder.transform(wc_df["away_team"])
        # Use feature_engineering_v2 on the synthetic schedule for basic features
        wc_df = feature_engineering_v2(wc_df)

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
        cal_logits = logits / temperature
        probs = torch.softmax(cal_logits, dim=-1).cpu().numpy()
        preds = torch.argmax(cal_logits, dim=1).cpu().numpy()

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
    print("\nTop 10 Teams by Current Elo Rating:")
    elo_df = pd.read_csv(ELO_RATINGS_PATH)
    for i, row in elo_df.head(10).iterrows():
        print(f"  {i+1}. {row['team']:20s} - {row['elo_rating']:.1f}")

    return results_df


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
