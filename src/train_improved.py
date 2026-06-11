"""
Improved training pipeline with better features and model architecture.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
import pickle
import json
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    SCALER_PATH,
    TEAM_ENCODER_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    RESULTS_DIR,
    IMPORTANT_TOURNAMENTS,
)
from improved_model import create_improved_model, ImprovedLoss


def load_and_prepare_data():
    """Load processed data and prepare for improved training."""
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(TEAM_ENCODER_PATH, "rb") as f:
        team_encoder = pickle.load(f)

    return df, scaler, team_encoder


def feature_engineering_v2(df):
    """Enhanced feature engineering."""
    print("Engineering enhanced features...")
    df = df.copy()

    # Basic Elo features
    df["elo_diff_norm"] = df["elo_diff"] / 400.0
    df["elo_ratio"] = (df["home_elo"] / df["away_elo"].clip(lower=1000)) - 1.0

    # Goal difference-based features
    df["home_goal_diff_avg"] = df["home_goals_scored_avg"] - df["home_goals_conceded_avg"]
    df["away_goal_diff_avg"] = df["away_goals_scored_avg"] - df["away_goals_conceded_avg"]
    df["goal_diff_advantage"] = df["home_goal_diff_avg"] - df["away_goal_diff_avg"]

    # Combined strength score
    df["home_strength"] = df["home_elo"] / 1500.0 + df["home_win_rate"] * 0.5 + df["home_form"] * 0.3
    df["away_strength"] = df["away_elo"] / 1500.0 + df["away_win_rate"] * 0.5 + df["away_form"] * 0.3
    df["strength_advantage"] = df["home_strength"] - df["away_strength"]

    # Match quality / competitiveness
    df["match_quality"] = (df["home_elo"] + df["away_elo"]) / 3000.0
    df["elo_gap"] = abs(df["elo_diff"]) / 400.0

    # H2H enhanced
    df["h2h_dominance"] = np.where(
        df["h2h_count"] >= 3,
        (df["h2h_home_wins"] - df["h2h_away_wins"]) / df["h2h_count"],
        0,
    )

    # Temporal features (normalize year)
    df["year_norm"] = (df["year"] - 1950) / 80.0

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


def prepare_enhanced_data(df, scaler, team_encoder, fit_scaler=False):
    """Prepare enhanced feature tensors."""
    feature_cols = [
        # Elo-based
        "elo_advantage_home",
        "elo_quality",
        "elo_diff_norm",
        "elo_ratio",
        "elo_gap",
        # Form-based
        "form_advantage",
        "form_quality",
        "wr_advantage",
        # Goal-based
        "gs_advantage",
        "gc_advantage",
        "goal_diff_advantage",
        # Strength
        "strength_advantage",
        "match_quality",
        # H2H
        "h2h_dominance",
        "has_h2h",
        # Context
        "is_neutral",
        "year_norm",
        "is_wc",
        "is_wcq",
        "is_continental",
        "is_friendly",
    ]

    X = df[feature_cols].fillna(0).values.astype(np.float32)

    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    home_ids = df["home_team_id"].values.astype(np.int64) + 1
    away_ids = df["away_team_id"].values.astype(np.int64) + 1
    y = df["result"].values.astype(np.int64)
    home_goals = df["home_score"].values.astype(np.float32)
    away_goals = df["away_score"].values.astype(np.float32)

    if fit_scaler:
        return X, home_ids, away_ids, y, home_goals, away_goals, feature_cols, scaler
    return X, home_ids, away_ids, y, home_goals, away_goals, feature_cols, None


def split_data_improved(df, train_end="2021-12-31", val_end="2023-12-31"):
    """Split data chronologically with better filtering."""
    # Remove ancient data - modern football patterns are different
    df = df[df["date"] >= pd.to_datetime("2000-01-01")]

    # Remove future matches (no actual results)
    df = df[df["date"] <= pd.to_datetime("2026-06-10")]

    # Remove matches with NaN scores
    df = df.dropna(subset=["home_score", "away_score"])

    train = df[df["date"] < pd.to_datetime(train_end)]
    val = df[
        (df["date"] >= pd.to_datetime(train_end))
        & (df["date"] < pd.to_datetime(val_end))
    ]
    test = df[df["date"] >= pd.to_datetime(val_end)]

    print(f"\nSplit: Train={len(train)} ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"       Val={len(val)}   ({val['date'].min().date()} to {val['date'].max().date()})")
    print(f"       Test={len(test)}  ({test['date'].min().date()} to {test['date'].max().date()})")

    # Show World Cup qualifiers in test
    wcq = test[test["tournament"].str.contains("qualification", na=False)]
    print(f"       WCQ in test: {len(wcq)}")

    return train, val, test


def train_epoch_improved(model, loader, optimizer, criterion, device):
    """Train one epoch with the improved model and loss."""
    model.train()
    total_loss = 0
    total_ce = 0
    total_goal = 0
    all_preds, all_labels = [], []

    for X, h_ids, a_ids, y, hg, ag in loader:
        X = X.to(device)
        h_ids = h_ids.to(device)
        a_ids = a_ids.to(device)
        y = y.to(device)
        hg = hg.to(device)
        ag = ag.to(device)

        optimizer.zero_grad()
        logits, goals = model(h_ids, a_ids, X, return_goals=True)
        loss, ce_loss, goal_loss = criterion(logits, goals, y, hg, ag)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_goal += goal_loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    n = len(loader)
    avg_loss = total_loss / n
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, f1


def evaluate_improved(model, loader, criterion, device):
    """Evaluate improved model."""
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X, h_ids, a_ids, y, hg, ag in loader:
            X = X.to(device)
            h_ids = h_ids.to(device)
            a_ids = a_ids.to(device)
            y = y.to(device)
            hg = hg.to(device)
            ag = ag.to(device)

            logits, goals = model(h_ids, a_ids, X, return_goals=True)
            loss, _, _ = criterion(logits, goals, y, hg, ag)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    n = len(loader)
    avg_loss = total_loss / n
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, f1, all_preds, all_labels, all_probs


def train_improved():
    """Main training function for improved model."""
    print("=" * 60)
    print("Training Improved Match Predictor")
    print("=" * 60)

    # Load data
    df, _, team_encoder = load_and_prepare_data()

    # Enhanced feature engineering
    df = feature_engineering_v2(df)

    # Split data
    train_df, val_df, test_df = split_data_improved(df)

    # Prepare features with dedicated scaler
    X_train, h_train, a_train, y_train, hg_train, ag_train, feature_cols, enhanced_scaler = \
        prepare_enhanced_data(train_df, None, team_encoder, fit_scaler=True)

    X_val, h_val, a_val, y_val, hg_val, ag_val, _, _ = prepare_enhanced_data(
        val_df, enhanced_scaler, team_encoder
    )
    X_test, h_test, a_test, y_test, hg_test, ag_test, _, _ = prepare_enhanced_data(
        test_df, enhanced_scaler, team_encoder
    )

    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # Create tensors
    X_train_t = torch.FloatTensor(X_train)
    h_train_t = torch.LongTensor(h_train)
    a_train_t = torch.LongTensor(a_train)
    y_train_t = torch.LongTensor(y_train)
    hg_train_t = torch.FloatTensor(hg_train)
    ag_train_t = torch.FloatTensor(ag_train)

    X_val_t = torch.FloatTensor(X_val)
    h_val_t = torch.LongTensor(h_val)
    a_val_t = torch.LongTensor(a_val)
    y_val_t = torch.LongTensor(y_val)
    hg_val_t = torch.FloatTensor(hg_val)
    ag_val_t = torch.FloatTensor(ag_val)

    X_test_t = torch.FloatTensor(X_test)
    h_test_t = torch.LongTensor(h_test)
    a_test_t = torch.LongTensor(a_test)
    y_test_t = torch.LongTensor(y_test)
    hg_test_t = torch.FloatTensor(hg_test)
    ag_test_t = torch.FloatTensor(ag_test)

    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_t, h_train_t, a_train_t, y_train_t, hg_train_t, ag_train_t)
    val_dataset = TensorDataset(X_val_t, h_val_t, a_val_t, y_val_t, hg_val_t, ag_val_t)
    test_dataset = TensorDataset(X_test_t, h_test_t, a_test_t, y_test_t, hg_test_t, ag_test_t)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Model
    num_teams = len(team_encoder.classes_)
    model = create_improved_model(num_teams, num_match_features=len(feature_cols), device=device)

    # Class weights
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights_t = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights}")

    # Loss and optimizer
    criterion = ImprovedLoss(class_weights=class_weights_t, goal_weight=0.15)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )

    # Training loop
    best_val_f1 = 0
    patience = 25
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

    print(f"\nTraining... (max 300 epochs)")
    for epoch in range(1, 301):
        train_loss, train_acc, train_f1 = train_epoch_improved(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_f1, _, _, _ = evaluate_improved(
            model, val_loader, criterion, device
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.3f} | F1: {val_f1:.3f}"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_teams": num_teams + 1,
                    "num_match_features": len(feature_cols),
                    "team_encoder": team_encoder,
                    "scaler": enhanced_scaler,
                    "feature_cols": feature_cols,
                },
                MODEL_PATH,
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nBest val F1: {best_val_f1:.4f}")

    # Test evaluation
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_f1, y_pred, y_true, y_prob = evaluate_improved(
        model, test_loader, criterion, device
    )

    print(f"\nTest: Loss={test_loss:.4f}, Acc={test_acc:.4f}, F1={test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Away Win", "Draw", "Home Win"]))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(pd.DataFrame(cm, index=["Away Win", "Draw", "Home Win"], columns=["Away Win", "Draw", "Home Win"]))

    # World Cup qualifier specific evaluation
    wcq_test = test_df[test_df["tournament"].str.contains("qualification", na=False)]
    if len(wcq_test) > 0:
        X_wcq, h_wcq, a_wcq, y_wcq, _, _, _, _ = prepare_enhanced_data(
            wcq_test, enhanced_scaler, team_encoder
        )
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_wcq).to(device)
            h_t = torch.LongTensor(h_wcq + 1).to(device)
            a_t = torch.LongTensor(a_wcq + 1).to(device)
            y_t = torch.LongTensor(y_wcq).to(device)
            logits = model(h_t, a_t, X_t)
            wcq_preds = torch.argmax(logits, dim=1).cpu().numpy()
            wcq_acc = accuracy_score(y_wcq, wcq_preds)
            wcq_f1 = f1_score(y_wcq, wcq_preds, average="macro")

        print(f"\nWorld Cup Qualifiers ({len(wcq_test)} matches):")
        print(f"  Accuracy: {wcq_acc:.4f}")
        print(f"  Macro F1: {wcq_f1:.4f}")
        print(classification_report(y_wcq, wcq_preds, target_names=["Away Win", "Draw", "Home Win"]))

    # Save results
    results = {
        "test_accuracy": float(test_acc),
        "test_f1": float(test_f1),
        "best_val_f1": float(best_val_f1),
        "wcq_accuracy": float(wcq_acc) if len(wcq_test) > 0 else None,
        "wcq_f1": float(wcq_f1) if len(wcq_test) > 0 else None,
    }
    with open(os.path.join(RESULTS_DIR, "improved_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return model, history, results


if __name__ == "__main__":
    train_improved()
