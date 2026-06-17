"""
Backtest: Compare 3 loss variants on WC 2022 (64 matches).

Variants:
  A: CrossEntropyLoss + class weights (baseline)
  B: CrossEntropyLoss, no class weights
  C: Focal Loss gamma=2, no class weights

All trained on identical data (2000-2021 train, 2022 pre-WC val)
and evaluated on WC 2022.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pickle
import json
import os
import sys
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DATA_PATH, SCALER_PATH, TEAM_ENCODER_PATH, RESULTS_DIR
from improved_model import create_improved_model, ImprovedLoss
from train_improved import feature_engineering_v2, prepare_enhanced_data


def brier_score(y_true, probs):
    """Multiclass Brier score: mean(sum((p_k - y_k)^2))."""
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(len(y_true)), y_true] = 1
    return np.mean(np.sum((probs - y_onehot) ** 2, axis=1))


def log_loss_score(y_true, probs, eps=1e-15):
    """Log loss: -mean(log(p_true_class))."""
    probs = np.clip(probs, eps, 1 - eps)
    return -np.mean(np.log(probs[np.arange(len(y_true)), y_true]))


def train_one_variant(name, X_train, h_train, a_train, y_train, hg_train, ag_train,
                      X_val, h_val, a_val, y_val, hg_val, ag_val,
                      X_test, h_test, a_test, y_test, hg_test, ag_test,
                      num_teams, num_features, is_neutral_idx, device, seed=99):
    """Train a single variant and return metrics."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create tensors
    t = lambda x, dtype: torch.tensor(x, dtype=dtype)

    train_dataset = TensorDataset(
        t(X_train, torch.float32), t(h_train, torch.long), t(a_train, torch.long),
        t(y_train, torch.long), t(hg_train, torch.float32), t(ag_train, torch.float32))
    val_dataset = TensorDataset(
        t(X_val, torch.float32), t(h_val, torch.long), t(a_val, torch.long),
        t(y_val, torch.long), t(hg_val, torch.float32), t(ag_val, torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Model
    model = create_improved_model(
        num_teams, num_match_features=num_features, device=device, is_neutral_idx=is_neutral_idx)

    # Loss config per variant
    if name == "A_CE_weights":
        class_weights_np = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weights = torch.FloatTensor(class_weights_np).to(device)
        criterion = ImprovedLoss(class_weights=class_weights, goal_weight=0.15, loss_type="ce")
    elif name == "B_CE_noweights":
        criterion = ImprovedLoss(loss_type="ce", goal_weight=0.15)
    elif name == "C_Focal":
        criterion = ImprovedLoss(loss_type="focal", gamma=2.0, goal_weight=0.15)
    else:
        raise ValueError(f"Unknown variant: {name}")

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)

    best_val_f1 = 0
    patience = 25
    patience_counter = 0
    best_state = None

    for epoch in range(1, 301):
        # Train
        model.train()
        for X, h_ids, a_ids, y, hg, ag in train_loader:
            X, h_ids, a_ids, y = X.to(device), h_ids.to(device), a_ids.to(device), y.to(device)
            hg, ag = hg.to(device), ag.to(device)

            optimizer.zero_grad()
            logits, goals = model(h_ids, a_ids, X, return_goals=True)
            loss, _, _ = criterion(logits, goals, y, hg, ag)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X, h_ids, a_ids, y, hg, ag in val_loader:
                X, h_ids, a_ids, y = X.to(device), h_ids.to(device), a_ids.to(device), y.to(device)
                logits, goals = model(h_ids, a_ids, X, return_goals=True)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # Evaluate on test set
    X_test_t = torch.FloatTensor(X_test).to(device)
    h_test_t = torch.LongTensor(h_test).to(device)
    a_test_t = torch.LongTensor(a_test).to(device)

    with torch.no_grad():
        logits = model(h_test_t, a_test_t, X_test_t)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    brier = brier_score(y_test, probs)
    logloss = log_loss_score(y_test, probs)

    # Mean probabilities per class
    mean_probs = probs.mean(axis=0)

    return {
        "name": name,
        "best_val_f1": float(best_val_f1),
        "test_acc": float(acc),
        "test_f1": float(f1),
        "test_brier": float(brier),
        "test_logloss": float(logloss),
        "mean_away_prob": float(mean_probs[0]),
        "mean_draw_prob": float(mean_probs[1]),
        "mean_home_prob": float(mean_probs[2]),
        "epochs_trained": epoch - patience_counter,
    }


def main():
    # Reproducibility base
    SEED = 99

    print("=" * 70)
    print("Loss Function Backtest — WC 2022 (64 matches)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    with open(TEAM_ENCODER_PATH, "rb") as f:
        team_encoder = pickle.load(f)

    # Feature engineering
    df = feature_engineering_v2(df)

    # Split: train 2000-2021, val 2022 pre-WC, test WC 2022
    train_df = df[(df["date"] >= "2000-01-01") & (df["date"] < "2022-01-01")]
    val_df = df[(df["date"] >= "2022-01-01") & (df["date"] < "2022-11-20")]
    test_df = df[df["tournament"].str.contains("FIFA World Cup", na=False) & (df["date"].dt.year == 2022)]

    # Only 2022 WC matches (not qualifiers)
    test_df = test_df[test_df["date"] >= "2022-11-20"]

    print(f"Train: {len(train_df)} ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"Val:   {len(val_df)} ({val_df['date'].min().date()} to {val_df['date'].max().date()})")
    print(f"Test:  {len(test_df)} (WC 2022: {test_df['date'].min().date()} to {test_df['date'].max().date()})")

    # Prepare features
    X_train, h_train, a_train, y_train, hg_train, ag_train, feature_cols, scaler = \
        prepare_enhanced_data(train_df, None, team_encoder, fit_scaler=True)
    X_val, h_val, a_val, y_val, hg_val, ag_val, _, _ = \
        prepare_enhanced_data(val_df, scaler, team_encoder)
    X_test, h_test, a_test, y_test, hg_test, ag_test, _, _ = \
        prepare_enhanced_data(test_df, scaler, team_encoder)

    is_neutral_idx = feature_cols.index("is_neutral")
    num_teams = len(team_encoder.classes_)
    num_features = len(feature_cols)

    print(f"\nFeatures ({num_features}): {feature_cols}")
    print(f"Teams: {num_teams}")

    # Class distribution
    for name, arr in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        _, counts = np.unique(arr, return_counts=True)
        print(f"{name} dist: Away={counts[0]}, Draw={counts[1]}, Home={counts[2]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Train all variants
    variants = [
        ("A_CE_weights", "CrossEntropy + class weights (baseline)"),
        ("B_CE_noweights", "CrossEntropy, no weights"),
        ("C_Focal", "Focal Loss gamma=2, no weights"),
    ]

    results_all = []

    for variant_name, variant_desc in variants:
        print(f"\n{'=' * 70}")
        print(f"Training: {variant_name} — {variant_desc}")
        print(f"{'=' * 70}")

        result = train_one_variant(
            variant_name,
            X_train, h_train, a_train, y_train, hg_train, ag_train,
            X_val, h_val, a_val, y_val, hg_val, ag_val,
            X_test, h_test, a_test, y_test, hg_test, ag_test,
            num_teams, num_features, is_neutral_idx, device, seed=SEED)

        results_all.append(result)

        print(f"\n{variant_name} results:")
        print(f"  Best val F1: {result['best_val_f1']:.4f}")
        print(f"  Epochs: {result['epochs_trained']}")
        print(f"  Test Acc:   {result['test_acc']:.4f}")
        print(f"  Test F1:    {result['test_f1']:.4f}")
        print(f"  Test Brier: {result['test_brier']:.4f}")
        print(f"  Test LogLoss: {result['test_logloss']:.4f}")
        print(f"  Mean probs: Away={result['mean_away_prob']:.3f}, Draw={result['mean_draw_prob']:.3f}, Home={result['mean_home_prob']:.3f}")

    # Comparison table
    print(f"\n{'=' * 70}")
    print("Final Comparison")
    print(f"{'=' * 70}")
    header = f"{'Variant':<30} {'Acc':>7} {'F1':>7} {'Brier':>7} {'LogLoss':>8} {'Draw%':>7} {'Home%':>7}"
    print(header)
    print("-" * len(header))
    for r in results_all:
        print(f"{r['name']:<30} {r['test_acc']:7.4f} {r['test_f1']:7.4f} "
              f"{r['test_brier']:7.4f} {r['test_logloss']:8.4f} "
              f"{r['mean_draw_prob']:7.3f} {r['mean_home_prob']:7.3f}")

    # Save results
    out_path = os.path.join(RESULTS_DIR, "backtest_loss.json")
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
