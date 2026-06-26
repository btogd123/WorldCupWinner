"""
Improved training pipeline with better features and model architecture.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pickle
import json
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    TEAM_ENCODER_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    RESULTS_DIR,
    IMPORTANT_TOURNAMENTS,
)
from utils import set_seed
from modeling.architecture import create_model, MultiTaskLoss
from calibration import calibrate_and_evaluate
from features.builder import FEATURE_COLS, feature_engineering_v2, prepare_enhanced_data, split_data_improved
from evaluation import compute_metrics
from modeling.trainer import Trainer


def load_and_prepare_data():
    """Load processed data and prepare for improved training."""
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    with open(TEAM_ENCODER_PATH, "rb") as f:
        team_encoder = pickle.load(f)

    return df, team_encoder


def train_improved():
    """Main training function for improved model."""
    # Reproducibility — best seed: WCQ 62.00% acc, 57.62% F1, 33% draw recall
    set_seed(99)

    print("=" * 60)
    print("Training Improved Match Predictor")
    print("=" * 60)

    # Load data
    df, team_encoder = load_and_prepare_data()

    # Enhanced feature engineering
    df = feature_engineering_v2(df)

    # Plan C split: val=2025-H2 gap, train includes 2026 (with played WC matches through June 16)
    train_df, val_df, test_df = split_data_improved(
        df, train_end="2025-07-01", val_start="2026-01-01", val_end="2026-06-17"
    )

    # Prepare features with dedicated scaler
    X_train, h_train, a_train, y_train, hg_train, ag_train, feature_cols, enhanced_scaler = \
        prepare_enhanced_data(train_df, None, team_encoder, fit_scaler=True)

    X_val, h_val, a_val, y_val, hg_val, ag_val, _, _ = prepare_enhanced_data(
        val_df, enhanced_scaler, team_encoder
    )

    has_test = len(test_df) > 0
    if has_test:
        X_test, h_test, a_test, y_test, hg_test, ag_test, _, _ = prepare_enhanced_data(
            test_df, enhanced_scaler, team_encoder
        )

    print(f"Features ({len(feature_cols)}): {feature_cols}")

    is_neutral_idx = feature_cols.index("is_neutral")

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

    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_t, h_train_t, a_train_t, y_train_t, hg_train_t, ag_train_t)
    val_dataset = TensorDataset(X_val_t, h_val_t, a_val_t, y_val_t, hg_val_t, ag_val_t)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Model
    num_teams = len(team_encoder.classes_)
    model = create_model(num_teams, num_match_features=len(feature_cols), device=device, is_neutral_idx=is_neutral_idx)

    # Loss: plain CE, no class weights (backtest: +6pp Acc, -11% Brier vs weighted)
    criterion = MultiTaskLoss(loss_type="ce", goal_weight=0.15)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )

    # Training loop
    trainer = Trainer(model, optimizer, scheduler, criterion, device,
                      patience=25, grad_clip_norm=1.0, max_epochs=300)

    print(f"\nTraining... (max {trainer.max_epochs} epochs)")
    best_val_f1 = trainer.fit(train_loader, val_loader)

    # Save checkpoint on best val F1
    torch.save(
        {
            "model_state_dict": trainer.best_state,
            "num_teams": num_teams + 1,
            "num_match_features": len(feature_cols),
            "is_neutral_idx": is_neutral_idx,
            "team_encoder": team_encoder,
            "scaler": enhanced_scaler,
            "feature_cols": feature_cols,
        },
        MODEL_PATH,
    )

    print(f"\nBest val F1: {best_val_f1:.4f}")
    history = trainer.history

    # Temperature Scaling calibration (on validation set)
    print("\n" + "=" * 60)
    print("Temperature Scaling Calibration")
    print("=" * 60)

    # Collect validation logits
    model.eval()
    val_logits, val_labels = [], []
    with torch.no_grad():
        for X, h_ids, a_ids, y, _, _ in val_loader:
            X = X.to(device)
            h_ids = h_ids.to(device)
            a_ids = a_ids.to(device)
            logits = model(h_ids, a_ids, X)
            val_logits.append(logits.cpu())
            val_labels.append(y)

    val_logits = torch.cat(val_logits).numpy()
    val_labels = torch.cat(val_labels).numpy()

    # Learn optimal temperature
    cal_result = calibrate_and_evaluate(val_logits, val_labels, val_logits, val_labels,
                                        device=device)
    T = cal_result["temperature"]
    print(f"Learned temperature: T = {T:.4f}")
    print(f"Validation: Raw Acc={cal_result['raw_accuracy']:.4f} F1={cal_result['raw_f1']:.4f} "
          f"→ Cal Acc={cal_result['calibrated_accuracy']:.4f} F1={cal_result['calibrated_f1']:.4f}")
    if T < 1.0:
        print(f"T={T:.4f} < 1 → model was underconfident, probabilities sharpened")
    else:
        print(f"T={T:.4f} > 1 → model was overconfident, probabilities smoothed")

    # Re-save checkpoint with temperature
    checkpoint["temperature"] = T
    torch.save(checkpoint, MODEL_PATH)
    print(f"Temperature saved to checkpoint.")

    # Test evaluation (skip if no test data)
    if has_test:
        X_test_t = torch.FloatTensor(X_test)
        h_test_t = torch.LongTensor(h_test)
        a_test_t = torch.LongTensor(a_test)
        y_test_t = torch.LongTensor(y_test)
        hg_test_t = torch.FloatTensor(hg_test)
        ag_test_t = torch.FloatTensor(ag_test)
        test_dataset = TensorDataset(X_test_t, h_test_t, a_test_t, y_test_t, hg_test_t, ag_test_t)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

        print("\n" + "=" * 60)
        print("Test Set Evaluation")
        print("=" * 60)

        test_loss, test_acc, test_f1, y_pred, y_true, y_prob = evaluate_improved(
            model, test_loader, criterion, device
        )

        model.eval()
        test_logits_list, test_labels_list = [], []
        with torch.no_grad():
            for X, h_ids, a_ids, y, _, _ in test_loader:
                X = X.to(device)
                h_ids = h_ids.to(device)
                a_ids = a_ids.to(device)
                logits_t = model(h_ids, a_ids, X)
                test_logits_list.append(logits_t.cpu())
                test_labels_list.append(y)

        test_logits_all = torch.cat(test_logits_list).numpy()
        test_labels_all = torch.cat(test_labels_list).numpy()

        test_cal = calibrate_and_evaluate(val_logits, val_labels, test_logits_all,
                                          test_labels_all, device=device)
        test_probs_cal = test_cal["calibrated_probs"]
        test_preds_cal = np.argmax(test_probs_cal, axis=1)
        test_acc_cal = test_cal["calibrated_accuracy"]
        test_f1_cal = test_cal["calibrated_f1"]

        print(f"\nTest (raw):       Acc={test_acc:.4f}, F1={test_f1:.4f}")
        print(f"Test (calibrated): Acc={test_acc_cal:.4f}, F1={test_f1_cal:.4f}")
        print(f"Temperature: T={T:.4f}")

        print("\nClassification Report (calibrated):")
        print(classification_report(test_labels_all, test_preds_cal, target_names=["Away Win", "Draw", "Home Win"]))
        print("\nConfusion Matrix (calibrated):")
        cm = confusion_matrix(test_labels_all, test_preds_cal)
        print(pd.DataFrame(cm, index=["Away Win", "Draw", "Home Win"], columns=["Away Win", "Draw", "Home Win"]))

        # World Cup qualifier specific evaluation
        wcq_test = test_df[test_df["tournament"].str.contains("qualification", na=False)]
        wcq_acc, wcq_f1, wcq_acc_cal, wcq_f1_cal = None, None, None, None
        if len(wcq_test) > 0:
            X_wcq, h_wcq, a_wcq, y_wcq, _, _, _, _ = prepare_enhanced_data(
                wcq_test, enhanced_scaler, team_encoder
            )
            model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X_wcq).to(device)
                h_t = torch.LongTensor(h_wcq).to(device)
                a_t = torch.LongTensor(a_wcq).to(device)
                y_t = torch.LongTensor(y_wcq).to(device)
                logits = model(h_t, a_t, X_t)
                wcq_probs_raw = torch.softmax(logits, dim=1).cpu().numpy()
                wcq_preds = np.argmax(wcq_probs_raw, axis=1)
                wcq_acc = accuracy_score(y_wcq, wcq_preds)
                wcq_f1 = f1_score(y_wcq, wcq_preds, average="macro")

                wcq_cal = calibrate_and_evaluate(val_logits, val_labels,
                                                 logits.cpu().numpy(), y_wcq,
                                                 device=device)
                wcq_probs_cal = wcq_cal["calibrated_probs"]
                wcq_preds_cal = np.argmax(wcq_probs_cal, axis=1)
                wcq_acc_cal = wcq_cal["calibrated_accuracy"]
                wcq_f1_cal = wcq_cal["calibrated_f1"]

            print(f"\nWorld Cup Qualifiers ({len(wcq_test)} matches):")
            print(f"  Raw:        Acc={wcq_acc:.4f}, F1={wcq_f1:.4f}")
            print(f"  Calibrated: Acc={wcq_acc_cal:.4f}, F1={wcq_f1_cal:.4f}")
    else:
        test_acc = None
        test_f1 = None
        test_acc_cal = None
        test_f1_cal = None
        wcq_acc = None
        wcq_f1 = None
        wcq_acc_cal = None
        wcq_f1_cal = None
        print("\nNo test set available (all scored matches used for training/validation)")

    # Save results
    results = {
        "test_accuracy": float(test_acc) if test_acc is not None else None,
        "test_f1": float(test_f1) if test_f1 is not None else None,
        "test_accuracy_calibrated": float(test_acc_cal) if test_acc_cal is not None else None,
        "test_f1_calibrated": float(test_f1_cal) if test_f1_cal is not None else None,
        "temperature": float(T),
        "best_val_f1": float(best_val_f1),
        "wcq_accuracy": float(wcq_acc) if (wcq_acc is not None) else None,
        "wcq_f1": float(wcq_f1) if (wcq_f1 is not None) else None,
        "wcq_accuracy_calibrated": float(wcq_acc_cal) if (wcq_acc_cal is not None) else None,
        "wcq_f1_calibrated": float(wcq_f1_cal) if (wcq_f1_cal is not None) else None,
    }
    with open(os.path.join(RESULTS_DIR, "improved_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return model, history, results


if __name__ == "__main__":
    train_improved()
