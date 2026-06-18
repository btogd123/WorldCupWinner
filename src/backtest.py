"""
Reusable backtest runner: train → validate → test with optional temperature scaling.

Ports the training loop from train_one_variant (backtest_positional_strength.py)
into a configurable, model-agnostic function.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

from model import create_model, MultiTaskLoss
from evaluation import compute_metrics
from calibration import TemperatureScaler
from sklearn.metrics import f1_score


@dataclass
class BacktestConfig:
    lr: float = 0.001
    weight_decay: float = 1e-4
    train_batch_size: int = 64
    val_batch_size: int = 128
    max_epochs: int = 300
    patience: int = 25
    seed: int = 99
    grad_clip_norm: float = 1.0
    loss_type: str = "ce"          # "ce" | "brier"
    goal_weight: float = 0.15
    T_0: int = 30
    T_mult: int = 2
    eta_min: float = 1e-6
    apply_temp_scaling: bool = False
    verbose: bool = False


def run_backtest(
    model_factory,
    train_data,
    val_data,
    test_data,
    num_teams: int,
    num_features: int,
    is_neutral_idx: int,
    device: torch.device,
    config: BacktestConfig = None,
    test_df: pd.DataFrame = None,
):
    """Run a complete train → validate → test backtest.

    Args:
        model_factory: Callable(num_teams, num_match_features, device, is_neutral_idx) -> nn.Module
        train_data: (X, home_ids, away_ids, y, home_goals, away_goals)
        val_data:   (X, home_ids, away_ids, y, home_goals, away_goals)
        test_data:  (X, home_ids, away_ids, y, home_goals, away_goals)
        num_teams: number of unique teams (pre-padding)
        num_features: number of match features
        is_neutral_idx: column index of is_neutral in feature vector
        device: torch device
        config: BacktestConfig (uses defaults if None)
        test_df: optional DataFrame with per-row metadata for predictions output

    Returns:
        dict with keys: metrics, predictions_df, best_val_f1, epochs_trained,
                        temperature, best_state_dict
    """
    if config is None:
        config = BacktestConfig()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    X_train, h_train, a_train, y_train, hg_train, ag_train = train_data
    X_val, h_val, a_val, y_val, hg_val, ag_val = val_data
    X_test, h_test, a_test, y_test, hg_test, ag_test = test_data

    t = lambda x, dtype: torch.tensor(x, dtype=dtype)

    train_dataset = TensorDataset(
        t(X_train, torch.float32), t(h_train, torch.long), t(a_train, torch.long),
        t(y_train, torch.long), t(hg_train, torch.float32), t(ag_train, torch.float32))
    val_dataset = TensorDataset(
        t(X_val, torch.float32), t(h_val, torch.long), t(a_val, torch.long),
        t(y_val, torch.long), t(hg_val, torch.float32), t(ag_val, torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                               shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False)

    model = model_factory(
        num_teams, num_match_features=num_features, device=device,
        is_neutral_idx=is_neutral_idx)

    criterion = MultiTaskLoss(loss_type=config.loss_type, goal_weight=config.goal_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                   weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.T_0, T_mult=config.T_mult, eta_min=config.eta_min)

    best_val_f1 = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for X, h_ids, a_ids, y, hg, ag in train_loader:
            X, h_ids, a_ids, y = X.to(device), h_ids.to(device), a_ids.to(device), y.to(device)
            hg, ag = hg.to(device), ag.to(device)
            optimizer.zero_grad()
            logits, goals = model(h_ids, a_ids, X, return_goals=True)
            loss, _, _ = criterion(logits, goals, y, hg, ag)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X, h_ids, a_ids, y, hg, ag in val_loader:
                X, h_ids, a_ids, y = X.to(device), h_ids.to(device), a_ids.to(device), y.to(device)
                logits, _ = model(h_ids, a_ids, X, return_goals=True)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average="macro")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= config.patience:
            break

    epochs_trained = epoch - patience_counter

    model.load_state_dict(best_state)
    model.eval()

    X_test_t = torch.FloatTensor(X_test).to(device)
    h_test_t = torch.LongTensor(h_test).to(device)
    a_test_t = torch.LongTensor(a_test).to(device)

    with torch.no_grad():
        logits = model(h_test_t, a_test_t, X_test_t).cpu().numpy()

    # Temperature scaling
    temperature = 1.0
    if config.apply_temp_scaling:
        val_logits_list, val_labels_list = [], []
        with torch.no_grad():
            for X, h_ids, a_ids, y, hg, ag in val_loader:
                X, h_ids, a_ids, y = X.to(device), h_ids.to(device), a_ids.to(device), y.to(device)
                vl = model(h_ids, a_ids, X).cpu().numpy()
                val_logits_list.append(vl)
                val_labels_list.append(y.cpu().numpy())
        val_logits_all = np.concatenate(val_logits_list, axis=0)
        val_labels_all = np.concatenate(val_labels_list, axis=0)
        scaler = TemperatureScaler()
        scaler.fit(val_logits_all, val_labels_all, device=device)
        probs = scaler.calibrate(logits)
        temperature = scaler.get_temperature()
    else:
        probs = F.softmax(torch.FloatTensor(logits), dim=-1).numpy()

    preds = np.argmax(probs, axis=1)

    metrics = compute_metrics(y_test, probs)

    # Neutral-only subset
    is_neut_test = X_test[:, is_neutral_idx] > 0.1
    if is_neut_test.sum() > 0:
        neut_metrics = compute_metrics(y_test[is_neut_test], probs[is_neut_test])
    else:
        neut_metrics = None

    predictions_df = None
    if test_df is not None:
        rows = []
        for i, (_, row) in enumerate(test_df.iterrows()):
            hs = row.get("home_score")
            aws = row.get("away_score")
            has_result = pd.notna(hs) and pd.notna(aws)
            actual = None
            actual_label = None
            correct = None
            if has_result:
                hs, aws = int(hs), int(aws)
                actual = f"{hs}-{aws}"
                if hs > aws:
                    actual_label = 2
                elif hs == aws:
                    actual_label = 1
                else:
                    actual_label = 0
                correct = int(preds[i] == actual_label)
            rows.append({
                "date": row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"]),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_elo": round(float(row.get("home_elo", 0)), 1),
                "away_elo": round(float(row.get("away_elo", 0)), 1),
                "actual_score": actual,
                "pred_away_win": round(float(probs[i][0]), 4),
                "pred_draw": round(float(probs[i][1]), 4),
                "pred_home_win": round(float(probs[i][2]), 4),
                "prediction": ["Away Win", "Draw", "Home Win"][preds[i]],
                "confidence": round(float(max(probs[i])), 4),
                "correct": correct,
            })
        predictions_df = pd.DataFrame(rows)

    return {
        "metrics": metrics,
        "neutral_metrics": neut_metrics,
        "predictions_df": predictions_df,
        "best_val_f1": float(best_val_f1),
        "epochs_trained": epochs_trained,
        "temperature": float(temperature),
        "best_state_dict": best_state,
    }
