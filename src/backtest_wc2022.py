"""
Backtest: Train model on pre-2022-WC data, predict WC 2022 group stage,
compare with actual results to measure true out-of-sample accuracy.
"""
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from config import PROCESSED_DATA_PATH, RESULTS_DIR, BATCH_SIZE
from improved_model import create_improved_model, ImprovedLoss

# WC 2022 dates
WC2022_START = "2022-11-20"


def load_and_filter_data():
    """Load processed data and filter to before WC 2022."""
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # Only keep matches before WC 2022 starts
    df = df[df["date"] < pd.to_datetime(WC2022_START)]

    # Modern football
    df = df[df["date"] >= pd.to_datetime("2000-01-01")]

    # Drop NaN scores
    df = df.dropna(subset=["home_score", "away_score"])

    print(f"Pre-WC2022 data: {len(df)} matches ({df['date'].min().date()} to {df['date'].max().date()})")
    return df


def augment_neutral_matches(df):
    """Flip home/away for neutral-venue matches to teach the model symmetry.

    On neutral ground, the 'home' label is arbitrary. By swapping home/away
    and concatenating, the model learns that directional features should not
    bias predictions toward home wins when the venue is neutral.
    """
    neutral_mask = df["neutral"].astype(bool)
    neutral_count = neutral_mask.sum()
    print(f"\nNeutral-venue matches to augment: {neutral_count} "
          f"({neutral_count / len(df) * 100:.1f}%)")

    if neutral_count == 0:
        return df

    flipped = df[neutral_mask].copy()

    # -- Swap teams --
    flipped["home_team"], flipped["away_team"] = (
        flipped["away_team"].values,
        flipped["home_team"].values,
    )
    flipped["home_team_id"], flipped["away_team_id"] = (
        flipped["away_team_id"].values,
        flipped["home_team_id"].values,
    )

    # -- Swap scores --
    flipped["home_score"], flipped["away_score"] = (
        flipped["away_score"].values,
        flipped["home_score"].values,
    )

    # -- Swap Elo --
    flipped["home_elo"], flipped["away_elo"] = (
        flipped["away_elo"].values,
        flipped["home_elo"].values,
    )
    flipped["home_elo_after"], flipped["away_elo_after"] = (
        flipped["away_elo_after"].values,
        flipped["home_elo_after"].values,
    )
    flipped["elo_diff"] = -flipped["elo_diff"]
    flipped["elo_advantage_home"] = -flipped["elo_advantage_home"]
    # elo_quality is symmetric, unchanged

    # -- Swap form --
    flipped["home_form"], flipped["away_form"] = (
        flipped["away_form"].values,
        flipped["home_form"].values,
    )
    flipped["form_advantage"] = -flipped["form_advantage"]
    # form_quality is symmetric, unchanged

    # -- Swap goal stats --
    flipped["home_goals_scored_avg"], flipped["away_goals_scored_avg"] = (
        flipped["away_goals_scored_avg"].values,
        flipped["home_goals_scored_avg"].values,
    )
    flipped["home_goals_conceded_avg"], flipped["away_goals_conceded_avg"] = (
        flipped["away_goals_conceded_avg"].values,
        flipped["home_goals_conceded_avg"].values,
    )
    flipped["gs_advantage"] = -flipped["gs_advantage"]
    flipped["gc_advantage"] = -flipped["gc_advantage"]

    # -- Swap win/draw rates --
    flipped["home_win_rate"], flipped["away_win_rate"] = (
        flipped["away_win_rate"].values,
        flipped["home_win_rate"].values,
    )
    flipped["home_draw_rate"], flipped["away_draw_rate"] = (
        flipped["away_draw_rate"].values,
        flipped["home_draw_rate"].values,
    )
    flipped["wr_advantage"] = -flipped["wr_advantage"]

    # -- Swap H2H --
    flipped["h2h_home_wins"], flipped["h2h_away_wins"] = (
        flipped["h2h_away_wins"].values,
        flipped["h2h_home_wins"].values,
    )
    flipped["h2h_home_advantage"] = -flipped["h2h_home_advantage"]
    # h2h_count, h2h_draws, has_h2h are symmetric, unchanged

    augmented = pd.concat([df, flipped], ignore_index=True)
    print(f"Augmented dataset: {len(df)} -> {len(augmented)} matches "
          f"(+{len(flipped)} flipped neutral matches)")
    return augmented


def feature_engineering_v2(df):
    """Same feature engineering as train_improved.py."""
    print("Engineering features...")
    df = df.copy()

    df["elo_diff_norm"] = df["elo_diff"] / 400.0
    df["elo_ratio"] = (df["home_elo"] / df["away_elo"].clip(lower=1000)) - 1.0
    df["elo_gap"] = abs(df["elo_diff"]) / 400.0

    df["home_goal_diff_avg"] = df["home_goals_scored_avg"] - df["home_goals_conceded_avg"]
    df["away_goal_diff_avg"] = df["away_goals_scored_avg"] - df["away_goals_conceded_avg"]
    df["goal_diff_advantage"] = df["home_goal_diff_avg"] - df["away_goal_diff_avg"]

    df["home_strength"] = df["home_elo"] / 1500.0 + df["home_win_rate"] * 0.5 + df["home_form"] * 0.3
    df["away_strength"] = df["away_elo"] / 1500.0 + df["away_win_rate"] * 0.5 + df["away_form"] * 0.3
    df["strength_advantage"] = df["home_strength"] - df["away_strength"]
    df["match_quality"] = (df["home_elo"] + df["away_elo"]) / 3000.0

    # Draw-specific features
    df["draw_rate_home"] = df["home_draw_rate"]
    df["draw_rate_away"] = df["away_draw_rate"]
    df["both_draw_prone"] = np.minimum(df["home_draw_rate"], df["away_draw_rate"])
    df["strength_parity"] = 1.0 / (1.0 + abs(df["elo_diff"]) / 100.0)
    df["defensive_similarity"] = 1.0 / (1.0 + abs(df["home_goals_conceded_avg"] - df["away_goals_conceded_avg"]))
    df["low_scoring_tendency"] = (
        (df["home_goals_scored_avg"] + df["home_goals_conceded_avg"] < 2.5)
        & (df["away_goals_scored_avg"] + df["away_goals_conceded_avg"] < 2.5)
    ).astype(float)

    df["h2h_dominance"] = np.where(
        df["h2h_count"] >= 3,
        (df["h2h_home_wins"] - df["h2h_away_wins"]) / df["h2h_count"],
        0,
    )

    df["year_norm"] = (df["year"] - 1950) / 80.0

    df["is_wc"] = df["tournament"].str.contains("FIFA World Cup", na=False).astype(int)
    df["is_wcq"] = df["tournament"].str.contains("qualification", na=False).astype(int)
    df["is_friendly"] = df["tournament"].str.contains("Friendly", na=False).astype(int)
    df["is_continental"] = (
        df["tournament"].str.contains(
            "UEFA Euro|Copa Am|African Cup|Asian Cup|Gold Cup|Nations League",
            na=False,
        )
    ).astype(int)
    df["is_neutral"] = df["neutral"].astype(int)

    df["result"] = np.where(
        df["home_score"] > df["away_score"],
        2,
        np.where(df["home_score"] == df["away_score"], 1, 0),
    )

    return df


def prepare_data(df, scaler, team_encoder, fit_scaler=False):
    """Prepare feature tensors."""
    feature_cols = [
        "elo_advantage_home", "elo_quality", "elo_diff_norm", "elo_ratio", "elo_gap",
        "form_advantage", "form_quality", "wr_advantage",
        "gs_advantage", "gc_advantage", "goal_diff_advantage",
        "strength_advantage", "match_quality",
        "h2h_dominance", "has_h2h",
        "draw_rate_home", "draw_rate_away", "both_draw_prone",
        "strength_parity", "defensive_similarity", "low_scoring_tendency",
        "is_neutral", "year_norm", "is_wc", "is_wcq", "is_continental", "is_friendly",
    ]

    X = df[feature_cols].fillna(0).values.astype(np.float32)

    if fit_scaler:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    # team_encoder already fitted, transform team names to IDs
    home_ids = team_encoder.transform(df["home_team"].values).astype(np.int64) + 1
    away_ids = team_encoder.transform(df["away_team"].values).astype(np.int64) + 1
    y = df["result"].values.astype(np.int64)
    home_goals = df["home_score"].values.astype(np.float32)
    away_goals = df["away_score"].values.astype(np.float32)

    if fit_scaler:
        return X, home_ids, away_ids, y, home_goals, away_goals, feature_cols, scaler
    return X, home_ids, away_ids, y, home_goals, away_goals, feature_cols, None


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for X, h_ids, a_ids, y, hg, ag in loader:
        X, h_ids, a_ids = X.to(device), h_ids.to(device), a_ids.to(device)
        y, hg, ag = y.to(device), hg.to(device), ag.to(device)

        optimizer.zero_grad()
        logits, goals = model(h_ids, a_ids, X, return_goals=True)
        loss, _, _ = criterion(logits, goals, y, hg, ag)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    n = len(loader)
    return total_loss / n, accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro")


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X, h_ids, a_ids, y, hg, ag in loader:
            X, h_ids, a_ids = X.to(device), h_ids.to(device), a_ids.to(device)
            y, hg, ag = y.to(device), hg.to(device), ag.to(device)

            logits, goals = model(h_ids, a_ids, X, return_goals=True)
            loss, _, _ = criterion(logits, goals, y, hg, ag)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    n = len(loader)
    return total_loss / n, accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro"), all_preds, all_labels, all_probs


def load_wc2022_group_matches():
    """Load WC 2022 group stage matches with actual results."""
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # WC 2022 group stage: 2022-11-20 to 2022-12-02
    wc = df[
        (df["date"] >= pd.to_datetime("2022-11-20"))
        & (df["date"] <= pd.to_datetime("2022-12-02"))
        & (df["tournament"].str.contains("FIFA World Cup", na=False))
    ].copy()

    print(f"\nWC 2022 Group Stage: {len(wc)} matches")
    return wc


def run_backtest():
    print("=" * 60)
    print("Backtesting: 2022 World Cup Group Stage")
    print("=" * 60)

    # Load pre-WC data
    df = load_and_filter_data()

    # Feature engineering
    df = feature_engineering_v2(df)

    # Chronological split (no future leakage)
    val_start = pd.to_datetime("2021-01-01")
    train_df = df[df["date"] < val_start].copy()
    val_df = df[df["date"] >= val_start].copy()

    print(f"\nTrain: {len(train_df)} matches ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"Val:   {len(val_df)} matches ({val_df['date'].min().date()} to {val_df['date'].max().date()})")

    # Fit team encoder on ALL teams seen before WC 2022
    # (knowing team names doesn't leak match outcomes)
    all_teams = pd.concat([df["home_team"], df["away_team"]]).unique()
    team_encoder = LabelEncoder()
    team_encoder.fit(all_teams)
    print(f"\nTeams before WC 2022: {len(team_encoder.classes_)}")

    # Prepare data
    X_train, h_train, a_train, y_train, hg_train, ag_train, feature_cols, scaler = \
        prepare_data(train_df, StandardScaler(), team_encoder, fit_scaler=True)

    X_val, h_val, a_val, y_val, hg_val, ag_val, _, _ = \
        prepare_data(val_df, scaler, team_encoder)

    print(f"Features ({len(feature_cols)}): {feature_cols}")

    is_neutral_idx = feature_cols.index("is_neutral")

    # Tensors
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

    # DataLoaders
    train_dataset = TensorDataset(X_train_t, h_train_t, a_train_t, y_train_t, hg_train_t, ag_train_t)
    val_dataset = TensorDataset(X_val_t, h_val_t, a_val_t, y_val_t, hg_val_t, ag_val_t)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Model
    num_teams = len(team_encoder.classes_)
    model = create_improved_model(num_teams, num_match_features=len(feature_cols), device=device, is_neutral_idx=is_neutral_idx)

    # Class weights
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights_t = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights}")

    # Loss & optimizer
    criterion = ImprovedLoss(class_weights=class_weights_t, goal_weight=0.15)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)

    # Training loop
    best_val_f1 = 0
    patience = 25
    patience_counter = 0

    print(f"\nTraining... (max 300 epochs)")
    for epoch in range(1, 301):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    print(f"\nBest val F1: {best_val_f1:.4f}")

    # ---- WC 2022 Group Stage Evaluation ----
    print("\n" + "=" * 60)
    print("WC 2022 Group Stage — Out-of-Sample Test")
    print("=" * 60)

    wc_df = load_wc2022_group_matches()
    wc_df = feature_engineering_v2(wc_df)

    # Handle unseen teams (teams in WC but not in training)
    unseen_teams = set(wc_df["home_team"].unique()) | set(wc_df["away_team"].unique())
    known_teams = set(team_encoder.classes_)
    new_teams = unseen_teams - known_teams
    if new_teams:
        print(f"\nWARNING: {len(new_teams)} teams in WC not seen in training: {new_teams}")
        print("Using mean embedding for unknown teams (ID=0).")

    # Map team IDs, unknown -> 0
    team_to_id = {t: i + 1 for i, t in enumerate(team_encoder.classes_)}
    wc_home_ids = np.array([team_to_id.get(t, 0) for t in wc_df["home_team"]], dtype=np.int64)
    wc_away_ids = np.array([team_to_id.get(t, 0) for t in wc_df["away_team"]], dtype=np.int64)

    X_wc = wc_df[feature_cols].fillna(0).values.astype(np.float32)
    X_wc = scaler.transform(X_wc)
    y_wc = wc_df["result"].values.astype(np.int64)

    # Predict
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_wc).to(device)
        h_t = torch.LongTensor(wc_home_ids).to(device)
        a_t = torch.LongTensor(wc_away_ids).to(device)
        logits = model(h_t, a_t, X_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    wc_acc = accuracy_score(y_wc, preds)
    wc_f1 = f1_score(y_wc, preds, average="macro")

    # Per-class recall
    cm = confusion_matrix(y_wc, preds)
    if cm.shape == (3, 3):
        draw_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0
        home_recall = cm[2, 2] / cm[2].sum() if cm[2].sum() > 0 else 0
        away_recall = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
    else:
        draw_recall = home_recall = away_recall = 0

    print(f"\nOverall Accuracy: {wc_acc:.4f} ({wc_acc*100:.1f}%)")
    print(f"Macro F1: {wc_f1:.4f}")
    print(f"Draw Recall: {draw_recall:.4f} ({draw_recall*100:.1f}%)")
    print(f"Home Win Recall: {home_recall:.4f} ({home_recall*100:.1f}%)")
    print(f"Away Win Recall: {away_recall:.4f} ({away_recall*100:.1f}%)")

    label_names = ["Away Win", "Draw", "Home Win"]
    print("\nClassification Report:")
    print(classification_report(y_wc, preds, target_names=label_names))
    print("Confusion Matrix:")
    print(pd.DataFrame(
        cm, index=label_names, columns=label_names
    ) if cm.shape == (3, 3) else cm)

    # ---- Per-match breakdown ----
    print("\n" + "=" * 60)
    print("Per-Match Predictions vs Actual")
    print("=" * 60)
    results = []
    for i in range(len(wc_df)):
        row = wc_df.iloc[i]
        actual = label_names[y_wc[i]]
        predicted = label_names[preds[i]]
        correct = "Y" if y_wc[i] == preds[i] else "N"
        home_prob = probs[i, 2]
        draw_prob = probs[i, 1]
        away_prob = probs[i, 0]
        results.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "home": row["home_team"],
            "away": row["away_team"],
            "actual": f"{int(row['home_score'])}-{int(row['away_score'])}",
            "result": actual,
            "predicted": predicted,
            "home_prob": f"{home_prob:.3f}",
            "draw_prob": f"{draw_prob:.3f}",
            "away_prob": f"{away_prob:.3f}",
            "correct": correct,
        })

    for r in results:
        marker = " [OK]" if r["correct"] == "Y" else "[WRONG]"
        print(f"  {r['date']} {r['home']:20s} vs {r['away']:20s}  Actual: {r['result']:9s}  Pred: {r['predicted']:9s}  "
              f"H:{r['home_prob']} D:{r['draw_prob']} A:{r['away_prob']} {marker}")

    # Summary
    n_correct = sum(1 for r in results if r["correct"] == "Y")
    print(f"\nCorrect: {n_correct}/{len(results)} ({n_correct/len(results)*100:.1f}%)")

    # Save results
    backtest_results = {
        "train_matches": len(train_df),
        "val_matches": len(val_df),
        "wcq_matches": len(wc_df),
        "best_val_f1": float(best_val_f1),
        "wc_accuracy": float(wc_acc),
        "wc_f1": float(wc_f1),
        "draw_recall": float(draw_recall),
        "home_recall": float(home_recall),
        "away_recall": float(away_recall),
        "confusion_matrix": cm.tolist() if cm.shape == (3, 3) else None,
        "match_details": results,
        "unseen_teams": list(new_teams) if new_teams else [],
    }

    output_path = os.path.join(RESULTS_DIR, "backtest_wc2022.json")
    with open(output_path, "w") as f:
        json.dump(backtest_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return backtest_results


if __name__ == "__main__":
    run_backtest()
