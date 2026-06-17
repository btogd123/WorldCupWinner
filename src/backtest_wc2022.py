"""
Backtest: Train on pre-tournament data, predict all final tournament group stages
from 2022 to 2024. Gold-standard out-of-sample evaluation across 6 tournaments.

Tournaments: AFCON 2021, Gold Cup 2023, AFC Asian Cup 2023, AFCON 2023,
             UEFA Euro 2024, Copa America 2024
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

# Tournament definitions: (name, tournament keyword, date range for the edition)
TOURNAMENTS = [
    ("AFCON 2021", "African Cup of Nations", "2022-01-01", "2022-06-01"),
    ("FIFA World Cup 2022", "FIFA World Cup", "2022-11-20", "2022-12-19"),
    ("Gold Cup 2023", "Gold Cup", "2023-01-01", "2023-12-31"),
    ("AFC Asian Cup 2023", "AFC Asian Cup", "2024-01-01", "2024-06-01"),
    ("AFCON 2023", "African Cup of Nations", "2024-01-01", "2024-06-01"),
    ("UEFA Euro 2024", "UEFA Euro", "2024-01-01", "2024-12-31"),
    ("Copa America 2024", "Copa América", "2024-01-01", "2024-12-31"),
]


def load_full_data():
    """Load processed data with actual results only."""
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.to_datetime("2000-01-01")]
    df = df.dropna(subset=["home_score", "away_score"])
    print(f"Total data: {len(df)} matches ({df['date'].min().date()} to {df['date'].max().date()})")
    return df


def extract_group_stage(df, name, keyword, date_start, date_end):
    """Extract group stage matches for a final tournament.

    A match is group stage if both teams have played < 3 matches in this
    edition before this match (4-team groups, 3 round-robin rounds).
    """
    mask = (
        (df["tournament"].str.contains(keyword, na=False, case=False))
        & ~df["tournament"].str.contains("qualification", na=False, case=False)
        & (df["date"] >= pd.to_datetime(date_start))
        & (df["date"] < pd.to_datetime(date_end))
    )
    edition = df[mask].sort_values("date").copy()

    if len(edition) == 0:
        return None

    # Count team appearances chronologically; only keep first 3 per team
    appearances = {}
    group_indices = []

    for idx, row in edition.iterrows():
        home, away = row["home_team"], row["away_team"]
        h_app = appearances.get(home, 0)
        a_app = appearances.get(away, 0)

        if h_app < 3 and a_app < 3:
            group_indices.append(idx)
            appearances[home] = h_app + 1
            appearances[away] = a_app + 1

    group_df = edition.loc[group_indices]
    total_teams = len(appearances)

    print(f"  {name}: {len(group_df)} group matches, {total_teams} teams, "
          f"{group_df['date'].min().date()} to {group_df['date'].max().date()}")
    return group_df


def feature_engineering_v2(df):
    """Same feature engineering as train_improved.py."""
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

    df["sim_wr_advantage"] = df["home_sim_win_rate"] - df["away_sim_win_rate"]
    df["sim_dr_advantage"] = df["home_sim_draw_rate"] - df["away_sim_draw_rate"]
    df["sim_gs_advantage"] = df["home_sim_gs"] - df["away_sim_gs"]
    df["sim_gc_advantage"] = df["home_sim_gc"] - df["away_sim_gc"]
    df["sim_wr_quality"] = (df["home_sim_win_rate"] + df["away_sim_win_rate"]) / 2
    df["sim_dr_quality"] = (df["home_sim_draw_rate"] + df["away_sim_draw_rate"]) / 2

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


def prepare_data(df, scaler, team_encoder, fit_scaler=False,
                 use_draw_features=True, use_group_round=True,
                 use_sim_features=True, extra_features=None):
    """Prepare feature tensors."""
    feature_cols = [
        "elo_advantage_home", "elo_quality", "elo_diff_norm", "elo_ratio", "elo_gap",
        "form_advantage", "form_quality", "wr_advantage",
        "gs_advantage", "gc_advantage", "goal_diff_advantage",
        "strength_advantage", "match_quality",
        "h2h_dominance", "has_h2h",
    ]
    if use_sim_features:
        feature_cols += [
            "sim_wr_advantage", "sim_gs_advantage",
            "sim_wr_quality", "sim_dr_quality",
        ]
    if extra_features:
        feature_cols += extra_features
    if use_draw_features:
        feature_cols += [
            "draw_rate_home", "draw_rate_away", "both_draw_prone",
            "strength_parity", "defensive_similarity", "low_scoring_tendency",
        ]
    if use_group_round:
        feature_cols.append("group_round")
    feature_cols += ["is_neutral", "year_norm", "is_wc", "is_wcq", "is_continental", "is_friendly"]

    X = df[feature_cols].fillna(0).values.astype(np.float32)

    if fit_scaler:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    # Map teams to IDs, assigning unseen teams to 0 (mean embedding)
    known = set(team_encoder.classes_)
    home_ids = np.array([team_encoder.transform([t])[0] if t in known else -1
                         for t in df["home_team"].values], dtype=np.int64) + 1
    away_ids = np.array([team_encoder.transform([t])[0] if t in known else -1
                         for t in df["away_team"].values], dtype=np.int64) + 1
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


def backtest_single_tournament(name, keyword, date_start, date_end, full_df,
                               use_draw_features=True, use_neutral_gating=True,
                               use_group_round=True, use_sim_features=True,
                               extra_features=None):
    """Run backtest for a single tournament: train on pre-tournament data,
    test on tournament group stage. Returns metrics dict or None."""
    config_desc = f"draw={use_draw_features} sim={use_sim_features} neutral_gate={use_neutral_gating} group_round={use_group_round} extra={extra_features}"
    print(f"\n{'=' * 60}")
    print(f"Backtesting: {name}  [{config_desc}]")
    print(f"{'=' * 60}")

    # Extract group stage matches
    group_df = extract_group_stage(full_df, name, keyword, date_start, date_end)
    if group_df is None or len(group_df) < 6:
        print(f"  Skipping {name}: insufficient group matches ({len(group_df) if group_df is not None else 0})")
        return None

    tournament_start = group_df["date"].min()

    # Training data: all matches before tournament start
    train_val = full_df[full_df["date"] < tournament_start].copy()

    # Chronological split: val = last 2 years before tournament
    val_start = tournament_start - pd.DateOffset(years=2)
    train_df = train_val[train_val["date"] < val_start].copy()
    val_df = train_val[train_val["date"] >= val_start].copy()

    if len(train_df) < 1000:
        print(f"  Skipping {name}: insufficient training data ({len(train_df)} matches)")
        return None

    print(f"  Train: {len(train_df)} ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"  Val:   {len(val_df)} ({val_df['date'].min().date()} to {val_df['date'].max().date()})")
    print(f"  Test:  {len(group_df)} ({group_df['date'].min().date()} to {group_df['date'].max().date()})")

    # Feature engineering
    train_df = feature_engineering_v2(train_df)
    val_df = feature_engineering_v2(val_df)
    test_df = feature_engineering_v2(group_df)

    # Fit team encoder on all teams seen before tournament
    all_teams = pd.concat([train_df["home_team"], train_df["away_team"]]).unique()
    team_encoder = LabelEncoder()
    team_encoder.fit(all_teams)

    # Prepare data
    X_train, h_train, a_train, y_train, hg_train, ag_train, feature_cols, scaler = \
        prepare_data(train_df, StandardScaler(), team_encoder, fit_scaler=True,
                     use_draw_features=use_draw_features, use_group_round=use_group_round,
                     use_sim_features=use_sim_features, extra_features=extra_features)

    X_val, h_val, a_val, y_val, hg_val, ag_val, _, _ = \
        prepare_data(val_df, scaler, team_encoder,
                     use_draw_features=use_draw_features, use_group_round=use_group_round,
                     use_sim_features=use_sim_features, extra_features=extra_features)

    # Handle unseen teams in test
    known_teams = set(team_encoder.classes_)
    test_teams = set(test_df["home_team"].unique()) | set(test_df["away_team"].unique())
    new_teams = test_teams - known_teams

    team_to_id = {t: i + 1 for i, t in enumerate(team_encoder.classes_)}
    test_home_ids = np.array([team_to_id.get(t, 0) for t in test_df["home_team"]], dtype=np.int64)
    test_away_ids = np.array([team_to_id.get(t, 0) for t in test_df["away_team"]], dtype=np.int64)
    X_test = test_df[feature_cols].fillna(0).values.astype(np.float32)
    X_test = scaler.transform(X_test)
    y_test = test_df["result"].values.astype(np.int64)

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

    # Model
    num_teams = len(team_encoder.classes_)
    model = create_improved_model(num_teams, num_match_features=len(feature_cols),
                                   device=device, is_neutral_idx=is_neutral_idx,
                                   use_neutral_gating=use_neutral_gating)

    # Class weights
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights_t = torch.FloatTensor(class_weights).to(device)

    # Loss & optimizer
    criterion = ImprovedLoss(class_weights=class_weights_t, goal_weight=0.15)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)

    # Training loop
    best_val_f1 = 0
    patience = 25
    patience_counter = 0

    for epoch in range(1, 301):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)

    # ---- Test on tournament group stage ----
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(device)
        h_t = torch.LongTensor(test_home_ids).to(device)
        a_t = torch.LongTensor(test_away_ids).to(device)
        logits = model(h_t, a_t, X_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    cm = confusion_matrix(y_test, preds)
    if cm.shape == (3, 3):
        draw_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0
        home_recall = cm[2, 2] / cm[2].sum() if cm[2].sum() > 0 else 0
        away_recall = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
    else:
        draw_recall = home_recall = away_recall = 0

    print(f"  Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Macro F1: {f1:.4f}")
    print(f"  Draw Recall: {draw_recall*100:.1f}%  Home: {home_recall*100:.1f}%  Away: {away_recall*100:.1f}%")

    # Build match details
    label_names = ["Away Win", "Draw", "Home Win"]
    match_details = []
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        match_details.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "home": row["home_team"],
            "away": row["away_team"],
            "actual": f"{int(row['home_score'])}-{int(row['away_score'])}",
            "result": label_names[y_test[i]],
            "predicted": label_names[preds[i]],
            "home_prob": f"{probs[i, 2]:.3f}",
            "draw_prob": f"{probs[i, 1]:.3f}",
            "away_prob": f"{probs[i, 0]:.3f}",
            "correct": "Y" if y_test[i] == preds[i] else "N",
        })

    return {
        "name": name,
        "train_matches": len(train_df),
        "val_matches": len(val_df),
        "test_matches": len(test_df),
        "unseen_teams": list(new_teams) if new_teams else [],
        "best_val_f1": float(best_val_f1),
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "draw_recall": float(draw_recall),
        "home_recall": float(home_recall),
        "away_recall": float(away_recall),
        "confusion_matrix": cm.tolist() if cm.shape == (3, 3) else None,
        "match_details": match_details,
    }


def run_backtest():
    print("=" * 60)
    print("Multi-Tournament Backtest: 2022-2024 Final Tournament Group Stages")
    print("=" * 60)

    full_df = load_full_data()

    # Extract all group stages first for overview
    print("\n--- Tournament Group Stages ---")
    all_group_dfs = {}
    for name, keyword, ds, de in TOURNAMENTS:
        group_df = extract_group_stage(full_df, name, keyword, ds, de)
        if group_df is not None and len(group_df) >= 6:
            all_group_dfs[name] = group_df

    total_group_matches = sum(len(g) for g in all_group_dfs.values())
    print(f"\nTotal group stage matches across {len(all_group_dfs)} tournaments: {total_group_matches}")

    # Run backtest for each tournament
    all_results = []
    for name, keyword, ds, de in TOURNAMENTS:
        result = backtest_single_tournament(name, keyword, ds, de, full_df)
        if result is not None:
            all_results.append(result)

    # ---- Aggregate Results ----
    print("\n" + "=" * 60)
    print("AGGREGATE BACKTEST RESULTS")
    print("=" * 60)

    total_test = sum(r["test_matches"] for r in all_results)
    n_total_correct = sum(
        sum(1 for m in r["match_details"] if m["correct"] == "Y")
        for r in all_results
    )
    overall_acc = n_total_correct / total_test if total_test > 0 else 0

    # Macro-average metrics across tournaments
    avg_acc = np.mean([r["accuracy"] for r in all_results])
    avg_f1 = np.mean([r["macro_f1"] for r in all_results])
    avg_draw = np.mean([r["draw_recall"] for r in all_results])
    avg_home = np.mean([r["home_recall"] for r in all_results])
    avg_away = np.mean([r["away_recall"] for r in all_results])

    print(f"\nOverall: {n_total_correct}/{total_test} correct ({overall_acc*100:.1f}%)")
    print(f"Average across tournaments:")
    print(f"  Accuracy:    {avg_acc*100:.1f}%")
    print(f"  Macro F1:    {avg_f1:.4f}")
    print(f"  Draw Recall: {avg_draw*100:.1f}%")
    print(f"  Home Recall: {avg_home*100:.1f}%")
    print(f"  Away Recall: {avg_away*100:.1f}%")

    print(f"\n{'Tournament':25s} {'Matches':>8s} {'Acc':>7s} {'F1':>7s} {'Draw':>7s} {'Home':>7s} {'Away':>7s}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['name']:25s} {r['test_matches']:>8d} {r['accuracy']*100:>6.1f}% {r['macro_f1']:>7.4f} "
              f"{r['draw_recall']*100:>6.1f}% {r['home_recall']*100:>6.1f}% {r['away_recall']*100:>6.1f}%")

    # Aggregate confusion matrix
    agg_cm = np.zeros((3, 3), dtype=int)
    for r in all_results:
        if r["confusion_matrix"] is not None:
            agg_cm += np.array(r["confusion_matrix"])

    label_names = ["Away Win", "Draw", "Home Win"]
    print("\nAggregate Confusion Matrix:")
    print(pd.DataFrame(agg_cm, index=label_names, columns=label_names))

    # ---- Per-match listing ----
    print("\n" + "=" * 60)
    print("PER-TOURNAMENT MATCH DETAILS")
    print("=" * 60)
    for r in all_results:
        print(f"\n--- {r['name']} ({r['accuracy']*100:.1f}%) ---")
        for m in r["match_details"]:
            marker = " [OK]" if m["correct"] == "Y" else "[WRONG]"
            print(f"  {m['date']} {m['home']:20s} vs {m['away']:20s}  "
                  f"Actual: {m['result']:9s}  Pred: {m['predicted']:9s}  "
                  f"H:{m['home_prob']} D:{m['draw_prob']} A:{m['away_prob']} {marker}")

    # Show teams that were unseen per tournament
    for r in all_results:
        if r["unseen_teams"]:
            print(f"\n{r['name']} — Unseen teams: {r['unseen_teams']}")

    # Save results
    output = {
        "backtest_date": datetime.now().strftime("%Y-%m-%d"),
        "total_test_matches": total_test,
        "overall_accuracy": float(overall_acc),
        "avg_accuracy": float(avg_acc),
        "avg_macro_f1": float(avg_f1),
        "avg_draw_recall": float(avg_draw),
        "avg_home_recall": float(avg_home),
        "avg_away_recall": float(avg_away),
        "aggregate_confusion_matrix": agg_cm.tolist(),
        "tournaments": all_results,
    }

    output_path = os.path.join(RESULTS_DIR, "backtest_wc2022.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return output


def run_ablation():
    """Ablation study: measure marginal contribution of each improvement.

    Tests 4 configurations:
      1. Baseline (no draw features, no neutral gating, no group_round)
      2. +Draw features only
      3. +Draw features + Neutral gating
      4. +Draw features + Neutral gating + group_round (current best)
    """
    print("=" * 60)
    print("ABLATION STUDY: Marginal contribution of each improvement")
    print("=" * 60)

    full_df = load_full_data()

    configs = [
        ("A: Baseline",        False, False, False),
        ("B: +Draw features",  True,  False, False),
        ("C: +Neutral gating", True,  True,  False),
        ("D: +group_round",    True,  True,  True),
    ]

    all_config_results = {}

    for label, use_draw, use_neutral, use_group in configs:
        print(f"\n{'#' * 60}")
        print(f"# CONFIG: {label}")
        print(f"{'#' * 60}")

        results = []
        for name, keyword, ds, de in TOURNAMENTS:
            result = backtest_single_tournament(
                name, keyword, ds, de, full_df,
                use_draw_features=use_draw,
                use_neutral_gating=use_neutral,
                use_group_round=use_group,
            )
            if result is not None:
                results.append(result)

        all_config_results[label] = results

    # ---- Comparison Table ----
    print("\n\n" + "=" * 80)
    print("ABLATION RESULTS — Per-Configuration Summary")
    print("=" * 80)

    for label, results in all_config_results.items():
        total = sum(r["test_matches"] for r in results)
        n_correct = sum(
            sum(1 for m in r["match_details"] if m["correct"] == "Y")
            for r in results
        )
        overall = n_correct / total * 100 if total > 0 else 0
        avg_acc = np.mean([r["accuracy"] for r in results]) * 100
        avg_f1 = np.mean([r["macro_f1"] for r in results])
        avg_draw = np.mean([r["draw_recall"] for r in results]) * 100
        print(f"\n{label}: {n_correct}/{total} correct ({overall:.1f}%), "
              f"avg acc={avg_acc:.1f}%, F1={avg_f1:.4f}, draw recall={avg_draw:.1f}%")

    # ---- Marginal Gains ----
    print("\n\n" + "=" * 80)
    print("MARGINAL CONTRIBUTION OF EACH IMPROVEMENT")
    print("=" * 80)

    prev_label = None
    for label, results in all_config_results.items():
        total = sum(r["test_matches"] for r in results)
        n_correct = sum(
            sum(1 for m in r["match_details"] if m["correct"] == "Y")
            for r in results
        )
        overall = n_correct / total * 100 if total > 0 else 0
        avg_acc = np.mean([r["accuracy"] for r in results]) * 100
        avg_f1 = np.mean([r["macro_f1"] for r in results])
        avg_draw = np.mean([r["draw_recall"] for r in results]) * 100

        if prev_label:
            prev_total = sum(r["test_matches"] for r in all_config_results[prev_label])
            prev_correct = sum(
                sum(1 for m in r["match_details"] if m["correct"] == "Y")
                for r in all_config_results[prev_label]
            )
            prev_overall = prev_correct / prev_total * 100 if prev_total > 0 else 0
            prev_acc = np.mean([r["accuracy"] for r in all_config_results[prev_label]]) * 100
            prev_f1 = np.mean([r["macro_f1"] for r in all_config_results[prev_label]])
            prev_draw = np.mean([r["draw_recall"] for r in all_config_results[prev_label]]) * 100

            improvement = label.split(": ")[1]
            print(f"\n{improvement}:")
            print(f"  Overall acc:  {prev_overall:.1f}% → {overall:.1f}%  (Δ {overall - prev_overall:+.1f}%)")
            print(f"  Avg acc:      {prev_acc:.1f}% → {avg_acc:.1f}%  (Δ {avg_acc - prev_acc:+.1f}%)")
            print(f"  Macro F1:     {prev_f1:.4f} → {avg_f1:.4f}  (Δ {avg_f1 - prev_f1:+.4f})")
            print(f"  Draw recall:  {prev_draw:.1f}% → {avg_draw:.1f}%  (Δ {avg_draw - prev_draw:+.1f}%)")

        prev_label = label

    # ---- Per-Tournament Comparison ----
    print("\n\n" + "=" * 80)
    print("PER-TOURNAMENT ACCURACY COMPARISON")
    print("=" * 80)

    config_labels = [c[0] for c in configs]
    header = f"{'Tournament':25s}"
    for cl in config_labels:
        header += f" {cl:>18s}"
    print(header)
    print("-" * (25 + 18 * len(config_labels)))

    for t_idx, (name, _, _, _) in enumerate(TOURNAMENTS):
        row = f"{name:25s}"
        for label, results in all_config_results.items():
            if t_idx < len(results):
                r = results[t_idx]
                row += f" {r['accuracy']*100:>6.1f}% (D:{r['draw_recall']*100:>4.0f}%)"
            else:
                row += f" {'—':>18s}"
        print(row)

    # Save
    output = {
        "ablation_date": datetime.now().strftime("%Y-%m-%d"),
        "configs": {},
    }
    for label, results in all_config_results.items():
        output["configs"][label] = {
            "total_correct": sum(
                sum(1 for m in r["match_details"] if m["correct"] == "Y")
                for r in results
            ),
            "total_matches": sum(r["test_matches"] for r in results),
            "avg_accuracy": float(np.mean([r["accuracy"] for r in results])),
            "avg_f1": float(np.mean([r["macro_f1"] for r in results])),
            "avg_draw_recall": float(np.mean([r["draw_recall"] for r in results])),
            "tournaments": results,
        }

    output_path = os.path.join(RESULTS_DIR, "ablation_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_ablation()
