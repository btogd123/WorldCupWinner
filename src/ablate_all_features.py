"""
Comprehensive feature ablation: group-level first, then individual features
in any group where removal improves Brier/ECE.
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
import sys
import random
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_PATH, TEAM_ENCODER_PATH, BATCH_SIZE, RESULTS_DIR
from improved_model import create_improved_model, ImprovedLoss

SEED = 99

ALL_FEATURE_COLS = [
    "elo_advantage_home", "elo_quality", "elo_diff_norm", "elo_ratio", "elo_gap",
    "form_advantage", "form_quality", "wr_advantage",
    "gs_advantage", "gc_advantage", "goal_diff_advantage",
    "strength_advantage", "match_quality",
    "h2h_dominance", "has_h2h",
    "sim_wr_advantage", "sim_gs_advantage",
    "sim_wr_quality", "sim_dr_quality",
    "draw_rate_home", "draw_rate_away", "both_draw_prone",
    "strength_parity", "defensive_similarity", "low_scoring_tendency",
    "is_neutral", "year_norm", "is_wc", "is_wcq", "is_continental", "is_friendly",
]

# Feature groups for phase-1 ablation
FEATURE_GROUPS = {
    "Elo (5)": ["elo_advantage_home", "elo_quality", "elo_diff_norm", "elo_ratio", "elo_gap"],
    "Form (3)": ["form_advantage", "form_quality", "wr_advantage"],
    "Goals (3)": ["gs_advantage", "gc_advantage", "goal_diff_advantage"],
    "Strength (2)": ["strength_advantage", "match_quality"],
    "H2H (2)": ["h2h_dominance", "has_h2h"],
    "Sim (4)": ["sim_wr_advantage", "sim_gs_advantage",
                "sim_wr_quality", "sim_dr_quality"],
    "Draw (6)": ["draw_rate_home", "draw_rate_away", "both_draw_prone",
                 "strength_parity", "defensive_similarity", "low_scoring_tendency"],
    # is_neutral is structural (neutral venue gating) — cannot be removed
    "Context (5)": ["year_norm", "is_wc", "is_wcq", "is_continental", "is_friendly"],
}

STRUCTURAL_FEATURES = {"is_neutral"}


def load_and_prepare():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    with open(TEAM_ENCODER_PATH, "rb") as f:
        team_encoder = pickle.load(f)
    return df, team_encoder


def feature_engineering_v2(df):
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
        (df["h2h_home_wins"] - df["h2h_away_wins"]) / df["h2h_count"], 0)
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
    df["is_continental"] = df["tournament"].str.contains(
        "UEFA Euro|Copa Am|African Cup|Asian Cup|Gold Cup|Nations League", na=False).astype(int)
    df["is_neutral"] = df["neutral"].astype(int)
    df["result"] = np.where(df["home_score"] > df["away_score"], 2,
                            np.where(df["home_score"] == df["away_score"], 1, 0))
    return df


def split_data(df, train_end="2021-12-31", val_end="2023-12-31"):
    df = df[df["date"] >= pd.to_datetime("2000-01-01")]
    df = df[df["date"] <= pd.to_datetime("2026-06-10")]
    df = df.dropna(subset=["home_score", "away_score"])
    train = df[df["date"] < pd.to_datetime(train_end)]
    val = df[(df["date"] >= pd.to_datetime(train_end)) & (df["date"] < pd.to_datetime(val_end))]
    test = df[df["date"] >= pd.to_datetime(val_end)]
    return train, val, test


def prepare_data(df, feature_cols, scaler, fit_scaler=False):
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
    return X, home_ids, away_ids, y, home_goals, away_goals, scaler


def compute_brier_ece(probs, labels):
    probs = np.array(probs, dtype=np.float64)
    labels = np.array(labels)
    n = len(labels)
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(n), labels] = 1
    brier = np.mean(np.sum((probs - y_onehot) ** 2, axis=1))
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels).astype(float)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() > 0:
            bin_acc = correct[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return brier, ece


def train_and_evaluate(feature_cols, train_df, val_df, test_df, team_encoder):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, h_train, a_train, y_train, hg_train, ag_train, scaler = prepare_data(
        train_df, feature_cols, None, fit_scaler=True)
    X_val, h_val, a_val, y_val, hg_val, ag_val, _ = prepare_data(
        val_df, feature_cols, scaler)
    X_test, h_test, a_test, y_test, hg_test, ag_test, _ = prepare_data(
        test_df, feature_cols, scaler)

    assert "is_neutral" in feature_cols, "is_neutral is structural and must be present"
    is_neutral_idx = feature_cols.index("is_neutral")

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(h_train), torch.LongTensor(a_train),
        torch.LongTensor(y_train), torch.FloatTensor(hg_train), torch.FloatTensor(ag_train))
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(h_val), torch.LongTensor(a_val),
        torch.LongTensor(y_val), torch.FloatTensor(hg_val), torch.FloatTensor(ag_val))
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), torch.LongTensor(h_test), torch.LongTensor(a_test),
        torch.LongTensor(y_test), torch.FloatTensor(hg_test), torch.FloatTensor(ag_test))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

    num_teams = len(team_encoder.classes_)
    model = create_improved_model(
        num_teams, num_match_features=len(feature_cols),
        device=device, is_neutral_idx=is_neutral_idx)

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights_t = torch.FloatTensor(class_weights).to(device)

    criterion = ImprovedLoss(class_weights=class_weights_t, goal_weight=0.15)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6)

    best_val_f1 = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, 301):
        model.train()
        for X, h_ids, a_ids, y, hg, ag in train_loader:
            X, h_ids, a_ids = X.to(device), h_ids.to(device), a_ids.to(device)
            y, hg, ag = y.to(device), hg.to(device), ag.to(device)
            optimizer.zero_grad()
            logits, goals = model(h_ids, a_ids, X, return_goals=True)
            loss, _, _ = criterion(logits, goals, y, hg, ag)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for X, h_ids, a_ids, y, _, _ in val_loader:
                X, h_ids, a_ids = X.to(device), h_ids.to(device), a_ids.to(device)
                logits = model(h_ids, a_ids, X)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels_list.extend(y.numpy())
        val_f1 = f1_score(val_labels_list, val_preds, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= 25:
            break

    model.load_state_dict(best_state)
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, h_ids, a_ids, y, _, _ in test_loader:
            X, h_ids, a_ids = X.to(device), h_ids.to(device), a_ids.to(device)
            logits = model(h_ids, a_ids, X)
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())

    all_probs = np.array(all_probs, dtype=np.float64)
    all_labels = np.array(all_labels)
    all_preds = np.argmax(all_probs, axis=1)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    brier, ece = compute_brier_ece(all_probs, all_labels)
    draw_recall = recall_score(all_labels, all_preds, labels=[1], average="macro", zero_division=0)
    home_recall = recall_score(all_labels, all_preds, labels=[2], average="macro", zero_division=0)
    away_recall = recall_score(all_labels, all_preds, labels=[0], average="macro", zero_division=0)

    return {
        "n_features": len(feature_cols),
        "accuracy": acc, "macro_f1": f1, "brier": brier, "ece": ece,
        "draw_recall": draw_recall, "home_recall": home_recall, "away_recall": away_recall,
        "best_val_f1": best_val_f1,
    }


def print_table(results, baseline, title):
    print(f"\n{'=' * 110}")
    print(title)
    print(f"{'=' * 110}")
    print(f"{'Config':40s} {'Feats':>5s} {'Acc':>7s} {'F1':>7s} {'Brier':>7s} {'ECE':>7s} "
          f"{'DrawR':>7s} {'HomeR':>7s} {'AwayR':>7s} {'ΔBrier':>9s} {'ΔECE':>9s}")
    print("-" * 115)
    for r in results:
        dbrier = r["brier"] - baseline["brier"]
        dece = r["ece"] - baseline["ece"]
        marker = " ***" if dbrier < -0.002 else ""
        print(f"{r['label']:40s} {r['n_features']:>5d} "
              f"{r['accuracy']*100:>6.1f}% {r['macro_f1']:>7.4f} "
              f"{r['brier']:>7.4f} {r['ece']:>7.4f} "
              f"{r['draw_recall']*100:>6.1f}% {r['home_recall']*100:>6.1f}% {r['away_recall']*100:>6.1f}% "
              f"{dbrier:>+9.4f} {dece:>+9.4f}{marker}")


def main():
    print("=" * 70)
    print("Comprehensive Feature Ablation Study")
    print("=" * 70)

    df, team_encoder = load_and_prepare()
    df = feature_engineering_v2(df)
    train_df, val_df, test_df = split_data(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ============================================================
    # Phase 1: Group-level ablation
    # ============================================================
    print("\n\n" + "#" * 70)
    print("# PHASE 1: Group-Level Ablation")
    print("#" * 70)

    group_results = []

    # Baseline: all features
    print(f"\n[1/9] Baseline (all {len(ALL_FEATURE_COLS)} features)")
    baseline = train_and_evaluate(ALL_FEATURE_COLS, train_df, val_df, test_df, team_encoder)
    baseline["label"] = "ALL FEATURES (baseline)"
    group_results.append(baseline)
    print(f"  Acc={baseline['accuracy']*100:.1f}%  F1={baseline['macro_f1']:.4f}  "
          f"Brier={baseline['brier']:.4f}  ECE={baseline['ece']:.4f}")

    # Remove each group
    for i, (group_name, group_features) in enumerate(FEATURE_GROUPS.items()):
        cols = [c for c in ALL_FEATURE_COLS if c not in group_features]
        print(f"\n[{i+2}/9] Remove {group_name} ({len(cols)} features)")

        result = train_and_evaluate(cols, train_df, val_df, test_df, team_encoder)
        result["label"] = f"Remove {group_name}"
        group_results.append(result)
        print(f"  Acc={result['accuracy']*100:.1f}%  F1={result['macro_f1']:.4f}  "
              f"Brier={result['brier']:.4f}  ECE={result['ece']:.4f}")

    # Sort by Brier
    group_results.sort(key=lambda r: r["brier"])
    print_table(group_results, baseline, "GROUP ABLATION RESULTS (sorted by Brier)")

    # ============================================================
    # Phase 2: Individual feature ablation for groups that improved Brier
    # ============================================================
    candidates = [r for r in group_results if r["brier"] < baseline["brier"] - 0.001]

    if candidates:
        print("\n\n" + "#" * 70)
        print("# PHASE 2: Individual Feature Ablation")
        print("#   (drilling into groups where removal improved Brier)")
        print("#" * 70)

        individual_results = [baseline]

        for candidate in candidates:
            label = candidate["label"]
            # Extract group name from label
            group_name = label.replace("Remove ", "")
            if group_name not in FEATURE_GROUPS:
                continue
            group_features = FEATURE_GROUPS[group_name]
            print(f"\n--- Drilling into: {group_name} ({len(group_features)} features) ---")

            for feat in group_features:
                cols = [c for c in ALL_FEATURE_COLS if c != feat]
                print(f"  Remove {feat} ({len(cols)} features)")

                result = train_and_evaluate(cols, train_df, val_df, test_df, team_encoder)
                result["label"] = f"Remove {feat}"
                individual_results.append(result)
                print(f"    Acc={result['accuracy']*100:.1f}%  F1={result['macro_f1']:.4f}  "
                      f"Brier={result['brier']:.4f}  ECE={result['ece']:.4f}")

        individual_results.sort(key=lambda r: r["brier"])
        print_table(individual_results, baseline, "INDIVIDUAL FEATURE ABLATION RESULTS (sorted by Brier)")
    else:
        print("\n\nNo group removal improved Brier. All feature groups are beneficial.")

    # ============================================================
    # Phase 3: Always test individual features that are suspicious
    # (highly correlated features, potential noise)
    # ============================================================
    print("\n\n" + "#" * 70)
    print("# PHASE 3: Suspicious Individual Features")
    print("#   (redundant/correlated features that may add noise)")
    print("#" * 70)

    # Features worth checking individually even if group-level didn't improve:
    # - Redundant Elo features (elo_ratio, elo_gap may overlap with elo_diff_norm)
    # - goal_diff_advantage (composite of gs_advantage + gc_advantage)
    # - strength_advantage (composite of Elo + form)
    # - match_quality (may not add signal beyond individual Elos)
    # - has_h2h (binary, low variance)
    # - low_scoring_tendency (binary, may be noisy)
    # - is_friendly (may not matter given is_wc/is_wcq/is_continental)
    suspicious = [
        "elo_ratio", "elo_gap", "goal_diff_advantage",
        "match_quality", "has_h2h", "low_scoring_tendency",
        "is_friendly",
    ]

    # Only test features we haven't already tested in phase 2
    already_tested = set()
    if candidates:
        for r in individual_results:
            if r["label"].startswith("Remove "):
                already_tested.add(r["label"].replace("Remove ", ""))

    to_test = [f for f in suspicious if f not in already_tested]

    if to_test:
        suspicious_results = [baseline]
        for feat in to_test:
            cols = [c for c in ALL_FEATURE_COLS if c != feat]
            print(f"  Remove {feat} ({len(cols)} features)")

            result = train_and_evaluate(cols, train_df, val_df, test_df, team_encoder)
            result["label"] = f"Remove {feat}"
            suspicious_results.append(result)
            print(f"    Acc={result['accuracy']*100:.1f}%  F1={result['macro_f1']:.4f}  "
                  f"Brier={result['brier']:.4f}  ECE={result['ece']:.4f}")

        suspicious_results.sort(key=lambda r: r["brier"])
        print_table(suspicious_results, baseline, "SUSPICIOUS FEATURE RESULTS (sorted by Brier)")

    # ============================================================
    # Final Summary
    # ============================================================
    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    # Collect all results
    all_results = {r["label"]: r for r in group_results}
    if candidates:
        for r in individual_results:
            all_results[r["label"]] = r
    if to_test:
        for r in suspicious_results:
            all_results[r["label"]] = r

    better_than_baseline = [
        (label, r) for label, r in all_results.items()
        if r["brier"] < baseline["brier"] - 0.001
    ]
    better_than_baseline.sort(key=lambda x: x[1]["brier"])

    if better_than_baseline:
        print("\nConfigs that IMPROVE Brier over baseline:")
        for label, r in better_than_baseline:
            dbrier = r["brier"] - baseline["brier"]
            dece = r["ece"] - baseline["ece"]
            print(f"  {label}: Brier {r['brier']:.4f} (Δ={dbrier:+.4f}), ECE {r['ece']:.4f} (Δ={dece:+.4f})")
    else:
        print("\nNo feature removal improves Brier score over the baseline.")
        print("All 33 features are contributing positively to probability calibration.")

    print(f"\nBaseline: Acc={baseline['accuracy']*100:.1f}%  F1={baseline['macro_f1']:.4f}  "
          f"Brier={baseline['brier']:.4f}  ECE={baseline['ece']:.4f}")

    # Save
    output = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "baseline": {k: v for k, v in baseline.items() if k != "label"},
        "group_results": [{k: v for k, v in r.items()} for r in group_results],
    }
    if candidates:
        output["individual_results"] = [{k: v for k, v in r.items()} for r in individual_results]
    if to_test:
        output["suspicious_results"] = [{k: v for k, v in r.items()} for r in suspicious_results]
    output_path = os.path.join(RESULTS_DIR, "all_feature_ablation.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
