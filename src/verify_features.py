"""
Comprehensive feature pipeline verification.
Checks every step from raw data to model input for correctness.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from config import RAW_RESULTS_PATH, PROCESSED_DATA_PATH, ELO_RATINGS_PATH

print("=" * 70)
print("STEP 1: RAW DATA INTEGRITY")
print("=" * 70)

raw = pd.read_csv(RAW_RESULTS_PATH)
raw["date"] = pd.to_datetime(raw["date"])
print(f"Total rows: {len(raw)}")
print(f"Date range: {raw['date'].min()} to {raw['date'].max()}")
print(f"Unique teams: {raw['home_team'].nunique()} home, {raw['away_team'].nunique()} away")

# Check for duplicate matches
dupes = raw.duplicated(subset=["date", "home_team", "away_team"], keep=False)
if dupes.sum() > 0:
    print(f"\n⚠ DUPLICATE MATCHES: {dupes.sum()} rows")
    print(raw[dupes].sort_values(["date", "home_team"]).head(20))
else:
    print("✓ No duplicate matches found")

# WC 2026 fixtures
wc2026 = raw[raw["date"] >= pd.to_datetime("2026-01-01")]
print(f"\nWC 2026 fixtures: {len(wc2026)} matches")
played = wc2026[wc2026["home_score"].notna() & wc2026["away_score"].notna()]
unplayed = wc2026[wc2026["home_score"].isna() | wc2026["away_score"].isna()]
print(f"  Played (have scores): {len(played)}")
print(f"  Unplayed (no scores): {len(unplayed)}")

if len(played) > 0:
    print(f"\n  Played WC 2026 matches:")
    for _, r in played.iterrows():
        print(f"    {r['date'].date()} {r['home_team']} {int(r['home_score'])}-{int(r['away_score'])} {r['away_team']} ({r['tournament']})")

# Check score distributions
scored = raw[raw["home_score"].notna() & raw["away_score"].notna()]
home_win = (scored["home_score"] > scored["away_score"]).sum()
draw = (scored["home_score"] == scored["away_score"]).sum()
away_win = (scored["home_score"] < scored["away_score"]).sum()
total = len(scored)
print(f"\nScore distribution (scored matches = {total}):")
print(f"  Home Win: {home_win} ({home_win/total*100:.1f}%)")
print(f"  Draw:     {draw} ({draw/total*100:.1f}%)")
print(f"  Away Win: {away_win} ({away_win/total*100:.1f}%)")

print("\n" + "=" * 70)
print("STEP 2: PROCESSED DATA CHECK")
print("=" * 70)

proc = pd.read_csv(PROCESSED_DATA_PATH)
proc["date"] = pd.to_datetime(proc["date"])
print(f"Total rows: {len(proc)}")
print(f"Columns: {len(proc.columns)}")
print(f"Date range: {proc['date'].min()} to {proc['date'].max()}")

# Check result column
print(f"\nResult distribution:")
print(proc["result"].value_counts().sort_index())
print(f"  0=Away Win, 1=Draw, 2=Home Win")

# Check for NaN in critical columns
critical = ["home_elo", "away_elo", "home_elo_after", "away_elo_after",
            "home_team_id", "away_team_id", "result"]
for col in critical:
    if col in proc.columns:
        nans = proc[col].isna().sum()
        if nans > 0:
            print(f"⚠ {col}: {nans} NaN values")
        else:
            print(f"✓ {col}: no NaN")

print("\n" + "=" * 70)
print("STEP 3: ELO CALCULATION VERIFICATION")
print("=" * 70)

# Check Elo after matches
elo = pd.read_csv(ELO_RATINGS_PATH)
print(f"Teams in Elo file: {len(elo)}")
print("Top 15 teams by Elo:")
for i, row in elo.sort_values("elo_rating", ascending=False).head(15).iterrows():
    print(f"  {row['team']:25s} {row['elo_rating']:8.1f}")

# Verify: after a match, home_elo_after = process_elo(home_elo, away_elo, result)
# Spot check a few matches
scored_proc = proc[proc["home_score"].notna() & proc["away_score"].notna()].copy()
scored_proc = scored_proc[scored_proc["home_elo"].notna() & scored_proc["away_elo"].notna()]

# Check last 5 scored matches for Elo consistency
print("\nLast 5 scored matches - Elo before/after check:")
for _, r in scored_proc.tail(5).iterrows():
    elo_diff_before = r["home_elo"] - r["away_elo"]
    elo_diff_after = r["home_elo_after"] - r["away_elo_after"]
    result_str = "H" if r["result"] == 2 else ("D" if r["result"] == 1 else "A")
    print(f"  {r['date'].date()} {r['home_team']:20s} vs {r['away_team']:20s} "
          f"{int(r['home_score'])}-{int(r['away_score'])} ({result_str}) | "
          f"Elo diff: {elo_diff_before:+.0f} → {elo_diff_after:+.0f}")

# Check Elo drift - average Elo should be ~1500
all_home = proc["home_elo"].dropna()
all_away = proc["away_elo"].dropna()
all_elos = pd.concat([all_home, all_away])
print(f"\nElo statistics (all matches):")
print(f"  Mean: {all_elos.mean():.1f} (should be ~1500)")
print(f"  Min: {all_elos.min():.1f}, Max: {all_elos.max():.1f}")
print(f"  Std: {all_elos.std():.1f}")

print("\n" + "=" * 70)
print("STEP 4: V2 FEATURE ENGINEERING CHECK")
print("=" * 70)

# Replicate feature_engineering_v2 on processed data
df = proc.copy()

# Now check which V2 features are already present vs need to be computed
v2_features_expected = [
    "elo_diff_norm", "elo_ratio", "elo_gap",
    "goal_diff_advantage",
    "strength_advantage", "match_quality",
    "draw_rate_home", "draw_rate_away", "both_draw_prone",
    "strength_parity", "defensive_similarity", "low_scoring_tendency",
    "h2h_dominance",
    "sim_wr_advantage", "sim_dr_advantage", "sim_gs_advantage", "sim_gc_advantage",
    "sim_wr_quality", "sim_dr_quality",
    "year_norm", "is_wc", "is_wcq", "is_continental", "is_friendly",
    "is_neutral",
]

present = [f for f in v2_features_expected if f in proc.columns]
missing = [f for f in v2_features_expected if f not in proc.columns]
print(f"V2 features present in processed_matches.csv: {len(present)}/{len(v2_features_expected)}")
if missing:
    print(f"⚠ MISSING: {missing}")
else:
    print("✓ All V2 features present")

# Check sim feature ranges
sim_cols = ["sim_wr_advantage", "sim_dr_advantage", "sim_gs_advantage",
            "sim_gc_advantage", "sim_wr_quality", "sim_dr_quality"]
print("\nSim feature ranges (scored matches only):")
scored_proc = proc[proc["home_score"].notna()]
for col in sim_cols:
    if col in proc.columns:
        vals = scored_proc[col].dropna()
        print(f"  {col:25s}: mean={vals.mean():+.4f}  std={vals.std():.4f}  "
              f"min={vals.min():+.4f}  max={vals.max():+.4f}  NaN={scored_proc[col].isna().sum()}")

# Spot check: sim_wr_advantage should be home_sim_win_rate - away_sim_win_rate
print("\nSpot check sim_wr_advantage = home_sim_win_rate - away_sim_win_rate:")
sample = scored_proc.tail(10)
for _, r in sample.iterrows():
    expected = r["home_sim_win_rate"] - r["away_sim_win_rate"]
    actual = r["sim_wr_advantage"]
    match = "✓" if abs(expected - actual) < 0.001 else "✗ MISMATCH"
    print(f"  {match} {r['date'].date()} {r['home_team']:20s} vs {r['away_team']:20s} "
          f"hw={r['home_sim_win_rate']:.3f} aw={r['away_sim_win_rate']:.3f} "
          f"expected={expected:+.4f} actual={actual:+.4f}")

# Check feature ranges for potential outliers
print("\nKey feature statistics (scored matches, last 5000):")
key_feats = ["elo_advantage_home", "elo_quality", "elo_diff_norm", "elo_ratio", "elo_gap",
             "strength_advantage", "match_quality", "year_norm", "is_neutral"]
recent = scored_proc.tail(5000)
for col in key_feats:
    if col in proc.columns:
        vals = recent[col].dropna()
        print(f"  {col:30s}: mean={vals.mean():+.4f}  std={vals.std():.4f}  "
              f"[{vals.min():+.4f}, {vals.max():+.4f}]")

print("\n" + "=" * 70)
print("STEP 5: DATA SPLIT VERIFICATION")
print("=" * 70)

# Chronological split boundaries
train_end = pd.to_datetime("2021-12-31")
val_end = pd.to_datetime("2023-12-31")

train = proc[(proc["date"] >= pd.to_datetime("2000-01-01")) & (proc["date"] < train_end)]
val = proc[(proc["date"] >= train_end) & (proc["date"] < val_end)]
test = proc[proc["date"] >= val_end]

print(f"Train: {len(train)} matches ({train['date'].min().date()} to {train['date'].max().date()})")
print(f"Val:   {len(val)} matches ({val['date'].min().date()} to {val['date'].max().date()})")
print(f"Test:  {len(test)} matches ({test['date'].min().date()} to {test['date'].max().date()})")

# Check no overlap
train_teams_dates = set(zip(train["home_team"], train["date"]))
val_overlap = sum(1 for _, r in val.iterrows() if (r["home_team"], r["date"]) in train_teams_dates)
test_overlap = sum(1 for _, r in test.iterrows() if (r["home_team"], r["date"]) in train_teams_dates)
print(f"Date overlap: train-val={val_overlap}, train-test={test_overlap} (should be 0)")

# Check WCQ count in test
test_with_scores = test[test["home_score"].notna()]
wcq_in_test = test_with_scores[test_with_scores["tournament"].str.contains("qualification", na=False)]
print(f"\nWCQ in test (scored): {len(wcq_in_test)} matches")
if len(wcq_in_test) > 0:
    wcq_preds = wcq_in_test["result"].value_counts().sort_index()
    print(f"  Result dist: {dict(wcq_preds)}")

print("\n" + "=" * 70)
print("STEP 6: MODEL TRAINED FEATURE LIST vs ACTUAL")
print("=" * 70)

from train_improved import prepare_enhanced_data

# Load data the same way training does
train_df, val_df, test_df, team_encoder, scaler, all_features = prepare_enhanced_data()
print(f"Features going into model ({len(all_features)}):")
for i, f in enumerate(all_features):
    print(f"  [{i:2d}] {f}")

# Verify no NaNs in training data
X_train = train_df[all_features].fillna(0).values
nan_count = np.isnan(X_train).sum()
inf_count = np.isinf(X_train).sum()
print(f"\nNaN in training features: {nan_count}, Inf: {inf_count} (both should be 0)")
if nan_count > 0:
    nan_cols = [all_features[i] for i in range(X_train.shape[1]) if np.isnan(X_train[:, i]).sum() > 0]
    print(f"⚠ Columns with NaN: {nan_cols}")

# Check for degeneracy (zero variance features)
for i, f in enumerate(all_features):
    std = np.std(X_train[:, i])
    if std < 1e-8:
        print(f"⚠ ZERO VARIANCE: {f} (std={std})")

print("\n✓ Feature verification complete.")
