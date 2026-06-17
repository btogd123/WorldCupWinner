"""
Test removing sim_dr_advantage (always 0) and sim_gc_advantage (=-sim_gs_advantage).
Accept only if BOTH Brier and ECE improve.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablate_all_features import (
    train_and_evaluate,
    load_and_prepare,
    feature_engineering_v2,
    split_data,
    ALL_FEATURE_COLS,
)

ALL_FEATURES = list(ALL_FEATURE_COLS)
REMOVE = ["sim_dr_advantage", "sim_gc_advantage"]
CLEANED = [f for f in ALL_FEATURES if f not in REMOVE]

df, team_encoder = load_and_prepare()
df = feature_engineering_v2(df)
train_df, val_df, test_df = split_data(df)

results = {}
for label, feats in [("All 33 features", ALL_FEATURES), ("Remove sim_dr+sim_gc (31)", CLEANED)]:
    print(f"\n{'=' * 60}")
    print(f"Training: {label} ({len(feats)} features)")
    print(f"{'=' * 60}")
    r = train_and_evaluate(feats, train_df, val_df, test_df, team_encoder)
    results[label] = r
    print(f"  Acc={r['accuracy']:.4f}  F1={r['macro_f1']:.4f}  Brier={r['brier']:.4f}  ECE={r['ece']:.4f}")
    print(f"  DrawR={r['draw_recall']:.4f}  HomeR={r['home_recall']:.4f}  AwayR={r['away_recall']:.4f}")

print("\n" + "=" * 80)
print("COMPARISON: Remove sim_dr_advantage + sim_gc_advantage")
print("=" * 80)
print(f"{'Metric':15s} {'33-feat':>10s} {'31-feat':>10s} {'Delta':>10s}  Verdict")
print("-" * 70)

base = results["All 33 features"]
clean = results["Remove sim_dr+sim_gc (31)"]

brier_improved = False
ece_improved = False

for key, label, lower_better in [
    ("accuracy", "Accuracy", False),
    ("macro_f1", "Macro F1", False),
    ("brier", "Brier", True),
    ("ece", "ECE", True),
    ("draw_recall", "Draw Recall", False),
    ("home_recall", "Home Recall", False),
    ("away_recall", "Away Recall", False),
]:
    delta = clean[key] - base[key]
    if lower_better:
        improved = delta < 0
        verdict = "BETTER" if improved else "worse"
    else:
        improved = delta > 0
        verdict = "BETTER" if improved else "worse"

    if key == "brier":
        brier_improved = improved
    if key == "ece":
        ece_improved = improved

    print(f"{label:15s} {base[key]:>10.4f} {clean[key]:>10.4f} {delta:>+10.4f}  {verdict}")

print("\n" + "=" * 80)
if brier_improved and ece_improved:
    print("ACCEPT: Both Brier and ECE improved. Remove sim_dr_advantage + sim_gc_advantage.")
elif brier_improved:
    print("REJECT: Brier improved but ECE did not. Keep all 33 features.")
elif ece_improved:
    print("REJECT: ECE improved but Brier did not. Keep all 33 features.")
else:
    print("REJECT: Neither Brier nor ECE improved. Keep all 33 features.")
