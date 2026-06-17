"""
Backtest comparison: 33 features vs 31 features (no sim_dr_advantage, sim_gc_advantage).
Tests both configs across all 7 post-2022 tournaments.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_wc2022 import (
    TOURNAMENTS, load_full_data, backtest_single_tournament,
)
import numpy as np
import json
from datetime import datetime
from config import RESULTS_DIR


def run_comparison():
    print("=" * 70)
    print("BACKTEST COMPARISON: 33 features vs 31 features")
    print("=" * 70)

    full_df = load_full_data()

    all_results = {}

    for label, feat_config in [
        ("33 features (old)", {"use_sim_features": True, "extra_features": ["sim_dr_advantage", "sim_gc_advantage"]}),
        ("31 features (no sim_dr, sim_gc)", {"use_sim_features": True, "extra_features": None}),
    ]:
        print(f"\n{'#' * 70}")
        print(f"# CONFIG: {label}")
        print(f"{'#' * 70}")

        results = []
        for name, keyword, ds, de in TOURNAMENTS:
            result = backtest_single_tournament(
                name, keyword, ds, de, full_df,
                use_draw_features=True,
                use_neutral_gating=True,
                use_group_round=False,
                use_sim_features=feat_config["use_sim_features"],
                extra_features=feat_config["extra_features"],
            )
            if result is not None:
                results.append(result)

        all_results[label] = results

    # ---- Comparison Table ----
    print("\n\n" + "=" * 90)
    print("BACKTEST COMPARISON RESULTS")
    print("=" * 90)

    for label, results in all_results.items():
        total = sum(r["test_matches"] for r in results)
        n_correct = sum(
            sum(1 for m in r["match_details"] if m["correct"] == "Y")
            for r in results
        )
        overall = n_correct / total * 100 if total > 0 else 0
        avg_acc = np.mean([r["accuracy"] for r in results]) * 100
        avg_f1 = np.mean([r["macro_f1"] for r in results])
        avg_draw = np.mean([r["draw_recall"] for r in results]) * 100
        avg_home = np.mean([r["home_recall"] for r in results]) * 100
        avg_away = np.mean([r["away_recall"] for r in results]) * 100

        # Brier/ECE
        all_probs, all_labels = [], []
        for r in results:
            for m in r["match_details"]:
                all_probs.append([float(m["away_prob"]), float(m["draw_prob"]), float(m["home_prob"])])
                if m["result"] == "Away Win":
                    all_labels.append(0)
                elif m["result"] == "Draw":
                    all_labels.append(1)
                else:
                    all_labels.append(2)

        probs = np.array(all_probs, dtype=np.float64)
        labels = np.array(all_labels)
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

        print(f"\n{label}:")
        print(f"  Overall:       {n_correct}/{total} correct ({overall:.1f}%)")
        print(f"  Avg Accuracy:  {avg_acc:.1f}%")
        print(f"  Avg Macro F1:  {avg_f1:.4f}")
        print(f"  Avg Draw Rec:  {avg_draw:.1f}%")
        print(f"  Avg Home Rec:  {avg_home:.1f}%")
        print(f"  Avg Away Rec:  {avg_away:.1f}%")
        print(f"  Brier:         {brier:.4f}")
        print(f"  ECE:           {ece:.4f}")

    # ---- Per-Tournament Comparison ----
    print(f"\n{'Tournament':25s} {'Matches':>7s} | {'33-Acc':>7s} {'33-F1':>7s} | {'31-Acc':>7s} {'31-F1':>7s} | {'dAcc':>7s} {'dF1':>7s}")
    print("-" * 110)

    old_results = all_results["33 features (old)"]
    new_results = all_results["31 features (no sim_dr, sim_gc)"]

    for o, n in zip(old_results, new_results):
        name = o["name"]
        matches = o["test_matches"]
        d_acc = (n["accuracy"] - o["accuracy"]) * 100
        d_f1 = n["macro_f1"] - o["macro_f1"]
        print(f"{name:25s} {matches:>7d} | {o['accuracy']*100:>6.1f}% {o['macro_f1']:>7.4f} | {n['accuracy']*100:>6.1f}% {n['macro_f1']:>7.4f} | {d_acc:>+6.1f}% {d_f1:>+7.4f}")

    # ---- Brier/ECE per tournament ----
    print(f"\n{'Tournament':25s} | {'33-Brier':>9s} {'33-ECE':>8s} | {'31-Brier':>9s} {'31-ECE':>8s} | {'dBrier':>9s} {'dECE':>8s}")
    print("-" * 90)

    for o, n in zip(old_results, new_results):
        name = o["name"]
        ob, oe = _compute_calibration(o["match_details"])
        nb, ne = _compute_calibration(n["match_details"])
        print(f"{name:25s} | {ob:>9.4f} {oe:>8.4f} | {nb:>9.4f} {ne:>8.4f} | {nb-ob:>+9.4f} {ne-oe:>+8.4f}")

    # ---- Final verdict ----
    print("\n" + "=" * 90)
    old_total = sum(r["test_matches"] for r in old_results)
    new_total = sum(r["test_matches"] for r in new_results)

    # Aggregate Brier/ECE
    def agg_calib(results):
        probs, labels = [], []
        for r in results:
            for m in r["match_details"]:
                probs.append([float(m["away_prob"]), float(m["draw_prob"]), float(m["home_prob"])])
                if m["result"] == "Away Win":
                    labels.append(0)
                elif m["result"] == "Draw":
                    labels.append(1)
                else:
                    labels.append(2)
        probs = np.array(probs, dtype=np.float64)
        labels = np.array(labels)
        n = len(labels)
        y = np.zeros_like(probs)
        y[np.arange(n), labels] = 1
        b = np.mean(np.sum((probs - y) ** 2, axis=1))
        conf = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        corr = (preds == labels).astype(float)
        bins = np.linspace(0, 1, 11)
        e = 0.0
        for i in range(10):
            mask = (conf >= bins[i]) & (conf < bins[i + 1])
            if mask.sum() > 0:
                e += (mask.sum() / n) * abs(corr[mask].mean() - conf[mask].mean())
        return b, e

    ob, oe = agg_calib(old_results)
    nb, ne = agg_calib(new_results)

    brier_ok = nb < ob
    ece_ok = ne < oe

    print(f"Aggregate Brier: {ob:.4f} → {nb:.4f} (Δ={nb-ob:+.4f}) {'✓ BETTER' if brier_ok else '✗ WORSE'}")
    print(f"Aggregate ECE:   {oe:.4f} → {ne:.4f} (Δ={ne-oe:+.4f}) {'✓ BETTER' if ece_ok else '✗ WORSE'}")

    if brier_ok and ece_ok:
        print("\nACCEPT: Both Brier and ECE improved in backtest.")
    elif brier_ok:
        print("\nREJECT: Brier improved but ECE did not.")
    elif ece_ok:
        print("\nREJECT: ECE improved but Brier did not.")
    else:
        print("\nREJECT: Neither improved in backtest.")

    # Save
    output = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "old_33_features": _summarize(old_results),
        "new_31_features": _summarize(new_results),
        "aggregate_brier": {"old": ob, "new": nb},
        "aggregate_ece": {"old": oe, "new": ne},
        "verdict": "ACCEPT" if (brier_ok and ece_ok) else "REJECT",
        "per_tournament": {
            o["name"]: {
                "33_feat": _extract_metrics(o),
                "31_feat": _extract_metrics(n),
            }
            for o, n in zip(old_results, new_results)
        },
    }

    output_path = os.path.join(RESULTS_DIR, "backtest_31vs33.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


def _compute_calibration(match_details):
    probs = np.array([[float(m["away_prob"]), float(m["draw_prob"]), float(m["home_prob"])]
                       for m in match_details], dtype=np.float64)
    labels = np.array([0 if m["result"] == "Away Win" else (1 if m["result"] == "Draw" else 2)
                        for m in match_details])
    n = len(labels)
    y = np.zeros_like(probs)
    y[np.arange(n), labels] = 1
    brier = np.mean(np.sum((probs - y) ** 2, axis=1))
    conf = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    corr = (preds == labels).astype(float)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        mask = (conf >= bins[i]) & (conf < bins[i + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / n) * abs(corr[mask].mean() - conf[mask].mean())
    return brier, ece


def _extract_metrics(r):
    return {
        "accuracy": r["accuracy"],
        "macro_f1": r["macro_f1"],
        "draw_recall": r["draw_recall"],
        "home_recall": r["home_recall"],
        "away_recall": r["away_recall"],
        "test_matches": r["test_matches"],
    }


def _summarize(results):
    total = sum(r["test_matches"] for r in results)
    n_correct = sum(sum(1 for m in r["match_details"] if m["correct"] == "Y") for r in results)
    return {
        "total_matches": total,
        "total_correct": n_correct,
        "overall_accuracy": n_correct / total if total > 0 else 0,
        "avg_accuracy": float(np.mean([r["accuracy"] for r in results])),
        "avg_macro_f1": float(np.mean([r["macro_f1"] for r in results])),
        "avg_draw_recall": float(np.mean([r["draw_recall"] for r in results])),
    }


if __name__ == "__main__":
    run_comparison()
