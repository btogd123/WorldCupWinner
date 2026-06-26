"""
Evaluation metrics for 3-way match prediction.

Pure stateless functions — no PyTorch dependency, only numpy.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report as sk_classification_report,
)


def brier_score(y_true, probs):
    """Brier score (mean squared error of predicted probabilities)."""
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(len(y_true)), y_true] = 1
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def log_loss_score(y_true, probs, eps=1e-15):
    """Cross-entropy log loss."""
    probs = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true])))


def ece(y_true, probs, n_bins=10):
    """Expected Calibration Error."""
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accs = (preds == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_sum = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = accs[mask].mean()
            bin_conf = confidences[mask].mean()
            ece_sum += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)
    return float(ece_sum)


def compute_basic_metrics(y_true, preds):
    """Hard metrics from class predictions only — no probs needed.

    Suitable for train-epoch progress logging where calibration metrics
    (Brier/LogLoss/ECE) are meaningless and wasteful to compute.
    """
    return {
        "acc": float(accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average="macro")),
    }


def compute_prob_metrics(y_true, probs):
    """Soft metrics that require full probability distributions.

    Suitable for val/test/backtest where calibration quality matters.
    """
    mean_probs = probs.mean(axis=0)
    return {
        "brier": brier_score(y_true, probs),
        "logloss": log_loss_score(y_true, probs),
        "ece": ece(y_true, probs),
        "mean_away_prob": float(mean_probs[0]),
        "mean_draw_prob": float(mean_probs[1]),
        "mean_home_prob": float(mean_probs[2]),
        "mean_confidence": float(np.max(probs, axis=1).mean()),
    }


def compute_metrics(y_true, probs):
    """Full metrics: basic + probability + sample count."""
    preds = np.argmax(probs, axis=1)
    metrics = compute_basic_metrics(y_true, preds)
    metrics.update(compute_prob_metrics(y_true, probs))
    metrics["num_samples"] = len(y_true)
    return metrics


def classification_report_dict(y_true, y_pred, target_names=None):
    """Return classification report as a dict."""
    if target_names is None:
        target_names = ["Away Win", "Draw", "Home Win"]
    return sk_classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )


def print_backtest_report(y_true, probs, name=None, predictions_df=None):
    """Print standardized backtest metrics report.

    Args:
        y_true: (N,) integer labels (0=away, 1=draw, 2=home)
        probs: (N, 3) predicted probabilities
        name: optional variant name displayed in header
        predictions_df: optional DataFrame with per-row predictions for
                        actual distribution (takes precedence over y_true)
    """
    preds = np.argmax(probs, axis=1)
    m = compute_metrics(y_true, probs)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, preds, labels=[0, 1, 2], zero_division=0
    )

    header = f"Backtest Results — {name}" if name else "Backtest Results"
    print(f"\n{'='*60}")
    print(header)
    print(f"{'='*60}")
    print(f"Samples:       {m['num_samples']}")
    print(f"Accuracy:      {m['acc']:.4f}")
    print(f"F1 (macro):    {m['f1_macro']:.4f}")
    print(f"LogLoss:       {m['logloss']:.4f}")
    print(f"Brier:         {m['brier']:.4f}")
    print(f"ECE:           {m['ece']:.4f}")

    print(f"\nPer-class:")
    print(f"{'':<14} {'Precision':>9} {'Recall':>9} {'F1':>9} {'Support':>9}")
    for i, label in enumerate(["Away Win", "Draw", "Home Win"]):
        print(f"  {label:<12} {p[i]:>9.4f} {r[i]:>9.4f} {f1[i]:>9.4f} {support[i]:>9}")

    pred_counts = np.bincount(preds, minlength=3)
    pred_total = len(preds)
    print(f"\nPredicted:  Away {pred_counts[0]/pred_total*100:.1f}% | "
          f"Draw {pred_counts[1]/pred_total*100:.1f}% | "
          f"Home {pred_counts[2]/pred_total*100:.1f}%")

    if predictions_df is not None:
        actuals = predictions_df["actual_score"].dropna()
        n_actual = len(actuals)
        if n_actual > 0:
            away_n = draw_n = home_n = 0
            for score in actuals:
                hs, aws = score.split("-")
                hs, aws = int(hs), int(aws)
                if hs > aws:
                    home_n += 1
                elif hs == aws:
                    draw_n += 1
                else:
                    away_n += 1
            print(f"Actual:     Away {away_n/n_actual*100:.1f}% | "
                  f"Draw {draw_n/n_actual*100:.1f}% | "
                  f"Home {home_n/n_actual*100:.1f}%")

            correct = predictions_df["correct"].sum()
            n_valid = predictions_df["correct"].dropna().count()
            if n_valid > 0:
                print(f"\nCorrect:      {int(correct)}/{int(n_valid)} = {correct/n_valid*100:.1f}%")
    else:
        true_counts = np.bincount(y_true, minlength=3)
        true_total = len(y_true)
        print(f"Actual:     Away {true_counts[0]/true_total*100:.1f}% | "
              f"Draw {true_counts[1]/true_total*100:.1f}% | "
              f"Home {true_counts[2]/true_total*100:.1f}%")
