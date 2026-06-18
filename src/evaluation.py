"""
Evaluation metrics for 3-way match prediction.

Pure stateless functions — no PyTorch dependency, only numpy.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report as sk_classification_report


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


def compute_metrics(y_true, probs):
    """Compute all standard metrics from labels and predicted probabilities.

    Args:
        y_true: (N,) integer labels (0=away, 1=draw, 2=home)
        probs: (N, 3) predicted probabilities

    Returns:
        dict with keys: acc, f1_macro, brier, logloss, ece,
                        mean_away_prob, mean_draw_prob, mean_home_prob,
                        mean_confidence, num_samples
    """
    preds = np.argmax(probs, axis=1)
    mean_probs = probs.mean(axis=0)

    return {
        "acc": float(accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average="macro")),
        "brier": brier_score(y_true, probs),
        "logloss": log_loss_score(y_true, probs),
        "ece": ece(y_true, probs),
        "mean_away_prob": float(mean_probs[0]),
        "mean_draw_prob": float(mean_probs[1]),
        "mean_home_prob": float(mean_probs[2]),
        "mean_confidence": float(np.max(probs, axis=1).mean()),
        "num_samples": len(y_true),
    }


def classification_report_dict(y_true, y_pred, target_names=None):
    """Return classification report as a dict."""
    if target_names is None:
        target_names = ["Away Win", "Draw", "Home Win"]
    return sk_classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
