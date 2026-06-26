"""
Temperature Scaling for probability calibration.

Fixes model underconfidence/over-dispersion by learning a single
temperature parameter T on the validation set. At inference:
    calibrated_probs = softmax(logits / T)

T < 1 sharpens predictions (fixes underconfidence).
T > 1 smooths predictions (fixes overconfidence).

Reference: Guo et al. (2017) "On Calibration of Modern Neural Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemperatureScaler:
    """Learns an optimal temperature to calibrate model probabilities."""

    def __init__(self, initial_temp=1.0):
        self.temperature = nn.Parameter(torch.tensor(initial_temp, dtype=torch.float32))

    def fit(self, logits, labels, lr=0.01, max_iter=200, device=None):
        """
        Learn optimal temperature from logits and true labels.

        Args:
            logits: (N, 3) numpy array or torch tensor of model logits
            labels: (N,) numpy array or torch tensor of true labels (0/1/2)
            lr: learning rate for temperature optimization
            max_iter: maximum optimization iterations
            device: torch device

        Returns:
            self with learned temperature
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(logits, np.ndarray):
            logits = torch.FloatTensor(logits)
        if isinstance(labels, np.ndarray):
            labels = torch.LongTensor(labels)

        logits = logits.to(device)
        labels = labels.to(device)
        self.temperature = nn.Parameter(torch.tensor(1.0, device=device))

        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def _eval():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(_eval)

        self.temperature = self.temperature.detach()
        return self

    def calibrate(self, logits):
        """Apply temperature scaling to logits."""
        if isinstance(logits, np.ndarray):
            logits_t = torch.FloatTensor(logits)
            was_numpy = True
        else:
            logits_t = logits
            was_numpy = False

        with torch.no_grad():
            temp = self.temperature.to(logits_t.device)
            calibrated_logits = logits_t / temp
            probs = F.softmax(calibrated_logits, dim=-1)

        if was_numpy:
            return probs.numpy()
        return probs

    def get_temperature(self):
        """Return the learned temperature value."""
        return self.temperature.item()


def calibrate_and_evaluate(val_logits, val_labels, test_logits, test_labels=None,
                           device=None):
    """Fit temperature scaling on val logits and apply to test logits.

    Args:
        val_logits: (N, C) numpy array of validation logits
        val_labels: (N,) numpy array of validation labels
        test_logits: (M, C) numpy array of test logits
        test_labels: optional (M,) numpy array of test labels for evaluation
        device: torch device

    Returns:
        dict with keys: raw_probs, calibrated_probs, temperature,
                        and (if test_labels given) raw_accuracy, raw_f1,
                        calibrated_accuracy, calibrated_f1
    """
    scaler = TemperatureScaler()
    scaler.fit(val_logits, val_labels, device=device)
    T = scaler.get_temperature()

    cal_probs = scaler.calibrate(test_logits)

    logits_t = torch.FloatTensor(test_logits)
    raw_probs = F.softmax(logits_t, dim=-1).numpy()

    result = {"raw_probs": raw_probs, "calibrated_probs": cal_probs, "temperature": T}

    if test_labels is not None:
        from sklearn.metrics import accuracy_score, f1_score

        raw_preds = np.argmax(raw_probs, axis=1)
        cal_preds = np.argmax(cal_probs, axis=1)
        result["raw_accuracy"] = accuracy_score(test_labels, raw_preds)
        result["raw_f1"] = f1_score(test_labels, raw_preds, average="macro")
        result["calibrated_accuracy"] = accuracy_score(test_labels, cal_preds)
        result["calibrated_f1"] = f1_score(test_labels, cal_preds, average="macro")

    return result
