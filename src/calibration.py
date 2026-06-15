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
