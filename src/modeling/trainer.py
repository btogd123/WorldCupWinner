"""
Shared training loop used by train.py and backtest.py.
"""

import torch
import torch.nn as nn
import numpy as np
from evaluation import compute_basic_metrics, compute_metrics


class Trainer:
    """Model-agnostic training loop with early stopping and cosine annealing."""

    def __init__(self, model, optimizer, scheduler, criterion, device,
                 patience=25, grad_clip_norm=1.0, max_epochs=300):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.patience = patience
        self.grad_clip_norm = grad_clip_norm
        self.max_epochs = max_epochs

        self.best_val_f1 = 0.0
        self.best_state = None
        self.epochs_trained = 0
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [],
                        "val_acc": [], "val_f1": []}

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for X, h_ids, a_ids, y, hg, ag in loader:
            X = X.to(self.device)
            h_ids = h_ids.to(self.device)
            a_ids = a_ids.to(self.device)
            y = y.to(self.device)
            hg = hg.to(self.device)
            ag = ag.to(self.device)

            self.optimizer.zero_grad()
            logits, goals = self.model(h_ids, a_ids, X, return_goals=True)
            loss, _, _ = self.criterion(logits, goals, y, hg, ag)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                          max_norm=self.grad_clip_norm)
            self.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        n = len(loader)
        m = compute_basic_metrics(all_labels, all_preds)
        return total_loss / n, m["acc"], m["f1_macro"]

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        for X, h_ids, a_ids, y, hg, ag in loader:
            X = X.to(self.device)
            h_ids = h_ids.to(self.device)
            a_ids = a_ids.to(self.device)
            y = y.to(self.device)
            hg = hg.to(self.device)
            ag = ag.to(self.device)

            logits, goals = self.model(h_ids, a_ids, X, return_goals=True)
            loss, _, _ = self.criterion(logits, goals, y, hg, ag)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        n = len(loader)
        m = compute_metrics(all_labels, all_probs)
        return total_loss / n, m["acc"], m["f1_macro"], all_preds, all_labels, all_probs

    def fit(self, train_loader, val_loader, verbose=True):
        patience_counter = 0

        for epoch in range(1, self.max_epochs + 1):
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader)
            val_loss, val_acc, val_f1, _, _, _ = self.validate(val_loader)
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)

            if verbose and (epoch % 20 == 0 or epoch == 1):
                print(f"Epoch {epoch:3d} | "
                      f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.3f} | "
                      f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.3f} | F1: {val_f1:.3f}")

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                patience_counter = 0
                self.best_state = {k: v.cpu().clone()
                                   for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        self.epochs_trained = epoch - patience_counter
        self.model.load_state_dict(self.best_state)
        self.model.eval()
        return self.best_val_f1
