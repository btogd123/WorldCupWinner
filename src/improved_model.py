"""
Improved Match Predictor with attention and better features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TeamAttentionBlock(nn.Module):
    """Self-attention block for team feature interactions."""

    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class ImprovedMatchPredictor(nn.Module):
    """
    Improved neural network for match outcome prediction.

    Key improvements:
    - Team embeddings with attention interaction
    - Separate encoders for home/away context
    - Auxiliary goal prediction heads
    - Deeper residual architecture
    """

    def __init__(
        self,
        num_teams: int,
        team_embedding_dim: int = 64,
        num_match_features: int = 16,
        dropout_rate: float = 0.25,
    ):
        super().__init__()

        # Team embeddings - richer representation
        self.team_embedding = nn.Embedding(num_teams, team_embedding_dim, padding_idx=0)
        self.team_strength_bias = nn.Embedding(num_teams, 1, padding_idx=0)

        # Home/away indicator embeddings
        self.home_indicator = nn.Parameter(torch.randn(1, team_embedding_dim) * 0.1)
        self.away_indicator = nn.Parameter(torch.randn(1, team_embedding_dim) * 0.1)

        # Team interaction attention
        self.team_attention = TeamAttentionBlock(team_embedding_dim, num_heads=4)

        # Match feature processing
        self.match_encoder = nn.Sequential(
            nn.Linear(num_match_features, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        # Combined processing
        combined_dim = team_embedding_dim * 2 + 128

        self.shared = nn.Sequential(
            DenseBlock(combined_dim, 256, dropout_rate),
            DenseBlock(256, 256, dropout_rate),
            DenseBlock(256, 128, dropout_rate),
        )

        # Main classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

        # Auxiliary goal prediction heads (multi-task learning)
        self.goal_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2),  # Predict (home_goals, away_goals)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.05)

    def forward(self, home_team_ids, away_team_ids, match_features, return_goals=False):
        """
        Args:
            home_team_ids: (batch,) LongTensor
            away_team_ids: (batch,) LongTensor
            match_features: (batch, num_match_features) FloatTensor
            return_goals: if True, also return goal predictions
        """
        batch_size = home_team_ids.size(0)

        # Get team embeddings with strength bias
        home_emb = self.team_embedding(home_team_ids) + self.home_indicator
        away_emb = self.team_embedding(away_team_ids) + self.away_indicator
        home_strength = self.team_strength_bias(home_team_ids)
        away_strength = self.team_strength_bias(away_team_ids)

        # Team interaction via attention (treat home and away as a 2-element sequence)
        team_seq = torch.stack([home_emb, away_emb], dim=1)  # (batch, 2, emb_dim)
        team_seq = self.team_attention(team_seq)

        # Extract attended embeddings
        home_att = team_seq[:, 0, :]
        away_att = team_seq[:, 1, :]

        # Encode match features
        match_encoded = self.match_encoder(match_features)

        # Combine
        combined = torch.cat([home_att, away_att, match_encoded], dim=1)

        # Shared representation
        shared = self.shared(combined)

        # Classification
        logits = self.classifier(shared)

        if return_goals:
            goals = self.goal_head(shared)
            goals = F.softplus(goals)  # Ensure non-negative
            return logits, goals

        return logits

    def predict_proba(self, home_team_ids, away_team_ids, match_features):
        self.eval()
        with torch.no_grad():
            logits = self.forward(home_team_ids, away_team_ids, match_features)
            probs = F.softmax(logits, dim=-1)
        return probs


class DenseBlock(nn.Module):
    """Dense-style residual block."""

    def __init__(self, in_dim, out_dim, dropout_rate=0.3):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(in_dim)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.norm2 = nn.BatchNorm1d(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.norm1(x)
        out = F.gelu(out)
        out = self.linear1(out)
        out = self.norm2(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.linear2(out)

        return out + residual


class ImprovedLoss(nn.Module):
    """Combined loss: classification + auxiliary goal prediction."""

    def __init__(self, class_weights=None, goal_weight=0.1):
        super().__init__()
        self.class_weights = class_weights
        self.goal_weight = goal_weight

    def forward(self, logits, goals, targets, home_goals, away_goals):
        # Classification loss with class weights
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights)

        # Goal prediction loss (MSE)
        goal_targets = torch.stack([home_goals, away_goals], dim=1).float()
        goal_loss = F.mse_loss(goals, goal_targets)

        return ce_loss + self.goal_weight * goal_loss, ce_loss, goal_loss


def create_improved_model(num_teams, num_match_features=16, device=None):
    """Create the improved model."""
    model = ImprovedMatchPredictor(
        num_teams=num_teams + 1,
        team_embedding_dim=64,
        num_match_features=num_match_features,
        dropout_rate=0.25,
    )

    if device:
        model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Improved Model: {total_params:,} params ({trainable_params:,} trainable)")

    return model
