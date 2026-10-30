"""Editable two-tower model: ID + causally constructed recent-history features."""

import torch
from torch import nn
from torch.nn import functional as F


class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, dim=32, use_history=True, dropout=0.1):
        super().__init__()
        self.use_history = use_history
        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_embedding = nn.Embedding(n_items, dim, padding_idx=0)
        self.user_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim)
        )

    def encode_user(self, users, history):
        mask = (history != 0).unsqueeze(-1)
        pooled = (self.item_embedding(history) * mask).sum(1) / mask.sum(1).clamp_min(1)
        if not self.use_history:
            pooled = torch.zeros_like(pooled)
        return F.normalize(
            self.user_mlp(torch.cat([self.user_embedding(users), pooled], dim=-1)),
            dim=-1,
        )

    def encode_item(self, items):
        return F.normalize(self.item_mlp(self.item_embedding(items)), dim=-1)

    def loss(self, users, history, targets, temperature=0.1, logq=None):
        # In-batch sampled softmax. Duplicate positive IDs are accidental hits, not negatives.
        logits = (
            self.encode_user(users, history) @ self.encode_item(targets).T / temperature
        )
        if logq is not None:
            logits = logits - logq[targets][None, :]
        duplicate = targets[:, None].eq(targets[None, :])
        duplicate.fill_diagonal_(False)
        logits = logits.masked_fill(duplicate, -1e9)
        return F.cross_entropy(logits, torch.arange(len(users), device=users.device))
