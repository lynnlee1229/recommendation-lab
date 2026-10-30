"""Small neural building blocks, with explicit teaching simplifications."""

import torch
from torch import nn
from torch.nn import functional as F


def fm_interaction(fields):
    return 0.5 * ((fields.sum(1) ** 2) - (fields**2).sum(1)).sum(-1)


def fm_explicit(fields):
    return sum(
        (fields[:, i] * fields[:, j]).sum(-1)
        for i in range(fields.shape[1])
        for j in range(i + 1, fields.shape[1])
    )


class CrossV2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.affine = nn.Linear(dim, dim)

    def forward(self, x0, x):
        return x0 * self.affine(x) + x


class LHUC(nn.Module):
    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(context_dim, hidden_dim)

    def forward(self, hidden, context):
        scale = 2 * torch.sigmoid(self.gate(context))
        return hidden * scale, scale


class SEBilinear(nn.Module):
    """SE-style sigmoid field gates + shared-W bilinear interactions, not full FiBiNET."""

    def __init__(self, n_fields, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_fields, max(2, n_fields // 2)),
            nn.ReLU(),
            nn.Linear(max(2, n_fields // 2), n_fields),
            nn.Sigmoid(),
        )
        self.bilinear = nn.Linear(dim, dim, bias=False)

    def forward(self, fields):
        weights = self.gate(fields.mean(-1))
        rescaled = fields * weights.unsqueeze(-1)
        pairs = [
            self.bilinear(rescaled[:, i]) * rescaled[:, j]
            for i in range(fields.shape[1])
            for j in range(i + 1, fields.shape[1])
        ]
        return torch.cat(pairs, -1), weights


class MMoE(nn.Module):
    def __init__(self, input_dim, experts=3, tasks=2, hidden=12):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU())
                for _ in range(experts)
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(input_dim, experts) for _ in range(tasks)]
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(tasks)])

    def forward(self, x):
        experts = torch.stack([expert(x) for expert in self.experts], 1)
        weights = [gate(x).softmax(-1) for gate in self.gates]
        logits = torch.cat(
            [
                head((experts * w.unsqueeze(-1)).sum(1))
                for w, head in zip(weights, self.heads)
            ],
            -1,
        )
        return logits, torch.stack(weights, 1)


class ThreeTower(nn.Module):
    """Cached user/item representations plus per-candidate interaction; not full COLD."""

    def __init__(self, dim=8, hidden=12):
        super().__init__()
        self.user = nn.Linear(dim, hidden)
        self.item = nn.Linear(dim, hidden)
        self.cross = nn.Linear(dim, hidden)
        self.head = nn.Linear(hidden * 3, 1)

    def cached_score(self, user_vector, item_vectors, cross_features):
        u = self.user(user_vector).expand(len(item_vectors), -1)
        return self.head(
            torch.cat([u, self.item(item_vectors), self.cross(cross_features)], -1)
        ).squeeze(-1)


class TargetAttention(nn.Module):
    """DIN-style candidate-dependent activation, softmax-normalized teaching variant."""

    def __init__(self, dim, hidden=16):
        super().__init__()
        self.activation = nn.Sequential(
            nn.Linear(dim * 4, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, history, candidate, mask):
        query = candidate.unsqueeze(1).expand_as(history)
        logits = self.activation(
            torch.cat([history, query, history - query, history * query], -1)
        ).squeeze(-1)
        weights = logits.masked_fill(~mask, -1e9).softmax(-1) * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        return (history * weights.unsqueeze(-1)).sum(1), weights


def fit_multitask(x, labels, steps=80, seed=42):
    torch.manual_seed(seed)
    model = MMoE(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    curve = []
    for _ in range(steps):
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        curve.append(float(loss.detach()))
    return model, curve


def calibrate_negative_sampling(sampled_probability, negative_keep_rate):
    if not 0 < negative_keep_rate <= 1:
        raise ValueError("negative keep rate must be in (0,1]")
    p = sampled_probability
    a = negative_keep_rate
    return a * p / (1 - p + a * p)
