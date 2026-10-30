from itertools import combinations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def binary_matrix(pairs, n_users, n_items):
    matrix = np.zeros((n_users, n_items), dtype=np.float32)
    for user, item in pairs:
        matrix[user, item] = 1
    return matrix


def usercf(matrix, user, neighbors=10):
    """IUF weighted user cosine, followed by a user->user->item aggregation."""
    item_frequency = matrix.sum(0)
    weighted = matrix / np.sqrt(np.log1p(np.maximum(item_frequency, 1)))
    normalized = weighted / np.maximum(
        np.linalg.norm(weighted, axis=1, keepdims=True), 1e-8
    )
    similarities = normalized @ normalized[user]
    similarities[user] = -np.inf
    ids = np.argsort(-similarities, kind="stable")[: min(neighbors, len(matrix) - 1)]
    return similarities[ids] @ matrix[ids], ids, similarities[ids]


def swing(matrix, alpha=1.0):
    """Unweighted Swing teaching variant. Each unordered user-pair is counted once."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    histories = [set(np.flatnonzero(row)) for row in matrix]
    result = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=float)
    for u, v in combinations(range(len(histories)), 2):
        common = histories[u] & histories[v]
        contribution = 1 / (alpha + len(common))
        for i, j in combinations(sorted(common), 2):
            result[i, j] += contribution
            result[j, i] += contribution
    return result


class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, dim=12):
        super().__init__()
        self.users = nn.Embedding(n_users, dim)
        self.items = nn.Embedding(n_items, dim)
        nn.init.normal_(self.users.weight, std=0.1)
        nn.init.normal_(self.items.weight, std=0.1)

    def forward(self, users, items):
        return (self.users(users) * self.items(items)).sum(-1)

    def bpr_loss(self, users, positive, negative):
        return F.softplus(-(self(users, positive) - self(users, negative))).mean()


def objective_losses(positive, negative):
    """Same fixed score pairs, three distinct objectives; loss scales are not comparable."""
    pointwise = F.binary_cross_entropy_with_logits(
        torch.cat([positive, negative]),
        torch.cat([torch.ones_like(positive), torch.zeros_like(negative)]),
    )
    pairwise = F.softplus(negative - positive).mean()
    listwise = F.cross_entropy(
        torch.stack([positive, negative], 1),
        torch.zeros(len(positive), dtype=torch.long),
    )
    return {"pointwise": pointwise, "pairwise": pairwise, "listwise": listwise}


def kmeans(vectors, clusters=12, iterations=8, seed=42):
    rng = np.random.default_rng(seed)
    centers = vectors[rng.choice(len(vectors), clusters, replace=False)].copy()
    for _ in range(iterations):
        labels = ((vectors[:, None, :] - centers[None, :, :]) ** 2).sum(-1).argmin(1)
        for cluster in range(clusters):
            if np.any(labels == cluster):
                centers[cluster] = vectors[labels == cluster].mean(0)
    labels = ((vectors[:, None, :] - centers[None, :, :]) ** 2).sum(-1).argmin(1)
    return centers, labels


def partition_retrieval(vectors, query, centers, labels, probes=1, k=20):
    """Coarse centroid routing; inspect approximation relative to exact dot-product."""
    selected = np.argsort(((centers - query) ** 2).sum(1))[:probes]
    candidates = np.flatnonzero(np.isin(labels, selected))
    order = np.argsort(-(vectors[candidates] @ query), kind="stable")[:k]
    return candidates[order], len(candidates)


def beam_search(transition, depth=2, width=1):
    """Conditional path probabilities. No neural path/index learning is claimed."""
    beams = [((), 0.0)]
    for _ in range(depth):
        expanded = []
        for path, log_prob in beams:
            for node, probability in transition[path].items():
                expanded.append((path + (node,), log_prob + np.log(probability)))
        beams = sorted(expanded, key=lambda row: (-row[1], row[0]))[:width]
    return beams


class SideInfoEncoder(nn.Module):
    """ID/category field masking for a small SSL mechanism exercise."""

    def __init__(self, n_items, n_categories, dim=12):
        super().__init__()
        self.item = nn.Embedding(n_items, dim, padding_idx=0)
        self.category = nn.Embedding(n_categories, dim, padding_idx=0)
        self.project = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim)
        )

    def forward(self, ids, categories):
        return F.normalize(
            self.project(torch.cat([self.item(ids), self.category(categories)], -1)),
            dim=-1,
        )


def mask_fields(ids, categories, drop_probability=0.5):
    id_mask = torch.rand(ids.shape, device=ids.device) < drop_probability
    cat_mask = torch.rand(categories.shape, device=categories.device) < drop_probability
    return ids.masked_fill(id_mask, 0), categories.masked_fill(cat_mask, 0)
