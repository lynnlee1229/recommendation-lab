"""LightGCN propagation/BPR port; smaller explicit CPU training recipe.

Champion comparison: src/retrieval/lgcn_v2/model.py. Layer averaging and
BPR/ego-L2 are retained. Local default:32 dims,2 layers,no edge dropout,
large sampled batches; original winning recipe is NOT claimed reproduced.
"""
from dataclasses import dataclass, asdict
import time
import numpy as np
import torch
from torch import nn
from scipy import sparse
from .data import encode


@dataclass(frozen=True)
class GraphConfig:
    dim: int = 32
    layers: int = 2
    epochs: int = 12
    steps_per_epoch: int = 3
    batch_size: int = 32768
    lr: float = 0.01
    decay: float = 0.0001
    negative_power: float = 0.0
    seed: int = 2022
    threads: int = 4

    def __post_init__(self):
        if not (1 <= self.dim <= 128 and 0 <= self.layers <= 4 and 1 <= self.epochs <= 100
                and 1 <= self.steps_per_epoch <= 20 and 1 <= self.batch_size <= 131072
                and 1 <= self.threads <= 4 and 0 < self.lr <= 0.1
                and 0 <= self.decay <= 1 and 0 <= self.negative_power <= 1):
            raise ValueError("Graph configuration exceeds local experiment bounds")


def normalized_graph(history):
    n_users, n_items = history.shape
    bi = sparse.bmat([[None, history], [history.T, None]], format="coo", dtype=np.float32)
    inv = 1 / np.sqrt(np.maximum(np.asarray(bi.sum(axis=1)).ravel(), 1))
    values = bi.data * inv[bi.row] * inv[bi.col]
    return torch.sparse_coo_tensor(np.vstack([bi.row, bi.col]), values,
                                   (n_users + n_items,) * 2, check_invariants=True).coalesce()


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, dim=32, layers=2):
        super().__init__()
        self.n_users, self.layers = n_users, layers
        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_embedding = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def propagate(self, graph):
        embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        average = embeddings
        for _ in range(self.layers):
            embeddings = torch.sparse.mm(graph, embeddings)
            average = average + embeddings
        average = average / (self.layers + 1)
        return average[:self.n_users], average[self.n_users:]

    def bpr_loss(self, graph, users, pos, neg):
        u, i = self.propagate(graph)
        delta = (u[users] * (i[neg] - i[pos])).sum(-1)
        loss = nn.functional.softplus(delta).mean()
        reg = (self.user_embedding(users).square().sum()
               + self.item_embedding(pos).square().sum()
               + self.item_embedding(neg).square().sum()) / (2 * len(users))
        return loss, reg


class NegativeSampler:
    def __init__(self, history, power=0.0, seed=2022):
        self.history = history
        self.rng = np.random.default_rng(seed)
        self.n_items = history.shape[1]
        counts = np.asarray(history.sum(axis=0)).ravel()
        weights = counts ** power
        self.prob = weights / weights.sum()
        self.keys = np.sort(history.tocoo().row.astype(np.int64) * self.n_items + history.tocoo().col)
        if (np.diff(history.indptr) == self.n_items).any():
            raise ValueError("Cannot draw unseen negative for a saturated user")

    def is_known(self, users, items):
        keys = users.astype(np.int64) * self.n_items + items
        p = np.searchsorted(self.keys, keys)
        return self.keys[np.minimum(p, len(self.keys) - 1)] == keys

    def sample(self, size):
        # Uniform users, uniform known positives, rejection-sampled unknown negatives.
        users = self.rng.integers(self.history.shape[0], size=size)
        lengths = np.diff(self.history.indptr)[users]
        pos_idx = self.history.indptr[users] + (self.rng.random(size) * lengths).astype(int)
        positives = self.history.indices[pos_idx]
        negatives = self.rng.choice(self.n_items, size=size, p=self.prob)
        known = self.is_known(users, negatives)
        for _ in range(100):
            if not known.any():
                return users, positives, negatives
            negatives[known] = self.rng.choice(self.n_items, size=known.sum(), p=self.prob)
            known = self.is_known(users, negatives)
        raise RuntimeError("Negative rejection did not converge")


def train_graph(frame, config=GraphConfig()):
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    users, items, matrix = encode(frame)
    matrix.data[:] = 1
    graph = normalized_graph(matrix)
    model = LightGCN(len(users), len(items), config.dim, config.layers)
    sampler = NegativeSampler(matrix, config.negative_power, config.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    trace = []
    started = time.monotonic()
    for epoch in range(config.epochs):
        losses = []
        for _ in range(config.steps_per_epoch):
            batch = [torch.as_tensor(x, dtype=torch.long) for x in sampler.sample(config.batch_size)]
            optimizer.zero_grad()
            loss, reg = model.bpr_loss(graph, *batch)
            (loss + config.decay * reg).backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        trace.append({"epoch": epoch + 1, "bpr": float(np.mean(losses)),
                      "seconds": round(time.monotonic() - started, 3)})
        print(f'LightGCN layers={config.layers} epoch={epoch + 1}: bpr={trace[-1]["bpr"]:.5f}', flush=True)
    model.eval()
    with torch.no_grad():
        u, i = model.propagate(graph)
    return {"users": users, "items": items, "user_vectors": u.numpy(), "item_vectors": i.numpy(),
            "model": model, "trace": trace, "config": asdict(config), "edges": matrix.nnz}


def graph_score(fitted, rows):
    u = fitted["users"].get_indexer(rows.userId)
    i = fitted["items"].get_indexer(rows.itemId)
    good = (u >= 0) & (i >= 0)
    scores = np.zeros(len(rows), np.float32)
    # Bound intermediate embeddings; avoid materializing millions x dim all at once.
    positions = np.flatnonzero(good)
    for start in range(0, len(positions), 65536):
        p = positions[start:start + 65536]
        scores[p] = (fitted["user_vectors"][u[p]] * fitted["item_vectors"][i[p]]).sum(1)
    return scores
