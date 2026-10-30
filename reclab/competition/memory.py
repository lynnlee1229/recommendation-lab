"""Sparse ports of OPDAI icf_v2 (IUF) and icf_v3 (rating cosine).

S = D_item^-1/2 B.T W_user B D_item^-1/2, diagonal zero.
IUF has W_uu=1/log(1+|history_u|). Rating cosine uses R.T R.
Queries sum similarities over the binary history, as in both originals.
No neighbor pruning or history truncation is applied by default.
"""
import numpy as np
from .data import encode


class ItemCF:
    def __init__(self, kind="iuf", top_k=0):
        if kind not in ("iuf", "cosine", "rating") or top_k < 0:
            raise ValueError("Invalid ItemCF configuration")
        self.kind, self.top_k = kind, top_k

    def fit(self, frame):
        self.users, self.items, ratings = encode(frame)
        binary = ratings.copy()
        binary.data[:] = 1
        self.history = binary
        values = ratings if self.kind == "rating" else binary
        if self.kind == "iuf":
            weight = 1 / np.log1p(np.diff(binary.indptr))
            weighted = values.multiply(weight[:, None]).tocsr()
        else:
            weighted = values
        sim = (values.T @ weighted).tocsr()
        norm = np.sqrt(np.asarray(values.power(2).sum(axis=0)).ravel())
        inv = 1 / np.maximum(norm, 1e-12)
        sim = sim.multiply(inv[:, None]).multiply(inv[None, :]).tocsr()
        sim.setdiag(0)
        sim.eliminate_zeros()
        if self.top_k:
            for row in range(sim.shape[0]):
                a, b = sim.indptr[row:row + 2]
                if b - a > self.top_k:
                    ids = np.lexsort((sim.indices[a:b], -sim.data[a:b]))[self.top_k:]
                    sim.data[a + ids] = 0
            sim.eliminate_zeros()
        self.similarity = sim
        self.popularity = np.asarray(binary.sum(axis=0)).ravel()
        return self

    def score(self, rows, batch_users=256):
        # Matrix multiplication is bounded to user batches; never form users x catalog dense.
        result = np.zeros(len(rows), dtype=np.float32)
        row_users = self.users.get_indexer(rows.userId)
        row_items = self.items.get_indexer(rows.itemId)
        unique = np.unique(row_users[row_users >= 0])
        for start in range(0, len(unique), batch_users):
            us = unique[start:start + batch_users]
            positions = np.flatnonzero(np.isin(row_users, us) & (row_items >= 0))
            predicted = (self.history[us] @ self.similarity).tocsr()
            local = np.searchsorted(us, row_users[positions])
            result[positions] = np.asarray(predicted[local, row_items[positions]]).ravel()
        return result

    def explain(self, user, candidate):
        u, c = self.users.get_indexer([user])[0], self.items.get_indexer([candidate])[0]
        if u < 0 or c < 0:
            return []
        hist = self.history[u].indices
        contribution = self.similarity[hist, c].toarray().ravel()
        return sorted([{"history_item": self.items[i], "similarity": float(s)}
                       for i, s in zip(hist, contribution)], key=lambda x: -x["similarity"])
