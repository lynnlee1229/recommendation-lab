import numpy as np
from scipy import sparse


def itemcf(dataset):
    pairs = dataset.train[["user_idx", "item_idx"]].drop_duplicates()
    matrix = sparse.csr_matrix(
        (np.ones(len(pairs), dtype=np.float32), (pairs.user_idx, pairs.item_idx)),
        shape=(dataset.n_users, dataset.n_items),
    )
    # Inverse user activity reduces extremely active users' co-occurrence contribution.
    activity = np.asarray(matrix.sum(1)).ravel()
    weighted = sparse.diags(1 / np.log1p(np.maximum(activity, 1))) @ matrix
    co = (matrix.T @ weighted).tocsr()
    frequency = np.asarray(matrix.sum(0)).ravel()
    norm = sparse.diags(1 / np.sqrt(np.maximum(frequency, 1)))
    sim = (norm @ co @ norm).tocsr()
    sim.setdiag(0)
    sim.eliminate_zeros()
    # Keep 100 neighbors per item, bounded memory and easy inspection.
    for i in range(sim.shape[0]):
        a, b = sim.indptr[i : i + 2]
        if b - a > 100:
            loc = np.argpartition(sim.data[a:b], -100)[:-100]
            sim.data[a + loc] = 0
    sim.eliminate_zeros()
    return sim


def cf_scores(similarity, history, n_items):
    scores = np.zeros(n_items, dtype=np.float32)
    if history:
        ids = history[-50:]
        weights = np.geomspace(0.2, 1, len(ids)).astype(np.float32)
        scores = np.asarray(similarity[ids].T @ weights).ravel()
    return scores


def topk(scores, k, excluded=()):
    values = np.asarray(scores, dtype=np.float64).copy()
    values[0] = -np.inf
    if excluded:
        values[list(excluded)] = -np.inf
    candidates = np.flatnonzero(np.isfinite(values))
    if not len(candidates):
        return []
    # Stable tie rule by item index; shared across every algorithm.
    order = np.lexsort((candidates, -values[candidates]))[:k]
    return candidates[order].astype(int).tolist()


def metrics(predictions, targets, k):
    recall, ndcg, mrr, hits = [], [], [], []
    for u, truth in targets.items():
        pred = predictions[u][:k]
        rel = np.array([i in truth for i in pred], dtype=float)
        count = int(rel.sum())
        recall.append(count / len(truth))
        hits.append(float(count > 0))
        discount = 1 / np.log2(np.arange(len(rel)) + 2)
        ideal = (1 / np.log2(np.arange(min(k, len(truth))) + 2)).sum()
        ndcg.append(float((rel * discount).sum() / ideal) if ideal else 0.0)
        ranks = np.flatnonzero(rel)
        mrr.append(1 / float(ranks[0] + 1) if len(ranks) else 0.0)
    avg = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {
        f"recall@{k}": avg(recall),
        f"ndcg@{k}": avg(ndcg),
        f"mrr@{k}": avg(mrr),
        f"hit_rate@{k}": avg(hits),
    }


def mmr(candidates, scores, vectors, k=10, relevance=0.8):
    if not candidates:
        return [], []
    raw = np.array([scores[i] for i in candidates], dtype=float)
    scaled = (raw - raw.min()) / max(float(np.ptp(raw)), 1e-8)
    quality = dict(zip(candidates, scaled))
    chosen, steps = [], []
    remaining = list(candidates)
    while remaining and len(chosen) < k:
        scored = []
        for i in remaining:
            redundancy = max(
                [float(vectors[i] @ vectors[j]) for j in chosen], default=0.0
            )
            value = (
                quality[i]
                if not chosen
                else relevance * quality[i] - (1 - relevance) * redundancy
            )
            scored.append((value, -i, i, redundancy))
        value, _, item, penalty = max(scored)
        chosen.append(item)
        remaining.remove(item)
        steps.append(
            {
                "item_idx": item,
                "relevance_normalized": float(quality[item]),
                "max_similarity_to_selected": penalty,
                "mmr_score": value,
            }
        )
    return chosen, steps
