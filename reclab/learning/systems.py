from hashlib import sha256
from itertools import combinations
import math
import numpy as np


def bucket(identifier, layer="retrieval", buckets=100):
    if buckets < 1:
        raise ValueError("buckets must be positive")
    key = f"{layer}:{identifier}".encode()
    return int.from_bytes(sha256(key).digest()[:8], "big") % buckets


class BloomFilter:
    def __init__(self, bits=4096, hashes=4):
        self.bits = np.zeros(bits, dtype=bool)
        self.hashes = hashes

    def positions(self, value):
        return [
            int.from_bytes(sha256(f"{h}:{value}".encode()).digest()[:8], "big")
            % len(self.bits)
            for h in range(self.hashes)
        ]

    def add(self, value):
        self.bits[self.positions(value)] = True

    def contains(self, value):
        return bool(self.bits[self.positions(value)].all())


def ild(vectors, ids):
    if len(ids) < 2:
        return 0.0
    return float(
        np.mean([1 - float(vectors[i] @ vectors[j]) for i, j in combinations(ids, 2)])
    )


def category_entropy(categories):
    _, counts = np.unique(categories, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def greedy_dpp(kernel, k):
    """Incremental Cholesky greedy MAP. It is an approximation to global MAP."""
    n = len(kernel)
    k = min(k, n)
    residual = kernel.diagonal().copy()
    coefficients = np.zeros((k, n))
    selected = []
    for step in range(k):
        item = int(np.argmax(residual))
        if residual[item] <= 1e-12:
            break
        correction = coefficients[:step, item] @ coefficients[:step, :]
        update = (kernel[item, :] - correction) / math.sqrt(residual[item])
        coefficients[step, :] = update
        residual = np.maximum(residual - update**2, 0)
        selected.append(item)
        residual[selected] = -np.inf
    return selected


def slow_greedy_dpp(kernel, k):
    selected = []
    for _ in range(min(k, len(kernel))):
        choices = []
        for i in range(len(kernel)):
            if i in selected:
                continue
            ids = selected + [i]
            sign, logdet = np.linalg.slogdet(kernel[np.ix_(ids, ids)])
            choices.append((float(logdet) if sign > 0 else -np.inf, -i, i))
        best = max(choices)
        if not np.isfinite(best[0]):
            break
        selected.append(best[2])
    return selected


def exhaustive_dpp(kernel, k):
    options = []
    for ids in combinations(range(len(kernel)), k):
        sign, logdet = np.linalg.slogdet(kernel[np.ix_(ids, ids)])
        if sign > 0:
            options.append((logdet, ids))
    return max(options, key=lambda pair: pair[0])


def constrained_ranking(candidates, categories, k=10, max_consecutive=2):
    """Greedy order with an explicit partial-result policy when no item is feasible."""
    selected = []
    remaining = list(candidates)
    while remaining and len(selected) < k:
        feasible = []
        for i in remaining:
            same_tail = len(selected) >= max_consecutive and all(
                categories[j] == categories[i] for j in selected[-max_consecutive:]
            )
            if not same_tail:
                feasible.append(i)
        if not feasible:
            break
        item = feasible[0]
        selected.append(item)
        remaining.remove(item)
    return selected


def allocate_quotas(quality, requests, quota=100, boost=2.0):
    """Controlled traffic simulation. No real-user CTR or causal lift is inferred."""
    exposures = np.zeros(len(quality), dtype=int)
    for _ in range(requests):
        priority = quality + boost * np.maximum(quota - exposures, 0) / max(quota, 1)
        item = int(np.argmax(priority))
        exposures[item] += 1
    return exposures


def distillation_loss(student, teacher, temperature=2.0):
    """Listwise distillation over the same candidate list."""
    import torch.nn.functional as F

    return F.kl_div(
        F.log_softmax(student / temperature, dim=-1),
        F.softmax(teacher.detach() / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)
