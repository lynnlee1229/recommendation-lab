"""Controlled real-data recommendation experiments; no held-out test access."""
from dataclasses import dataclass, asdict
import argparse
import time
import numpy as np
import torch
from torch import nn
from .data import Dataset
from .retrieval import topk, metrics
from .learning.retrieval import kmeans
from .experiments import begin, finish
from .paths import atomic_json


@dataclass(frozen=True)
class Config:
    epochs: int = 4
    samples: int = 12000
    dim: int = 24
    history: int = 20
    seeds: tuple = (42, 2026)

    def __post_init__(self):
        if not (1 <= self.epochs <= 20 and 100 <= self.samples <= 100000 and
                8 <= self.dim <= 64 and 2 <= self.history <= 50 and 1 <= len(self.seeds) <= 5):
            raise ValueError('Experiment exceeds local bounds')


class SequenceEncoder(nn.Module):
    def __init__(self, n_items, dim=24, kind='mean', length=20):
        super().__init__()
        if kind not in ('mean', 'gru', 'attention'):
            raise ValueError('Unknown encoder')
        self.kind = kind
        self.embedding = nn.Embedding(n_items, dim, padding_idx=0)
        self.project = nn.Linear(dim, dim)
        if kind == 'gru':
            self.gru = nn.GRU(dim, dim, batch_first=True)
        if kind == 'attention':
            self.position = nn.Embedding(length, dim)
            self.attention = nn.MultiheadAttention(dim, 2, batch_first=True, dropout=0)

    def encode(self, history):
        valid = history.ne(0)
        if not valid.any(1).all():
            raise ValueError('History must contain a real item')
        x = self.embedding(history)
        if self.kind == 'gru':
            # Remove left padding before packing; padding must not update hidden state.
            compact = nn.utils.rnn.pad_sequence([row[mask] for row, mask in zip(x, valid)], batch_first=True)
            packed = nn.utils.rnn.pack_padded_sequence(compact, valid.sum(1).cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.gru(packed)
            pooled = hidden[-1]
        elif self.kind == 'attention':
            x = x + self.position(torch.arange(history.shape[1], device=x.device))[None]
            x, _ = self.attention(x, x, x, key_padding_mask=~valid, need_weights=False)
            pooled = (x * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
        else:
            pooled = x.sum(1) / valid.sum(1, keepdim=True)
        return nn.functional.normalize(self.project(pooled), dim=-1)

    def items(self):
        return nn.functional.normalize(self.embedding.weight, dim=-1)


def negatives(history, positives, probability, rng, width=1):
    # Only causal prefix and current target are excluded. Unknown future positives remain unknown.
    out = rng.choice(len(probability), size=(len(history), width), p=probability)
    blocked = [set(row) | {int(pos), 0} for row, pos in zip(history, positives)]
    if any(sum(probability[i] for i in b) >= 1 - 1e-12 for b in blocked):
        raise ValueError('No negative support')
    for row, excluded in enumerate(blocked):
        for _ in range(100):
            bad = np.isin(out[row], list(excluded))
            if not bad.any():
                break
            out[row, bad] = rng.choice(len(probability), size=int(bad.sum()), p=probability)
        else:
            raise RuntimeError('Negative sampling failed')
    return out


def evaluate(ds, model, users, history_len, shuffle=False):
    rng = np.random.default_rng(19)
    histories = np.stack([ds.history_array(u, history_len) for u in users])
    if shuffle:
        for row in histories:
            ids = row[row != 0].copy(); rng.shuffle(ids); row[row != 0] = ids
    with torch.no_grad():
        vectors = model.items().numpy()
        queries = model.encode(torch.tensor(histories)).numpy()
    predictions = {u: topk(vectors @ q, 20, ds.seen[u]) for u, q in zip(users, queries)}
    return predictions, queries, vectors


def run(config=Config()):
    torch.set_num_threads(4)
    ds = Dataset()
    users, histories, positives, times = ds.examples(config.history)
    sample_idx = np.sort(np.random.default_rng(7).choice(len(positives), min(config.samples, len(positives)), replace=False))
    histories, positives, times = histories[sample_idx], positives[sample_idx], times[sample_idx]
    assert times.max() < ds.manifest['train_end_ms']
    truth, audit = ds.evaluation('valid', 1000)
    eval_users = list(truth)
    if not eval_users:
        raise ValueError('No validation users')
    folder = begin('recommendation', asdict(config), {'dataset_version': ds.manifest['dataset_version'] if 'dataset_version' in ds.manifest else ds.manifest,
                     'evaluation': 'warm novel full-catalog validation; no final-test', 'audit': audit})
    arms = [('mean', 'uniform'), ('mean', 'popular'), ('mean', 'hard'), ('gru', 'uniform'), ('attention', 'uniform')]
    rows, per_user = [], []
    try:
        for seed in config.seeds:
            for kind, sampler in arms:
                torch.manual_seed(seed)
                rng = np.random.default_rng(seed)
                order_rng = np.random.default_rng(seed)
                model = SequenceEncoder(ds.n_items, config.dim, kind, config.history)
                optimizer = torch.optim.Adam(model.parameters(), lr=.003)
                probability = np.ones(ds.n_items) if sampler != 'popular' else ds.popularity.astype(float) ** .75
                probability[0] = 0; probability /= probability.sum()
                curve = []; started = time.monotonic()
                for epoch in range(config.epochs):
                    losses = []
                    for ids in np.array_split(order_rng.permutation(len(positives)), max(1, int(np.ceil(len(positives) / 256)))):
                        h = torch.tensor(histories[ids]); p = torch.tensor(positives[ids])
                        n = torch.tensor(negatives(histories[ids], positives[ids], probability, rng, 16 if sampler == 'hard' else 1))
                        query = model.encode(h); vectors = model.items()
                        negative_scores = (query[:, None] * vectors[n]).sum(-1)
                        loss = nn.functional.softplus(negative_scores.max(1).values - (query * vectors[p]).sum(-1)).mean()
                        optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
                    curve.append(float(np.mean(losses)))
                model.eval()
                predictions, queries, vectors = evaluate(ds, model, eval_users, config.history)
                shuffled, _, _ = evaluate(ds, model, eval_users, config.history, shuffle=True)
                row = {'encoder': kind, 'sampler': sampler, 'seed': seed, 'parameters': sum(p.numel() for p in model.parameters()),
                       'seconds': time.monotonic() - started, 'loss': curve, **metrics(predictions, truth, 20),
                       'shuffled': metrics(shuffled, truth, 20),
                       'coverage@20': len(set(i for pred in predictions.values() for i in pred)) / (ds.n_items - 1),
                       'recommended_mean_train_frequency': float(np.mean([ds.popularity[i] for pred in predictions.values() for i in pred]))}
                per_user.extend({'encoder': kind, 'sampler': sampler, 'seed': seed, 'user_idx': u,
                                 **metrics({u: predictions[u]}, {u: truth[u]}, 20)} for u in truth)
                # Shared coarse index teaching implementation; evaluate search loss separately from model quality.
                if kind == 'mean' and sampler == 'uniform':
                    centers, labels = kmeans(vectors[1:], clusters=12, seed=seed)
                    row['index'] = []
                    for probes in (1, 4, 12):
                        approximate = {}; overlaps = []; scanned = []
                        t = time.perf_counter()
                        for u, q in zip(eval_users, queries):
                            chosen = np.argsort(((centers - q) ** 2).sum(1), kind='stable')[:probes]
                            candidates = np.flatnonzero(np.isin(labels, chosen)) + 1
                            scores = np.full(ds.n_items, -np.inf); scores[candidates] = vectors[candidates] @ q
                            approximate[u] = topk(scores, 20, ds.seen[u])
                            overlaps.append(len(set(approximate[u]) & set(predictions[u])) / max(1, len(predictions[u])))
                            scanned.append(len(candidates))
                        row['index'].append({'probes': probes, 'exact_top20_overlap': float(np.mean(overlaps)),
                                             'mean_scanned': float(np.mean(scanned)), 'seconds': time.perf_counter() - t,
                                             **metrics(approximate, truth, 20)})
                rows.append(row)
                torch.save(model.state_dict(), folder / f'{kind}-{sampler}-{seed}.pt')
                print(kind, sampler, seed, row['ndcg@20'], flush=True)
        atomic_json(folder / 'per-user.json', per_user)
        result = {'rows': rows, 'samples': len(positives), 'evaluation_users': len(truth),
                  'scope': 'Bounded real-data ablation; loss is not CTR; popularity proposal uses train-only aggregate frequency; hard negatives are model-selected from 16 unseen-prefix candidates. Validation is a visible development benchmark, not an untouched audit.',
                  'controls': 'Same samples, seed, epochs, dimension, optimizer and full-catalog metric. GRU/attention parameter counts differ; no equal-capacity claim. No final-test access.'}
        finish(folder, result)
        print(folder, flush=True)
        return folder
    except Exception as exc:
        from .paths import read_json
        receipt = read_json(folder / 'receipt.json'); receipt.update(state='failed', error=type(exc).__name__)
        atomic_json(folder / 'receipt.json', receipt)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--samples', type=int, default=12000)
    args = parser.parse_args()
    run(Config(epochs=args.epochs, samples=args.samples))
