"""Reusable binary probability evaluation; distinct from retrieval relevance metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss


def evaluate(y, probability, bins=10):
    y = np.asarray(y)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError('Expected nonempty aligned binary labels and probabilities')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid binary probabilities or labels')
    if not isinstance(bins, int) or bins < 1:
        raise ValueError('Invalid bin count')
    groups = np.minimum((p * bins).astype(int), bins - 1)
    reliability = []
    for b in range(bins):
        mask = groups == b
        if mask.any():
            reliability.append({'bin': b, 'count': int(mask.sum()),
                                'mean_probability': float(p[mask].mean()),
                                'click_rate': float(y[mask].mean())})
    return {'n': len(y), 'positives': int(y.sum()), 'click_rate': float(y.mean()),
            'mean_probability': float(p.mean()),
            'auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            'logloss': float(log_loss(y, np.clip(p, 1e-7, 1-1e-7), labels=[0, 1])),
            'brier': float(brier_score_loss(y, p)),
            'ece': float(sum(r['count'] * abs(r['mean_probability'] - r['click_rate']) for r in reliability) / len(y)),
            'reliability': reliability}
