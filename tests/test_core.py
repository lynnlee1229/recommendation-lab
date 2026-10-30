import math
import numpy as np
import pandas as pd
import pytest
import torch
from reclab.data import Dataset
from reclab.models import TwoTower
from reclab.retrieval import metrics, topk, mmr
from reclab.train import TrainConfig


def test_ranking_metrics_against_hand_calculation():
    m = metrics({0: [2, 4], 1: [9, 8]}, {0: {2, 3}, 1: {8}}, 2)
    assert m["recall@2"] == pytest.approx(0.75)
    assert m["ndcg@2"] == pytest.approx(
        (1 / (1 + 1 / math.log2(3)) + 1 / math.log2(3)) / 2
    )
    assert m["mrr@2"] == pytest.approx(0.75)


def test_exclusion_padding_and_ties():
    assert topk(np.array([999, 1, 4, 4, 2]), 10, {2}) == [3, 4, 1]
    assert topk(np.array([999, 1]), 10, {1}) == []


def test_mmr_selects_relevance_first_and_then_diversifies():
    vectors = np.array([[0, 0], [1, 0], [1, 0], [0, 1]], dtype=float)
    scores = np.array([0, 10, 9, 8], dtype=float)
    selected, _ = mmr([1, 2, 3], scores, vectors, k=2, relevance=0.1)
    assert selected == [1, 3]
    assert mmr([1, 2, 3], scores, vectors, k=3, relevance=1)[0] == [1, 2, 3]
    assert mmr([1, 2, 3], scores, vectors, k=2, relevance=0)[0][0] == 1


def test_causal_examples_exclude_target_and_equal_timestamp_context():
    ds = Dataset.__new__(Dataset)
    ds.train = pd.DataFrame(
        {
            "user_idx": [0] * 6,
            "item_idx": [1, 2, 3, 4, 1, 5],
            "timestamp": [10, 20, 20, 30, 40, 50],
        }
    )
    users, h, targets, times = ds.examples(3)
    assert targets.tolist() == [4, 5]
    assert times.tolist() == [30, 50]
    assert h.tolist() == [[1, 2, 3], [3, 4, 1]]
    assert all(t not in row for t, row in zip(targets, h))


def test_padding_does_not_change_user_vector():
    torch.manual_seed(2)
    model = TwoTower(2, 8, 8, True, 0).eval()
    a = model.encode_user(torch.tensor([0]), torch.tensor([[1, 2]]))
    b = model.encode_user(torch.tensor([0]), torch.tensor([[0, 0, 1, 2]]))
    assert torch.allclose(a, b, atol=1e-7)
    assert torch.allclose(a.norm(dim=1), torch.ones(1), atol=1e-6)


def test_loss_gradient_finite_and_duplicate_negatives_masked():
    model = TwoTower(3, 8, 8, True, 0)
    users = torch.tensor([0, 1, 2])
    history = torch.tensor([[1, 2], [3, 4], [2, 3]])
    # All targets equal => no negative classes remain; cross entropy must be zero.
    loss = model.loss(users, history, torch.tensor([5, 5, 5]))
    assert float(loss.detach()) == pytest.approx(0, abs=1e-6)
    loss = model.loss(
        users, history, torch.tensor([5, 6, 7]), logq=torch.log(torch.ones(8) / 8)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_real_optimization_reduces_fixed_batch_loss():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    model = TwoTower(4, 12, 8, True, 0)
    users = torch.arange(4)
    history = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    targets = torch.tensor([3, 5, 7, 9])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    before = float(model.loss(users, history, targets).detach())
    for _ in range(30):
        optimizer.zero_grad()
        loss = model.loss(users, history, targets)
        loss.backward()
        optimizer.step()
    assert float(model.loss(users, history, targets).detach()) < before * 0.3


def test_resource_bounds():
    with pytest.raises(ValueError):
        TrainConfig(dim=10000)
    with pytest.raises(ValueError):
        TrainConfig(epochs=10000)
