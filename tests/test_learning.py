import json
import numpy as np
import pytest
import torch
from reclab.learning.retrieval import (
    binary_matrix,
    swing,
    objective_losses,
    kmeans,
    partition_retrieval,
    beam_search,
)
from reclab.learning.networks import (
    fm_interaction,
    fm_explicit,
    CrossV2,
    LHUC,
    MMoE,
    TargetAttention,
    calibrate_negative_sampling,
)
from reclab.learning.systems import (
    BloomFilter,
    bucket,
    greedy_dpp,
    slow_greedy_dpp,
    exhaustive_dpp,
    allocate_quotas,
    distillation_loss,
)
from reclab.paths import ROOT


def test_swing_hand_calculation():
    matrix = binary_matrix([(0, 0), (0, 1), (1, 0), (1, 1)], 2, 3)
    actual = swing(matrix, alpha=1)
    assert actual[0, 1] == pytest.approx(1 / 3)
    assert actual[1, 0] == pytest.approx(1 / 3)
    assert not np.any(actual.diagonal())
    with pytest.raises(ValueError):
        swing(matrix, alpha=0)


def test_pairwise_equals_two_class_listwise():
    losses = objective_losses(
        torch.tensor([1.0, 2.0, 0.1]), torch.tensor([0.0, 3.0, 0.2])
    )
    assert float(losses["pairwise"]) == pytest.approx(float(losses["listwise"]))


def test_partition_all_centroids_matches_exact():
    x = np.random.default_rng(9).normal(size=(60, 5))
    query = x[0]
    centers, labels = kmeans(x, clusters=4)
    assert np.all(labels == ((x[:, None, :] - centers) ** 2).sum(-1).argmin(1))
    ids, count = partition_retrieval(x, query, centers, labels, probes=4, k=10)
    assert count == len(x)
    assert ids.tolist() == np.argsort(-(x @ query))[:10].tolist()


def test_beam_can_miss_global_best():
    transition = {
        (): {"A": 0.6, "B": 0.4},
        ("A",): {"x": 0.5, "y": 0.5},
        ("B",): {"x": 0.99, "y": 0.01},
    }
    assert beam_search(transition, width=1)[0][0] == ("A", "x")
    assert beam_search(transition, width=4)[0][0] == ("B", "x")


def test_fm_equivalent_values_and_gradients():
    torch.manual_seed(4)
    x = torch.randn(3, 4, 5, requires_grad=True)
    a = fm_interaction(x)
    b = fm_explicit(x)
    assert torch.allclose(a, b, atol=1e-5)
    ga = torch.autograd.grad(a.sum(), x, retain_graph=True)[0]
    gb = torch.autograd.grad(b.sum(), x)[0]
    assert torch.allclose(ga, gb, atol=1e-5)


def test_cross_zero_residual_and_lhuc_bounds():
    layer = CrossV2(4)
    torch.nn.init.zeros_(layer.affine.weight)
    torch.nn.init.zeros_(layer.affine.bias)
    x = torch.randn(3, 4)
    assert torch.equal(layer(x, x), x)
    _, scales = LHUC(2, 4)(x, torch.randn(3, 2))
    assert bool(((scales > 0) & (scales < 2)).all())


def test_mmoe_gates_and_attention_padding():
    output, gates = MMoE(6)(torch.randn(4, 6))
    assert output.shape == (4, 2)
    assert torch.allclose(gates.sum(-1), torch.ones(4, 2), atol=1e-6)
    attention = TargetAttention(5)
    pooled, weights = attention(
        torch.randn(2, 4, 5),
        torch.randn(2, 5),
        torch.tensor([[True, False, True, False], [False] * 4]),
    )
    assert torch.allclose(weights[0].sum(), torch.tensor(1.0))
    assert torch.equal(weights[:, 1], torch.zeros(2))
    assert torch.equal(pooled[1], torch.zeros(5))
    assert torch.isfinite(pooled).all()


def test_sampling_calibration_inverse():
    p = np.array([0.01, 0.1, 0.5, 0.99])
    alpha = 0.2
    sampled = p / (p + alpha * (1 - p))
    assert np.allclose(calibrate_negative_sampling(sampled, alpha), p)


def test_bloom_no_false_negatives_and_buckets_stable():
    bf = BloomFilter(bits=128, hashes=3)
    for i in range(20):
        bf.add(i)
    assert all(bf.contains(i) for i in range(20))
    assert bucket("x", "layer") == bucket("x", "layer")
    with pytest.raises(ValueError):
        bucket("x", buckets=0)


def test_dpp_incremental_matches_slow_greedy():
    for seed in range(5):
        v = np.random.default_rng(seed).normal(size=(7, 8))
        kernel = v @ v.T
        selected = greedy_dpp(kernel, 3)
        assert selected == slow_greedy_dpp(kernel, 3)
        best, _ = exhaustive_dpp(kernel, 3)
        assert np.linalg.slogdet(kernel[np.ix_(selected, selected)])[1] <= best + 1e-8


def test_quota_conservation_and_kl_identity():
    counts = allocate_quotas(np.array([0.2, 0.8]), requests=10, quota=100)
    assert counts.sum() == 10
    assert not np.all(counts >= 100)
    logits = torch.randn(3, 6)
    assert abs(float(distillation_loss(logits, logits))) < 1e-6


def test_all_45_lectures_have_notebook_companions():
    curriculum = json.loads((ROOT / "docs/curriculum.json").read_text())
    assert len(curriculum["lessons"]) == 45
    for lesson in curriculum["lessons"]:
        companion = lesson["notebook"]
        assert (ROOT / "notebooks" / companion["file"]).is_file()
        assert companion["mode"]
