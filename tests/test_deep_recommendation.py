import json
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from reclab.deep_recommendation import Config, SequenceEncoder, negatives
from reclab import experiments


def test_negative_prefix_mask_and_distribution():
    h = np.array([[0, 1, 2], [0, 2, 3]])
    p = np.array([3, 1])
    draws = negatives(h, p, np.array([0, .1, .1, .1, .7]), np.random.default_rng(1), 50)
    assert np.all(draws == 4)
    with pytest.raises(ValueError):
        negatives(h, p, np.array([0, 1/3, 1/3, 1/3, 0]), np.random.default_rng(1))


@pytest.mark.parametrize('kind', ['mean', 'gru', 'attention'])
def test_padding_and_gradients(kind):
    torch.manual_seed(2)
    model = SequenceEncoder(9, 8, kind, 5)
    histories = torch.tensor([[0, 0, 1, 2, 3], [0, 1, 2, 3, 4]])
    output = model.encode(histories)
    assert output.shape == (2, 8) and torch.isfinite(output).all()
    output.sum().backward()
    assert torch.equal(model.embedding.weight.grad[0], torch.zeros(8))
    with pytest.raises(ValueError):
        model.encode(torch.zeros((1, 5), dtype=torch.long))
    if kind == 'mean':
        assert torch.allclose(output[0], model.encode(torch.tensor([[3, 0, 2, 1, 0]]))[0])
    if kind == 'gru':
        assert torch.allclose(output[0], model.encode(torch.tensor([[0, 1, 2, 3]]))[0], atol=1e-6)


def test_config_bounds():
    with pytest.raises(ValueError):
        Config(epochs=0)


def test_shared_experiment_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'BASE', tmp_path)
    folder = experiments.begin('search', {'seed': 42}, {'source': 'fixture'})
    assert experiments.catalog()[0]['state'] == 'running'
    experiments.finish(folder, {'ndcg': .5})
    assert experiments.result('search', folder.name)['ndcg'] == .5
    assert experiments.catalog()[0]['state'] == 'completed'
    with pytest.raises(ValueError):
        experiments.result('search', '../../etc/passwd')
    with pytest.raises(ValueError):
        experiments.begin('unknown', {}, {})


def test_shared_api_and_result_boundaries(tmp_path, monkeypatch):
    from reclab.service import app
    monkeypatch.setattr(experiments, 'BASE', tmp_path)
    folder = experiments.begin('recommendation', {}, {'source': 'fixture'})
    experiments.finish(folder, {'rows': []})
    with pytest.raises(ValueError):
        experiments.finish(folder, {'rows': [1]})
    with TestClient(app, base_url='http://localhost') as client:
        assert client.get('/api/experiments').json()['experiments'][0]['state'] == 'completed'
        assert client.get('/api/experiments/recommendation/' + folder.name).json() == {'rows': []}
        assert client.get('/api/experiments/unknown/' + folder.name).status_code == 404


def test_exploratory_run_annotation_keeps_original_result(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'BASE', tmp_path)
    folder = experiments.begin('recommendation', {}, {})
    experiments.finish(folder, {'metric': 1})
    (folder / 'annotation.json').write_text(json.dumps({'superseded': True, 'reason': 'controls changed'}))
    assert experiments.catalog()[0]['annotation']['superseded']
    assert experiments.result('recommendation', folder.name) == {'metric': 1}
