import math
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from reclab.competition import data
from reclab.competition.data import candidates, partition_users, corpus
from reclab.competition.memory import ItemCF
from reclab.competition.metrics import evaluate
from reclab.competition.graph import NegativeSampler, GraphConfig
from reclab.competition.reference import check_memory, check_graph, verify_references


@pytest.fixture
def histories():
    return pd.DataFrame([
        ('u1', 'i1', 5.), ('u1', 'i2', 3.), ('u2', 'i2', 4.),
        ('u2', 'i3', 5.), ('u3', 'i1', 2.), ('u3', 'i3', 4.),
        ('u3', 'i4', 5.)], columns=['userId', 'itemId', 'rating'])


def test_port_matches_pinned_original_scores_and_gradients(histories):
    check_memory(histories)
    check_graph(histories)


def test_real_data_parity():
    assert verify_references()['users'] == 8


def test_target_relevance_never_enters_feature_corpus(monkeypatch):
    original = data.interactions
    def guarded(market, part='train'):
        assert not (market.startswith('t') and part == 'valid_qrel')
        return original(market, part)
    monkeypatch.setattr(data, 'interactions', guarded)
    train = corpus(['s3', 't1', 't2'])
    assert not train.duplicated(['userId', 'itemId']).any()


def test_official_candidates_labels_and_group_isolation():
    for market in ('t1', 't2'):
        rows = candidates(market)
        assert rows.groupby('userId').size().eq(100).all()
        assert rows.groupby('userId').label.sum().eq(1).all()
        rows['partition'] = partition_users(rows)
        assert rows.groupby('userId').partition.nunique().eq(1).all()
        assert set(rows.partition) == {'fit', 'select', 'audit'}
        assert 'label' not in candidates(market, 'test')


def test_ndcg_user_weighted_combination_and_top10_boundary():
    rows = []
    scores = []
    for market, user, rank in [('t1', 'a', 1), ('t2', 'b', 2), ('t2', 'c', 11)]:
        for k in range(1, 12):
            rows.append((market, user, str(k), int(k == rank)))
            scores.append(-k)
    frame = pd.DataFrame(rows, columns=['market', 'userId', 'itemId', 'label'])
    table = evaluate(frame, scores).set_index('market')
    assert table.loc['combined', 'ndcg10'] == pytest.approx((1 + 1 / math.log2(3)) / 3)
    assert table.loc['combined', 'hr10'] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        evaluate(frame, [np.nan] * len(frame))


def test_negative_sampling_rejects_known_and_is_seeded():
    history = sparse.csr_matrix([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=np.float32)
    for power in (0., .75):
        a = NegativeSampler(history, power, seed=7)
        b = NegativeSampler(history, power, seed=7)
        u, p, n = a.sample(2000)
        assert not a.is_known(u, n).any()
        assert a.is_known(u, p).all()
        assert all(np.array_equal(x, y) for x, y in zip((u, p, n), b.sample(2000)))


def test_itemcf_unknowns_and_history_explanations(histories):
    model = ItemCF().fit(histories)
    rows = pd.DataFrame([('u1', 'i3'), ('missing', 'i1'), ('u1', 'unknown')],
                        columns=['userId', 'itemId'])
    pred = model.score(rows)
    assert pred[0] == pytest.approx(sum(x['similarity'] for x in model.explain('u1', 'i3')))
    assert (pred[1:] == 0).all()
    assert model.similarity.diagonal().sum() == 0


def test_resource_bounds():
    with pytest.raises(ValueError):
        GraphConfig(dim=2048)
    with pytest.raises(ValueError):
        GraphConfig(threads=15)


def test_competition_notebooks_and_source_targets_are_distinct():
    from reclab.notebook import TARGETS
    assert TARGETS['xmrec_graph'] == 'notebooks/13_XMRec_Graph.ipynb'
    assert TARGETS['xmrec_ranking'] == 'notebooks/14_XMRec_Ranking.ipynb'
    assert TARGETS['xmrec_code_graph'] == 'reclab/competition/graph.py'
    from fastapi.testclient import TestClient
    from reclab.service import app
    with TestClient(app, base_url='http://localhost') as client:
        rows = client.get('/api/notebooks').json()['notebooks']
        assert len(rows) >= 16
        assert {r['file'][:2] for r in rows} >= {'11','12','13','14','15'}
        for name in ['data','memory','graph','ranking','experiment','metrics']:
            response = client.get('/api/code/xmrec_' + name)
            assert response.status_code == 200
            assert response.json()['source']


def test_ndcg_matches_pinned_competition_solution():
    from reclab.competition.reference import original_ndcg
    for rank in (1, 2, 5, 10, 11):
        items = [str(i) for i in range(1, 12)]
        expected = 1 / np.log2(rank + 1) if rank <= 10 else 0.
        assert original_ndcg(items, str(rank)) == pytest.approx(expected, abs=1e-7)


def test_lightgbm_after_torch_training_completes_without_openmp_deadlock():
    import subprocess
    import sys
    script = '''
import torch,numpy as np,pandas as pd
from reclab.competition.ranking import train_ranker
torch.set_num_threads(4)
x=torch.randn(64,64,requires_grad=True);(x@x).sum().backward()
n=30000
rows=pd.DataFrame({'market':'t1','userId':np.repeat(np.arange(300).astype(str),100),
'label':np.tile([1]+[0]*99,300),'partition':np.repeat(['fit']*200+['select']*100,100)})
features=pd.DataFrame(np.random.default_rng(2022).normal(size=(n,4)),columns=list('abcd'))
for keep in (99,9):
 model,scores,receipt=train_ranker(rows,features,negative_keep=keep,rounds=5)
 assert model.n_jobs==1 and len(scores)==n and np.isfinite(scores).all()
 contributions=model.booster_.predict(features.iloc[:3],pred_contrib=True,num_threads=1)
 raw=model.booster_.predict(features.iloc[:3],raw_score=True,num_threads=1)
 assert np.allclose(contributions.sum(axis=1),raw)
print('MIXED_RUNTIME_PASS')
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert 'MIXED_RUNTIME_PASS' in result.stdout
