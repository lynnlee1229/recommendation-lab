import numpy as np
import pytest
from reclab.search import BM25, Config, rrf, features
from reclab.retrieval import topk, metrics
from reclab.learning.retrieval import kmeans, partition_retrieval


def test_bm25_matches_hand_calculation_and_reserved_zero():
    model = BM25(['cat cat dog', 'dog'], k1=1.2, b=.75)
    scores = model.score('CAT cat')
    expected = np.log(2) * (2*2.2)/(2+1.2*(.25+.75*3/2))
    assert scores[1] == pytest.approx(expected)
    assert scores[2] == 0 and scores[0] == -np.inf
    assert topk(scores,2) == [1,2]
    assert topk(model.score('unseen'),2) == [1,2]


def test_rrf_union_ties_and_no_outside_fill():
    scores = rrf([[2,1],[3,1]],5,depth=2)
    assert scores[1] == pytest.approx(2/62)
    assert topk(scores,10) == [1,2,3]
    assert not np.isfinite(scores[4])


def test_multi_positive_metrics_and_missing_judgments():
    result=metrics({'q':[3,2,1]},{'q':{1,2}},2)
    assert result['recall@2']==.5 and result['mrr@2']==.5
    assert result['ndcg@2']==pytest.approx((1/np.log2(3))/(1+1/np.log2(3)))


def test_all_partitions_recover_exact_vector_topk():
    vectors=np.random.default_rng(42).normal(size=(40,8)).astype('float32')
    centers,labels=kmeans(vectors,clusters=4)
    query=vectors[7]
    ids,count=partition_retrieval(vectors,query,centers,labels,probes=4,k=10)
    assert count==40
    assert (ids+1).tolist()==topk(np.r_[-np.inf,vectors@query],10)


def test_config_and_features():
    with pytest.raises(ValueError):Config(rerank_depth=200)
    with pytest.raises(ValueError):Config(max_length=0)
    x=features('Cat DOG',['cat fish','dog cat'],[2],np.array([0,1,2]),np.array([0,.2,.5]),np.array([0,.01,.02]))
    assert x.shape==(1,5) and x[0,3]==1 and x[0,0]==2


def test_source_manifest_and_adapter():
    from reclab.search import load_data, verify_sources
    from reclab.paths import ROOT
    import json
    manifest=json.loads((ROOT/'docs/search_sources.json').read_text())
    assert len(manifest['assets'])==20
    assert all(len(x['sha256'])==64 and len(x['revision'])==40 for x in manifest['assets'])
    if not (ROOT/'data/search/qrels/test.tsv').exists():pytest.skip('Download search inputs for integration check')
    corpus,queries,splits=load_data()
    assert len(corpus)==5183 and len(splits['test'])==300
    assert not (set(splits['train'])&set(splits['test']))
    assert all(0 not in truth for truth in splits['test'].values())


def test_search_api_validation_and_shared_run(monkeypatch):
    from fastapi.testclient import TestClient
    from reclab import service, search
    class Index:
        def search(self, query, method, limit):
            return {'run_id':'fixture', 'method':method, 'items':[]}
    monkeypatch.setattr(search,'latest',lambda:({'run_id':'fixture'},{}))
    monkeypatch.setattr(service,'search_index',lambda run_id:Index())
    with TestClient(service.app,base_url='http://localhost') as client:
        assert client.post('/api/search',json={'query':'  '}).status_code==422
        assert client.post('/api/search',json={'query':'x','method':'../bad'}).status_code==422
        assert client.post('/api/search',json={'query':'x','limit':21}).status_code==422
        assert client.post('/api/search',json={'query':'x','method':'bm25'}).json()['method']=='bm25'
        assert client.post('/api/search',json={'query':'x'},headers={'Origin':'https://other.example'}).status_code==403
