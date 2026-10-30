import itertools
from fastapi.testclient import TestClient
from reclab.service import app
from reclab.learning.search_curriculum import (
    DOCUMENTS, PositionalIndex, terms, CandidateCache, pipeline, segment, graded_ndcg)


def test_positional_index_matches_exhaustive_all_bigrams():
    index=PositionalIndex(DOCUMENTS)
    vocabulary=sorted({t for f in DOCUMENTS.values() for t in terms(f['title'])})
    for pair in itertools.product(vocabulary,repeat=2):
        expected=sorted(d for d,f in DOCUMENTS.items() if any(terms(f['title'])[i:i+2]==list(pair) for i in range(len(terms(f['title'])))))
        assert index.retrieve(' '.join(pair),phrase=True)==expected


def test_cache_version_and_ttl_prevent_deleted_document():
    cache=CandidateCache(ttl=5)
    assert cache.get(' A ',0,0,lambda:['deleted'])[1]=='miss'
    assert cache.get('a',1,0,lambda:[])[0]==['deleted']
    assert cache.get('a',1,1,lambda:[])[0]==[]
    assert cache.get('a',6,1,lambda:['new'])[0]==['new']
    assert cache.cost==3


def test_constructed_api_keeps_evidence_and_failures_explicit():
    client=TestClient(app,base_url='http://localhost')
    r=client.post('/api/search/workshop/trace',json={'query':'battrey repair','fail_semantic':True})
    assert r.status_code==200
    assert r.json()['evidence']=='constructed mechanism'
    assert r.json()['trace']['fallback']=='lexical'
    assert client.post('/api/search/workshop/trace',json={'query':''}).status_code==422
    assert pipeline('unfindable')['empty']
    course=client.get('/api/search/course').json()
    assert [x['id'] for x in course['units']]==[f'S{i:02}' for i in range(1,19)]


def test_grades_and_segmentation_invariants():
    assert graded_ndcg(['a','b'],{'a':3,'b':1},2)==1
    vocabulary={'研究','研究生','生命','命','起源'}
    assert segment('研究生命起源',vocabulary)!=segment('研究生命起源',vocabulary,True)
