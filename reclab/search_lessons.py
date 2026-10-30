"""Reusable bounded exercises for the search course; production uses search.py."""
from functools import lru_cache
import hashlib
import json
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .paths import ARTIFACTS, read_json
from .search import (latest, verify_sources, load_data, Config, FrozenModels,
                     BM25, rrf, features)
from .retrieval import topk, metrics
from .learning.retrieval import kmeans, partition_retrieval


@lru_cache(maxsize=1)
def context():
    receipt, result = latest()
    verify_sources()
    corpus, queries, splits = load_data()
    folder = ARTIFACTS / 'experiments/search' / receipt['run_id']
    vectors = np.load(folder / 'document_vectors.npy', allow_pickle=False)
    texts = (corpus.title.fillna('') + '. ' + corpus.text.fillna('')).tolist()
    assert vectors.shape == (len(texts), 384)
    return dict(receipt=receipt, result=result, corpus=corpus, queries=queries,
                truth=splits, texts=texts, vectors=vectors, bm25=BM25(texts),
                per_query=read_json(folder/'per_query.json'))


@lru_cache(maxsize=1)
def models():
    return FrozenModels(Config(**context()['receipt']['config']))


def query_vectors(qids):
    ctx = context()
    payload = [(q, ctx['queries'][q]) for q in qids]
    key = hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:20]
    path = ARTIFACTS/'search/lessons'/ctx['receipt']['run_id']/(key+'.npy')
    if path.exists():
        vectors = np.load(path, allow_pickle=False)
        if vectors.shape != (len(qids), 384) or not np.isfinite(vectors).all():
            raise ValueError('Invalid exercise vector cache')
        return vectors
    vectors = models().encode([ctx['queries'][q] for q in qids])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors


def score_query(qid, vector=None, lexical=None, depth=100):
    ctx = context()
    query = ctx['queries'][qid]
    lexical = ctx['bm25'] if lexical is None else lexical
    bm = lexical.score(query)
    vector = query_vectors([qid])[0] if vector is None else vector
    ds = np.r_[-np.inf, ctx['vectors'] @ vector]
    fs = rrf([topk(bm, depth), topk(ds, depth)], len(ctx['texts']), depth=depth)
    return bm, ds, fs


def labels(qids, split='test'):
    return {q: context()['truth'][split][q] for q in qids}


def candidate_rows(qids):
    ctx = context()
    vectors = query_vectors(qids)
    rows = []
    for qid, v in zip(qids, vectors):
        bm, ds, fs = score_query(qid, v)
        ids = topk(fs, 20)
        x = features(ctx['queries'][qid], ctx['texts'], ids, bm, ds, fs)
        truth = ctx['truth']['train'].get(qid, ctx['truth']['test'].get(qid))
        y = np.array([int(d in truth) for d in ids])
        rows.append(dict(qid=qid, ids=ids, x=x, y=y))
    return rows


def fit_ranker(rows, columns=(0,1,2,3,4), c=1.0):
    x = np.concatenate([r['x'][:, columns] for r in rows])
    y = np.concatenate([r['y'] for r in rows])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=500, random_state=42))
    with threadpool_limits(limits=4):
        model.fit(x, y)
    return model


def rerank_rows(model, rows, columns=(0,1,2,3,4)):
    predictions = {}
    for row in rows:
        scores = model.decision_function(row['x'][:, columns])
        order = np.lexsort((row['ids'], -scores))
        predictions[row['qid']] = [row['ids'][i] for i in order]
    return predictions


def split_training_queries(qids):
    """Query-level fit/select separation; never random-split candidate pairs."""
    if len(set(qids)) != len(qids) or len(qids) < 5:
        raise ValueError('Need at least five distinct queries')
    boundary = int(len(qids)*.8)
    return qids[:boundary], qids[boundary:]


def coarse_sweep(qids, probes=(1,2,4,12)):
    ctx = context()
    vectors = query_vectors(qids)
    with threadpool_limits(limits=4):
        centers, assignments = kmeans(ctx['vectors'], clusters=12, seed=42)
        exact = {q:topk(np.r_[-np.inf, ctx['vectors']@v],10) for q,v in zip(qids,vectors)}
        results = []
        for p in probes:
            pred, durations, counts, agreement = {}, [], [], []
            for q, v in zip(qids,vectors):
                start = time.perf_counter()
                ids, count = partition_retrieval(ctx['vectors'],v,centers,assignments,probes=p,k=10)
                durations.append((time.perf_counter()-start)*1000)
                pred[q]=(ids+1).tolist(); counts.append(count)
                agreement.append(len(set(pred[q])&set(exact[q]))/10)
            results.append(dict(probes=p, **metrics(pred,labels(qids),10),
                                agreement=float(np.mean(agreement)), candidates=float(np.mean(counts)),
                                index_p50_ms=float(np.median(durations))))
    return results


def ranking_workshop(fit_rows, select_rows):
    """Record two real fits in the shared registry; report selection queries only."""
    from .experiments import begin, finish
    from .paths import atomic_json
    ctx=context()
    config={'workshop':'linear_features','fit_queries':[r['qid'] for r in fit_rows],
            'select_queries':[r['qid'] for r in select_rows], 'C':1.,'seed':42,
            'parent_search_run':ctx['receipt']['run_id']}
    if set(config['fit_queries'])&set(config['select_queries']):
        raise ValueError('Query leakage between fit and selection')
    if not (set(config['fit_queries'])|set(config['select_queries'])) <= set(ctx['truth']['train']):
        raise ValueError('Workshops must fit/select only on training queries')
    folder=begin('search',config,ctx['receipt']['provenance'])
    trained={}; rows=[]
    try:
        for name,columns in [('bm25_only',(0,)),('five_features',(0,1,2,3,4))]:
            start=time.perf_counter(); model=fit_ranker(fit_rows,columns); seconds=time.perf_counter()-start
            trained[name]=(model,columns)
            prediction=rerank_rows(model,select_rows,columns)
            rows.append({'method':name,'features':len(columns),'fit_seconds':seconds,
                         'select':metrics(prediction,labels(config['select_queries'],'train'),10)})
            atomic_json(folder/(name+'.json'),{'columns':columns,'mean':model[0].mean_.tolist(),
                'scale':model[0].scale_.tolist(),'coef':model[1].coef_.tolist(),'intercept':model[1].intercept_.tolist()})
        finish(folder,{'kind':'workshop','scope':f'Query-grouped search teaching fits: {len(fit_rows)}/{len(select_rows)} training queries; select only; frozen official serving model unchanged.', 'rows':rows})
        return folder.name,trained
    except Exception as exc:
        receipt=read_json(folder/'receipt.json');receipt.update(state='failed',error=type(exc).__name__+': '+str(exc))
        atomic_json(folder/'receipt.json',receipt)
        raise
