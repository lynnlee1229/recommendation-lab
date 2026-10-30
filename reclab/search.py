"""Bounded real SciFact retrieval, sharing the lab's metrics and experiment registry."""
from dataclasses import dataclass, asdict
import hashlib
import json
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .paths import ROOT, ARTIFACTS, atomic_json, read_json
from .retrieval import topk, metrics
from .experiments import begin, finish
from .learning.retrieval import kmeans, partition_retrieval


@dataclass(frozen=True)
class Config:
    train_queries: int = 200
    candidate_depth: int = 100
    rerank_depth: int = 20
    max_length: int = 256
    batch_size: int = 16
    seed: int = 42

    def __post_init__(self):
        if not (1 <= self.train_queries <= 809 and 10 <= self.rerank_depth <= self.candidate_depth <= 500):
            raise ValueError('Invalid query or candidate budget')
        if not (32 <= self.max_length <= 512 and 1 <= self.batch_size <= 64):
            raise ValueError('Invalid inference budget')


def verify_sources(download=False):
    from download_public_assets import fetch, sha
    spec = read_json(ROOT / 'docs/search_sources.json')
    for row in spec['assets']:
        if download:
            fetch(row)
        path = ROOT / row['destination']
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Frozen search input mismatch: ' + row['destination'])
    return spec


def load_data():
    base = ROOT / 'data/search'
    corpus = pd.read_parquet(base / 'scifact/corpus/corpus-00000-of-00001.parquet')
    corpus['_id'] = corpus['_id'].astype(str)
    corpus = corpus.sort_values('_id', kind='stable').reset_index(drop=True)
    ids = {d: i + 1 for i, d in enumerate(corpus['_id'])}  # 0 reserved by shared topk
    queries = pd.read_parquet(base / 'scifact/queries/queries-00000-of-00001.parquet')
    queries['_id'] = queries['_id'].astype(str)
    queries = dict(zip(queries['_id'], queries['text']))
    splits = {}
    for split in ('train', 'test'):
        frame = pd.read_csv(base / f'qrels/{split}.tsv', sep='\t', dtype={'query-id': str, 'corpus-id': str})
        if frame.duplicated(['query-id', 'corpus-id']).any():
            raise ValueError('Duplicate qrel')
        truth = {}
        for q, d, score in frame.itertuples(index=False, name=None):
            if q not in queries or d not in ids:
                raise ValueError('Dangling qrel')
            if score > 0:
                truth.setdefault(q, set()).add(ids[d])
        splits[split] = truth
    if set(splits['train']) & set(splits['test']):
        raise ValueError('Train/test query overlap')
    return corpus, queries, splits


class BM25:
    def __init__(self, texts, k1=1.2, b=.75):
        self.vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b', dtype=np.float64)
        counts = self.vectorizer.fit_transform(texts).tocsr()
        length = np.asarray(counts.sum(1)).ravel()
        df = np.asarray((counts > 0).sum(0)).ravel()
        idf = np.log1p((len(texts) - df + .5) / (df + .5))
        rows = np.repeat(np.arange(len(texts)), np.diff(counts.indptr))
        counts.data = idf[counts.indices] * counts.data * (k1 + 1) / (
            counts.data + k1 * (1 - b + b * length[rows] / max(length.mean(), 1)))
        self.index = counts.tocsr()

    def score(self, query):
        q = self.vectorizer.transform([query])
        q.data[:] = 1  # binary query terms, conventional no query-tf factor
        return np.r_[-np.inf, (self.index @ q.T).toarray().ravel()]


def rrf(rankings, n_documents, depth=100, constant=60):
    scores = np.zeros(n_documents + 1)
    scores[0] = -np.inf
    for ranking in rankings:
        for rank, doc in enumerate(ranking[:depth], 1):
            scores[doc] += 1 / (constant + rank)
    # Documents outside the candidate union must never fill a truncated candidate list.
    scores[scores == 0] = -np.inf
    return scores


class FrozenModels:
    def __init__(self, config):
        import torch
        from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
        torch.set_num_threads(4)
        self.torch, self.config = torch, config
        base = ARTIFACTS / 'search/models'
        self.tokenizer = AutoTokenizer.from_pretrained(base / 'minilm', local_files_only=True)
        self.encoder = AutoModel.from_pretrained(base / 'minilm', local_files_only=True).eval()
        self.pair_tokenizer = AutoTokenizer.from_pretrained(base / 'reranker', local_files_only=True)
        self.reranker = AutoModelForSequenceClassification.from_pretrained(base / 'reranker', local_files_only=True).eval()

    def encode(self, texts):
        result = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.config.batch_size):
                batch = self.tokenizer(texts[start:start+self.config.batch_size], padding=True, truncation=True,
                                       max_length=self.config.max_length, return_tensors='pt')
                states = self.encoder(**batch).last_hidden_state
                mask = batch['attention_mask'].unsqueeze(-1)
                pooled = (states * mask).sum(1) / mask.sum(1).clamp(min=1)
                result.append(self.torch.nn.functional.normalize(pooled, dim=1).numpy())
        return np.concatenate(result)

    def rerank(self, query, texts):
        result = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.config.batch_size):
                subset = texts[start:start+self.config.batch_size]
                batch = self.pair_tokenizer([query]*len(subset), subset, padding=True, truncation=True,
                                            max_length=self.config.max_length, return_tensors='pt')
                result.extend(self.reranker(**batch).logits[:, 0].tolist())
        return np.asarray(result)


def features(query, texts, candidates, lexical, dense, fusion):
    terms = set(re.findall(r'\w+', query.lower()))
    return np.array([[lexical[d], dense[d], fusion[d],
                      len(terms & set(re.findall(r'\w+', texts[d-1].lower()))) / max(len(terms), 1),
                      np.log1p(len(texts[d-1]))] for d in candidates])


def run(config=Config(), download=False):
    provenance = verify_sources(download)
    folder = begin('search', asdict(config), provenance)
    try:
        with threadpool_limits(limits=4):
            result = _run(config, folder)
        finish(folder, result)
        print('COMPLETED', folder.name, flush=True)
        return result
    except Exception as exc:
        receipt = read_json(folder / 'receipt.json')
        receipt.update(state='failed', error=type(exc).__name__ + ': ' + str(exc))
        atomic_json(folder / 'receipt.json', receipt)
        raise


def _run(config, folder):
    corpus, queries, splits = load_data()
    texts = (corpus.title.fillna('') + '. ' + corpus.text.fillna('')).tolist()
    started = time.perf_counter()
    lexical = BM25(texts)
    models = FrozenModels(config)
    vectors = models.encode(texts)
    np.save(folder / 'document_vectors.npy', vectors)
    centers, labels = kmeans(vectors, clusters=12, seed=config.seed)
    build_seconds = time.perf_counter() - started
    print('INDEXED', len(texts), round(build_seconds, 2), flush=True)
    def retrieve(q):
        start = time.perf_counter(); bm = lexical.score(q); b_rank = topk(bm, config.candidate_depth)
        b_ms = (time.perf_counter()-start)*1000
        start = time.perf_counter(); v = models.encode([q])[0]; encode_ms = (time.perf_counter()-start)*1000
        start = time.perf_counter(); ds = np.r_[-np.inf, vectors @ v]; d_rank = topk(ds, config.candidate_depth)
        dense_ms = encode_ms + (time.perf_counter()-start)*1000
        start = time.perf_counter(); fs = rrf([b_rank, d_rank], len(texts), config.candidate_depth)
        fused = topk(fs, config.candidate_depth); fusion_ms = (time.perf_counter()-start)*1000
        return bm, ds, fs, b_rank, d_rank, fused, v, b_ms, dense_ms, fusion_ms, encode_ms
    train_ids = sorted(splits['train'])[:config.train_queries]
    x, y = [], []
    for qid in train_ids:
        bm, ds, fs, _, _, candidates, *_ = retrieve(queries[qid])
        candidates = candidates[:config.rerank_depth]
        x.extend(features(queries[qid], texts, candidates, bm, ds, fs))
        y.extend(int(d in splits['train'][qid]) for d in candidates)
    if len(set(y)) < 2:
        raise ValueError('Training candidates need both classes')
    ranker = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=500, random_state=config.seed))
    train_start = time.perf_counter(); ranker.fit(x, y); train_seconds = time.perf_counter()-train_start
    atomic_json(folder/'linear_model.json', {'features':['bm25','cosine','rrf','query_term_coverage','log_characters'],
        'scaler_mean':ranker[0].mean_.tolist(),'scaler_scale':ranker[0].scale_.tolist(),
        'coef':ranker[1].coef_.tolist(),'intercept':ranker[1].intercept_.tolist()})
    names = ['bm25','dense','hybrid_rrf','linear_rerank','tinybert_rerank','dense_coarse2','dense_coarse12']
    predictions = {n:{} for n in names}; timing = {n:[] for n in names}
    agreements = {n:[] for n in ['dense_coarse2','dense_coarse12']}; candidates_counts = {n:[] for n in agreements}
    rows_query = []; candidate_recall = []; judged = {n:[] for n in names}
    for index, qid in enumerate(sorted(splits['test'])):
        query=queries[qid]; truth=splits['test'][qid]
        bm, ds, fs, br, dr, fused, v, bt, dt, ft, et = retrieve(query)
        hybrid_ms = bt+dt+ft
        ranked={'bm25':br,'dense':dr,'hybrid_rrf':fused}
        elapsed={'bm25':bt,'dense':dt,'hybrid_rrf':hybrid_ms}
        candidates=fused[:config.rerank_depth]
        candidate_recall.append(len(set(candidates)&truth)/len(truth))
        start=time.perf_counter(); score=ranker.decision_function(features(query,texts,candidates,bm,ds,fs))
        ranked['linear_rerank']=[candidates[i] for i in np.lexsort((candidates,-score))]
        elapsed['linear_rerank']=hybrid_ms+(time.perf_counter()-start)*1000
        start=time.perf_counter(); score=models.rerank(query,[texts[d-1] for d in candidates])
        ranked['tinybert_rerank']=[candidates[i] for i in np.lexsort((candidates,-score))]
        elapsed['tinybert_rerank']=hybrid_ms+(time.perf_counter()-start)*1000
        for probes in [2,12]:
            name=f'dense_coarse{probes}'; start=time.perf_counter()
            found,count=partition_retrieval(vectors,v,centers,labels,probes=probes,k=100)
            ranked[name]=(found+1).tolist();elapsed[name]=et+(time.perf_counter()-start)*1000
            agreements[name].append(len(set(ranked[name][:10])&set(dr[:10]))/10)
            candidates_counts[name].append(count)
        per={}
        for name in names:
            predictions[name][qid]=ranked[name];timing[name].append(elapsed[name])
            judged[name].append(len(set(ranked[name][:10])&truth)/10)
            per[name]=metrics({qid:ranked[name]},{qid:truth},10)
        rows_query.append({'qid':qid,'query':query,'relevant_document_ids':[str(corpus.iloc[d-1]['_id']) for d in sorted(truth)],
                          'metrics':per,'top10':{n:[str(corpus.iloc[d-1]['_id']) for d in ranked[n][:10]] for n in names}})
        if index%50==0: print('AUDIT',index,flush=True)
    rows=[]
    for name in names:
        rows.append({'method':name,**metrics(predictions[name],splits['test'],10),
                     'recall@100':metrics(predictions[name],splits['test'],100)['recall@100'],
                     'latency_p50_ms':float(np.median(timing[name])), 'latency_p95_ms':float(np.percentile(timing[name],95)),
                     'known_positive_fraction@10':float(np.mean(judged[name]))})
    failures=sorted(rows_query,key=lambda r:(r['metrics']['tinybert_rerank']['ndcg@10']-r['metrics']['bm25']['ndcg@10'],r['qid']))[:8]
    for failure in failures:
        failure['top_titles']={n:[str(corpus.iloc[int(np.flatnonzero(corpus['_id'].values==d)[0])]['title']) for d in docs[:3]] for n,docs in failure['top10'].items()}
    atomic_json(folder/'per_query.json',rows_query)
    atomic_json(folder/'latencies.json',timing)
    result={'scope':f'SciFact BEIR: all {len(texts):,} documents, all {len(splits["test"])} test queries; {len(train_ids)} lexicographically selected train queries; frozen MiniLM/TinyBERT, no neural fine-tuning. Unjudged documents treated as nonrelevant, not verified negatives.',
            'rows':rows,'failures':failures,'index_seconds':build_seconds,'linear_train_seconds':train_seconds,
            'train_queries':train_ids,'train_pairs':len(y),'train_positive_pairs':sum(y),
            'corpus_documents':len(texts),'test_queries':len(splits['test']),
            'rerank_candidate_recall@20':float(np.mean(candidate_recall)),
            'approximation':{n:{'exact_top10_agreement':float(np.mean(agreements[n])), 'mean_candidates':float(np.mean(candidates_counts[n]))} for n in agreements},
            'protocol':{'k':10,'candidate_depth':config.candidate_depth,'rerank_depth':config.rerank_depth,'rrf_constant':60,'bm25':{'k1':1.2,'b':.75},'label':'binary qrel score > 0; shared binary NDCG',
                        'latency':'warm CPU per-query wall time; includes tokenization and query encoding, retrieval/sort and rerank; excludes index/model loading, disk/API/network. Sequential mixed-method measurements, not throughput.',
                        'training':f'{len(train_ids)} train queries only; top20 hybrid candidates; unjudged-as-negative weak supervision for logistic baseline. C=1 fixed; no test tuning.',
                        'truncation':f'title + abstract; max {config.max_length} wordpieces; paired reranker combined max length; normalized mean pooling',
                        'limits':'English biomedical claim evidence retrieval only; tiny corpus, incomplete qrels, possible pretraining overlap; no biomedical advice or full BEIR replication.'}}
    return result


def latest():
    from .experiments import catalog, result
    rows=[r for r in catalog() if r['domain']=='search' and r['state']=='completed' and not r.get('config',{}).get('workshop') and not r.get('annotation',{}).get('superseded')]
    if not rows: raise FileNotFoundError('Run python -m reclab.search first')
    row=rows[-1]
    return row, result('search',row['run_id'])


class SearchIndex:
    """Read-only serving adapter for the latest completed shared experiment."""
    def __init__(self, receipt):
        verify_sources()
        self.config = Config(**receipt['config'])
        self.run_id = receipt['run_id']
        self.corpus, _, _ = load_data()
        self.texts = (self.corpus.title.fillna('') + '. ' + self.corpus.text.fillna('')).tolist()
        self.bm25 = BM25(self.texts)
        self.models = FrozenModels(self.config)
        folder = ARTIFACTS / 'experiments/search' / self.run_id
        self.vectors = np.load(folder / 'document_vectors.npy', allow_pickle=False)
        self.linear = read_json(folder / 'linear_model.json')

    def search(self, query, method='hybrid_rrf', limit=10):
        started = time.perf_counter()
        bm = self.bm25.score(query)
        br = topk(bm, self.config.candidate_depth)
        scores = bm
        if method != 'bm25':
            vector = self.models.encode([query])[0]
            ds = np.r_[-np.inf, self.vectors @ vector]
            scores = ds
            if method != 'dense':
                scores = rrf([br, topk(ds, self.config.candidate_depth)], len(self.texts), self.config.candidate_depth)
                candidates = topk(scores, self.config.rerank_depth)
                if method == 'tinybert_rerank':
                    values = self.models.rerank(query, [self.texts[d-1] for d in candidates])
                elif method == 'linear_rerank':
                    x = features(query,self.texts,candidates,bm,ds,scores)
                    values = ((x-np.array(self.linear['scaler_mean']))/np.array(self.linear['scaler_scale'])) @ np.array(self.linear['coef'])[0] + self.linear['intercept'][0]
                if method in ('tinybert_rerank','linear_rerank'):
                    scores = np.full(len(self.texts)+1, -np.inf)
                    scores[candidates] = values
        ids = topk(scores, limit)
        return {'run_id':self.run_id,'method':method,'latency_ms':(time.perf_counter()-started)*1000,
                'scope':'Scientific evidence retrieval; scores are not truth probabilities. First request also loads models outside this timer.',
                'items':[{'document_id':str(self.corpus.iloc[d-1]['_id']),
                          'title':str(self.corpus.iloc[d-1]['title']),
                          'text':str(self.corpus.iloc[d-1]['text']), 'score':float(scores[d])} for d in ids]}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    run(download=args.download)
