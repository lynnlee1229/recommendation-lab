"""Search curriculum mechanisms. All examples here are constructed, not user logs."""
from collections import Counter, defaultdict
import re
import unicodedata
import numpy as np


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).lower().split())


def segment(text, vocabulary, reverse=False):
    text = normalize(text).replace(' ', '')
    if reverse:
        return [t[::-1] for t in segment(text[::-1], {w[::-1] for w in vocabulary})][::-1]
    out = []
    while text:
        token = next((text[:i] for i in range(len(text), 0, -1) if text[:i] in vocabulary), text[0])
        out.append(token); text = text[len(token):]
    return out


def terms(text):
    return re.findall(r'\w+', normalize(text))


DOCUMENTS = {
    'a': {'title': 'apple repair', 'body': 'official phone battery service', 'topic': 'service', 'quality': 1.0},
    'b': {'title': 'apple fruit', 'body': 'battery repair mentioned in fruit spam', 'topic': 'food', 'quality': .1},
    'c': {'title': 'phone battery repair', 'body': 'independent apple service guide', 'topic': 'service', 'quality': .8},
    'd': {'title': 'apple repair guide', 'body': 'apple phone battery repair instructions', 'topic': 'guide', 'quality': .9},
}


class PositionalIndex:
    def __init__(self, docs):
        self.docs = docs
        self.postings = defaultdict(lambda: defaultdict(list))
        for did, fields in docs.items():
            for field in ('title', 'body'):
                for pos, term in enumerate(terms(fields[field])):
                    self.postings[(field, term)][did].append(pos)

    def retrieve(self, query, field='title', mode='OR', phrase=False):
        ts = terms(query)
        if not ts: return []
        sets = [set(self.postings[(field, t)]) for t in ts]
        found = set.intersection(*sets) if mode == 'AND' or phrase else set.union(*sets)
        if phrase:
            found = {d for d in found if any(all(p+i in self.postings[(field,t)][d] for i,t in enumerate(ts))
                                             for p in self.postings[(field,ts[0])][d])}
        return sorted(found)


def field_rank(query, docs=DOCUMENTS, title_weight=2., required=()):
    weights = Counter(terms(query))
    rows = []
    for d, f in docs.items():
        all_terms = set(terms(f['title']+' '+f['body']))
        if not set(required) <= all_terms: continue
        score = sum(w*(title_weight*terms(f['title']).count(t)+terms(f['body']).count(t)) for t,w in weights.items())
        if score: rows.append((d, score))
    return sorted(rows, key=lambda x:(-x[1],x[0]))


def intent(query):
    """Inspectable rule mechanism; abstain on ambiguous/unknown queries."""
    ts = set(terms(query))
    hits = {'transaction': bool(ts & {'buy','price'}), 'support': bool(ts & {'repair','battery'}),
            'navigation': bool(ts & {'official','login'})}
    labels = [k for k,v in hits.items() if v]
    return labels[0] if len(labels)==1 else 'abstain'


def rewrite(query, aggressive=False):
    ts = terms(query)
    fixes = {'battrey':'battery', 'cellphone':'phone'}
    out = [fixes.get(t,t) for t in ts]
    # Deliberately bad expansion demonstrates entity/constraint drift.
    if aggressive: out = [t for t in out if t != 'official'] + ['fruit']
    return ' '.join(out)


def graded_ndcg(order, judgments, k=3):
    def dcg(values): return sum((2**v-1)/np.log2(i+2) for i,v in enumerate(values[:k]))
    ideal = dcg(sorted(judgments.values(), reverse=True))
    return dcg([judgments.get(d,0) for d in order])/ideal if ideal else 0.


class CandidateCache:
    """Logical clock, explicit index generation, and TTL; no system clock sleeps."""
    def __init__(self, ttl=5):
        self.ttl, self.rows, self.cost = ttl, {}, 0
    def get(self, query, now, generation, loader, check_generation=True):
        key = normalize(query); row = self.rows.get(key)
        if row and now-row[0] < self.ttl and (not check_generation or row[1]==generation):
            return list(row[2]), 'hit'
        result = list(loader()); self.cost += 1
        self.rows[key] = (now,generation,result)
        return result, 'miss'


def pipeline(query, cache=None, now=0, generation=0, fail_semantic=False):
    """Traceable constructed pipeline, not the official SciFact ranking API."""
    cache = CandidateCache() if cache is None else cache
    q = rewrite(query)
    lexical = [d for d,_ in field_rank(q)]
    # Synonym channel is an explicit toy semantic substitute, not a neural model.
    semantic = [] if fail_semantic else (['a','c','d'] if 'phone' in q else [])
    ids, state = cache.get(q,now,generation,lambda:sorted(set(lexical+semantic)))
    ranking = [d for d,_ in field_rank(q,{d:DOCUMENTS[d] for d in ids})]
    return {'query':q,'intent':intent(q),'lexical':lexical,'semantic':semantic,
            'cache':state,'candidates':ids,'ranking':ranking,
            'fallback': 'lexical' if fail_semantic else None, 'empty':not ranking}


# Entirely constructed session log; repeated entries encode made-up frequency.
SESSIONS = [('apple repair','phone battery repair'),('apple repair','apple official repair'),
            ('apple fruit','apple pie'),('apple repair','phone battery repair'),
            ('phone battery repair','battery price'),('apple fruit','apple pie')]


def suggestions(prefix, sessions=SESSIONS):
    counts = Counter(q for s in sessions for q in s)
    return [q for q,_ in sorted(counts.items(),key=lambda x:(-x[1],x[0])) if q.startswith(normalize(prefix))]


def related(query, sessions=SESSIONS):
    counts=Counter(b for a,b in sessions if a==query)
    return [q for q,_ in sorted(counts.items(),key=lambda x:(-x[1],x[0]))]


def category(query):
    """Constructed topic taxonomy independent of action intent."""
    ts=set(terms(query))
    hits={'device':bool(ts & {'phone','battery','cellphone'}),
          'food':bool(ts & {'fruit','pie'})}
    labels=[k for k,v in hits.items() if v]
    return labels[0] if len(labels)==1 else 'unknown'
