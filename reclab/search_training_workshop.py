"""Small real TinyBERT fine-tune. Fixed training queries; no test access or serving mutation."""
import time
import numpy as np
from .paths import ARTIFACTS, atomic_json, read_json
from .search_lessons import context
from .retrieval import topk, metrics
from .experiments import begin, finish


def run():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(4); torch.manual_seed(42)
    ctx=context()
    # Fixed in advance, no selection from the already seen test set.
    qids=sorted(ctx['truth']['train'])[:40]
    fit, select=qids[:32], qids[32:]
    assert not set(fit)&set(select)
    assert not set(qids)&set(ctx['truth']['test'])
    config={'workshop':'tinybert_finetune','parent_search_run':ctx['receipt']['run_id'],
            'fit_queries':fit,'select_queries':select,'seed':42,'epochs':2,
            'batch_size':8,'max_length':96,'lr':2e-5,'device':'cpu',
            'budget':'32 fit queries, <=128 pairs, 32 updates, <1GB working tensors; no new downloads',
            'negative_policy':'top BM25 unjudged training documents, weak negatives',
            'licenses':'existing CC-BY-SA-4.0 SciFact and Apache-2.0 TinyBERT; fixed SHAs in docs/search_sources.json'}
    folder=begin('search',config,ctx['receipt']['provenance'])
    try:
        base=ARTIFACTS/'search/models/reranker'
        tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
        model=AutoModelForSequenceClassification.from_pretrained(base,local_files_only=True)
        def candidates(q):
            # Training pool contains a judged positive. Selection pool stays retrieval-only.
            ids=list(topk(ctx['bm25'].score(ctx['queries'][q]),10))
            return ids
        select_ids={q:candidates(q) for q in select}
        def evaluate():
            model.eval(); pred={}; shapes=[]
            with torch.inference_mode():
                for q,ids in select_ids.items():
                    batch=tok([ctx['queries'][q]]*len(ids),[ctx['texts'][d-1] for d in ids],
                              padding=True,truncation='longest_first',max_length=96,return_tensors='pt')
                    score=model(**batch).logits[:,0].numpy(); shapes.append(list(batch['input_ids'].shape))
                    pred[q]=[ids[i] for i in np.lexsort((ids,-score))]
            return metrics(pred,{q:ctx['truth']['train'][q] for q in select},10),pred,shapes
        frozen,_,shapes=evaluate()
        pairs=[]
        for q in fit:
            pos=sorted(ctx['truth']['train'][q])[:1]
            neg=[d for d in candidates(q) if d not in ctx['truth']['train'][q]][:3]
            pairs += [(q,d,float(d in pos)) for d in pos+neg]
        assert len(pairs)<=128
        before={n:p.detach().clone() for n,p in model.named_parameters() if 'encoder.layer.0.attention.self.query.weight' in n}
        assert before
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5)
        losses=[]; start=time.perf_counter()
        for epoch in range(2):
            model.train()
            order=np.random.default_rng(42+epoch).permutation(len(pairs))
            for offset in range(0,len(pairs),8):
                rows=[pairs[i] for i in order[offset:offset+8]]
                batch=tok([ctx['queries'][q] for q,d,y in rows],[ctx['texts'][d-1] for q,d,y in rows],
                          padding=True,truncation=True,max_length=96,return_tensors='pt')
                y=torch.tensor([y for q,d,y in rows])
                optimizer.zero_grad(); scores=model(**batch).logits[:,0]
                loss=torch.nn.functional.binary_cross_entropy_with_logits(scores,y)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step(); losses.append(float(loss.detach()))
        seconds=time.perf_counter()-start
        changed=sum(float((dict(model.named_parameters())[n].detach()-p).abs().sum()) for n,p in before.items())
        assert changed>0 and np.isfinite(losses).all()
        tuned,pred,_=evaluate()
        model.save_pretrained(folder/'model',safe_serialization=True)
        atomic_json(folder/'selection_predictions.json',pred)
        result={'kind':'workshop','scope':'32/8 query-disjoint SciFact training queries; 2 epochs CPU TinyBERT full fine-tuning; weak negatives; select only; no test or serving change.',
                'rows':[{'method':'frozen_tinybert','select':frozen},{'method':'finetuned_tinybert','select':tuned}],
                'losses':losses,'fit_seconds':seconds,'pairs':len(pairs),'encoder_weight_l1_change':changed,
                'pair_shapes':shapes,'candidate_hit_rate':float(np.mean([bool(set(ids)&set(ctx['truth']['train'][q])) for q,ids in select_ids.items()])),
                'limits':'Tiny selected training subset; pretraining overlap unknown; unjudged are not verified negative; fixed top10 candidate ceiling; not paper reproduction or calibrated relevance probability.'}
        finish(folder,result)
        return folder.name,result
    except Exception as exc:
        receipt=read_json(folder/'receipt.json'); receipt.update(state='failed',error=str(exc)); atomic_json(folder/'receipt.json',receipt)
        raise
