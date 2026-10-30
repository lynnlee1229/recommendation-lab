"""Reload the latest completed TinyBERT teaching checkpoint and verify exact saved predictions."""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from reclab.search_lessons import context
from reclab.search import latest
from reclab.retrieval import topk,metrics
from reclab.paths import ROOT,ARTIFACTS,atomic_json,read_json
folders=[]
for p in (ARTIFACTS/'experiments/search').glob('*/receipt.json'):
 r=read_json(p)
 if r.get('state')=='completed' and r.get('config',{}).get('workshop')=='tinybert_finetune': folders.append(p.parent)
p=sorted(folders)[-1]; r=read_json(p/'receipt.json'); result=read_json(p/'result.json')
ctx=context(); tok=AutoTokenizer.from_pretrained(ARTIFACTS/'search/models/reranker',local_files_only=True)
model=AutoModelForSequenceClassification.from_pretrained(p/'model',local_files_only=True).eval()
torch.set_num_threads(4); predictions={}
with torch.inference_mode():
 for q in r['config']['select_queries']:
  ids=topk(ctx['bm25'].score(ctx['queries'][q]),10)
  batch=tok([ctx['queries'][q]]*len(ids),[ctx['texts'][d-1] for d in ids],padding=True,truncation=True,max_length=96,return_tensors='pt')
  scores=model(**batch).logits[:,0].numpy()
  predictions[q]=[ids[i] for i in np.lexsort((ids,-scores))]
assert predictions==read_json(p/'selection_predictions.json')
assert metrics(predictions,{q:ctx['truth']['train'][q] for q in predictions},10)==result['rows'][1]['select']
assert latest()[0]['run_id']==r['config']['parent_search_run']
print('checkpoint predictions and saved metrics exact; official latest unchanged',p.name)
atomic_json(ROOT/'artifacts/search/lessons/finetune-verification.json',{'run_id':p.name,**result,'checkpoint_reload':'exact predictions and recomputed metrics','official_latest_unchanged':True})
