"""Check real ads predictions and preserved recommendation interfaces; requires local trained inputs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from reclab import ads
from reclab.paths import ARTIFACTS,atomic_json
from reclab.service import app
receipt,out=ads.latest()
folder=ARTIFACTS/'experiments/ads'/receipt['run_id']
frame,_=ads.load_data(receipt['config']['rows'])
pred=np.load(folder/'predictions.npz',allow_pickle=False)
masks,groups=ads.split_groups(frame)
np.testing.assert_array_equal(groups,pred['group'])
assert np.array_equal(frame.label.to_numpy(),pred['label'])
assert not any(set(groups[masks[a]])&set(groups[masks[b]]) for a in masks for b in masks if a!=b)
from reclab.probability import evaluate
for row in out['rows']:
    for split in ('select','audit'):
        actual=evaluate(pred['label'][masks[split]],pred[row['method']][masks[split]])
        for metric in ('auc','logloss','brier','ece'):
            assert abs(actual[metric]-row[split][metric])<1e-12
pure=ads.Predictor(receipt['run_id'])
check=np.random.default_rng(42).choice(np.flatnonzero(masks['audit']),size=500,replace=False)
encoded=ads.encode(frame.iloc[check],pure.vocab)
values=np.array([pure.raw_probability(x) for x in encoded])
assert np.max(np.abs(values-pred[out['selected_base']][check]))<1e-10
with TestClient(app,base_url='http://localhost') as c:
    example=c.get('/api/ads/example').json()
    response=c.post('/api/ads/predict',json=example)
    assert response.status_code==200,response.text
    first=int(np.flatnonzero(masks['audit'])[0])
    assert abs(response.json()['raw']-pred[out['selected_base']][first])<1e-10
    assert abs(response.json()['sigmoid']-pred['selected_sigmoid'][first])<1e-10
    assert abs(response.json()['isotonic']-pred['selected_isotonic'][first])<1e-10
    selected_ids=np.flatnonzero(masks['audit'])[:30]
    for i in selected_ids:
        row=frame.iloc[i]
        body={'numeric':[None if pd.isna(row[n]) else float(row[n]) for n in ads.NUM], 'categorical':[row[n] for n in ads.CAT]}
        r=c.post('/api/ads/predict',json=body)
        assert r.status_code==200,r.text
        for method,key in [(out['selected_base'],'raw'),('selected_sigmoid','sigmoid'),('selected_isotonic','isotonic')]:
            assert abs(r.json()[key]-pred[method][i])<1e-10
    # Four original recommendation models and ItemCF use unchanged APIs.
    summary=c.get('/api/summary').json()
    user=summary['users'][0]['user_idx']
    recomm=[]
    for run in summary['runs']:
        rid=run.get('run_id',run.get('id'))
        if not rid: continue
        r=c.post('/api/recommend',json={'user_idx':user,'run_id':rid,'channel':'two_tower','limit':10})
        assert r.status_code==200,(rid,r.text)
        assert len(r.json()['items'])==10
        recomm.append(rid)
    r=c.post('/api/recommend',json={'user_idx':user,'run_id':'baseline-history32','channel':'itemcf','limit':10})
    assert r.status_code==200 and len(r.json()['items'])==10
    books=c.get('/api/notebooks').json()['notebooks']
    from reclab.notebook import TARGETS
    expected=sum(value.startswith('notebooks/') and value.endswith('.ipynb') for value in TARGETS.values())
    assert len(books)==expected and all(r['available'] for r in books)
    assert c.get('/api/code/ads').status_code==200
report={'run_id':receipt['run_id'],'status':'PASS','offline_online_probability_rows':30,'probability_tolerance':1e-10,'tree_export_golden_rows':500,'saved_metrics_recomputed':True,'feature_group_overlap':0,'notebooks':len(books),'recommendation_versions':recomm,'itemcf':True,'simulation_budget_all_seeds':all(r['budget_ok'] for r in out['simulations'])}
atomic_json(folder/'acceptance.json',report)
print(report)
