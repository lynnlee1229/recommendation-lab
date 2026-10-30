import numpy as np
import pandas as pd
import pytest
from reclab.probability import evaluate
from reclab.auction import simulate, market
from reclab.ads import split_groups, encode, vocabulary, FEATURES, NUM, CAT, Config


def test_probability_proper_scores_and_endpoint_reliability():
    r=evaluate([0,1],[.25,.75],bins=2)
    assert r['auc']==1 and r['brier']==.0625
    assert r['logloss']==pytest.approx(-np.log(.75))
    assert r['ece']==.25 and sum(b['count'] for b in r['reliability'])==2
    r=evaluate([0,1],[0,1])
    assert r['brier']==0 and r['ece']==0 and r['reliability'][-1]['bin']==9
    assert evaluate([0,0],[.1,.2])['auc'] is None
    for y,p in [([2],[.2]),([1],[np.nan]),([0],[-.1]),([0,1],[.2])]:
        with pytest.raises(ValueError):evaluate(y,p)


def test_group_split_no_label_leakage_or_duplicate_cross_split():
    frame=pd.DataFrame({name:np.arange(500).astype(str) for name in FEATURES})
    frame=pd.concat([frame,frame.iloc[:10]],ignore_index=True)
    masks,groups=split_groups(frame)
    assert np.stack(list(masks.values())).sum(axis=0).tolist()==[1]*len(frame)
    for a,ma in masks.items():
        for b,mb in masks.items():
            if a!=b: assert not set(groups[ma])&set(groups[mb])
    frame['label']=1
    masks2,groups2=split_groups(frame)
    np.testing.assert_array_equal(groups,groups2)


def test_train_vocabulary_unseen_category_is_other():
    frame=pd.DataFrame({name:[1.,2.] for name in NUM}|{name:['seen','frequent'] for name in CAT})
    vocab=vocabulary(frame,2)
    other=frame.copy(); other.loc[0,CAT]='unseen'
    x=encode(other,vocab)
    assert x.shape==(2,39) and (x[0,13:]==0).all()
    assert (x[1,13:]>0).all()
    with pytest.raises(ValueError):Config(rows=999)


def test_second_price_payment_and_realized_value_hand_case():
    data={'estimate':np.array([[.5,.3]]),'true_p':np.array([[.4,.2]]),'value':np.array([2.,2.]),
          'clicks':np.array([[True,False]]),'external':np.array([.2])}
    r=simulate(data,budget=2.)
    assert r['spend']==pytest.approx(.6) and r['expected_click_value']==.8
    assert r['realized_click_value']==2 and r['wins']==1
    assert r['advertisers'][0]['spend']==.6
    data['external']=np.array([1.1])
    assert simulate(data,budget=2.)['wins']==0


def test_simulation_pairing_budget_and_pacing_constraints():
    data=market(7,requests=100)
    for paced in (True,False):
        r=simulate(data,budget=.7,paced=paced)
        assert r['budget_ok'] and all(a['spend']<=.7+1e-8 for a in r['advertisers'])
        if paced:
            assert all(max(t['spend'])<=.7*t['fraction']+1e-8 for t in r['trace'])
    assert simulate(data)==simulate(market(7,requests=100))
    with pytest.raises(ValueError):simulate(data,budget=-1)


def test_ads_api_validation_and_same_origin():
    from fastapi.testclient import TestClient
    from reclab.service import app
    with TestClient(app,base_url='http://localhost') as c:
        assert c.post('/api/ads/predict',json={'numeric':[1],'categorical':[]}).status_code==422
        assert c.post('/api/ads/predict',json={'numeric':[1e20]*13,'categorical':['x']*26}).status_code==422
        assert c.post('/api/ads/simulate',json={'budget':-1}).status_code==422
        assert c.post('/api/ads/simulate',json={'strategy':'bad'}).status_code==422
        assert c.post('/api/ads/simulate',json={},headers={'Origin':'https://other.example'}).status_code==403
        r=c.post('/api/ads/simulate',json={'budget':.01}).json()
        assert r['budget_ok'] and 'SYNTHETIC' in r['scope']


def test_exported_tree_numeric_missing_and_categorical_routes():
    from reclab.ads import Predictor
    predictor=object.__new__(Predictor)
    predictor.trees={'tree_info':[{'tree_structure':{'split_feature':0,'decision_type':'<=','threshold':2.,'missing_type':'NaN','default_left':True,'left_child':{'leaf_value':-1.},'right_child':{'split_feature':1,'decision_type':'==','_category_set':{2,4},'left_child':{'leaf_value':1.},'right_child':{'leaf_value':2.}}}}]}
    assert predictor.raw_probability([np.nan,2])==pytest.approx(1/(1+np.exp(1)))
    assert predictor.raw_probability([3,2])==pytest.approx(1/(1+np.exp(-1)))
    assert predictor.raw_probability([3,-1])==pytest.approx(1/(1+np.exp(-2)))


def test_local_serving_matches_real_offline_probabilities():
    from reclab.ads import latest, Predictor
    from reclab.paths import ARTIFACTS
    try: receipt,result=latest()
    except FileNotFoundError: pytest.skip('Train ads first for integration check')
    folder=ARTIFACTS/'experiments/ads'/receipt['run_id']
    if not (folder/'ctr_trees.json').exists(): pytest.skip('Serving tree export unavailable')
    from fastapi.testclient import TestClient
    from reclab.service import app
    with TestClient(app,base_url='http://localhost') as client:
        example=client.get('/api/ads/example').json()
        response=client.post('/api/ads/predict',json=example)
        assert response.status_code==200,response.text
        pred=np.load(folder/'predictions.npz',allow_pickle=False)
        first=np.flatnonzero(pred['split']==3)[0]
        for method,key in [(result['selected_base'],'raw'),('selected_sigmoid','sigmoid'),('selected_isotonic','isotonic')]:
            assert response.json()[key]==pytest.approx(pred[method][first],abs=1e-10)


def test_workshop_never_replaces_formal_serving_run(monkeypatch):
    from reclab import ads
    monkeypatch.setattr(ads,'catalog',lambda:[{'domain':'ads','state':'completed','run_id':'formal','config':{}},{'domain':'ads','state':'completed','run_id':'new-workshop','config':{'workshop':'tree_capacity'}}])
    monkeypatch.setattr(ads,'result',lambda domain,run_id:{'identity':run_id})
    receipt,result=ads.latest()
    assert receipt['run_id']=='formal' and result['identity']=='formal'
