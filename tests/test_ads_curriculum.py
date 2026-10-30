import itertools
import numpy as np
import pytest
from reclab import ads_curriculum as a


def test_vcg_truthful_grid_and_efficiency():
    # Three independent valuation profiles, all bidders, all unilateral deviations.
    for values in ((7.,6.,1.),(2.,3.,4.),(1.,1.,1.)):
        def utility(bids,i):
            row=next((r for r in a.slots(bids) if r['bidder']==i),None)
            return values[i]*row['click_probability']-row['payment'] if row else 0.
        for i in range(3):
            truthful=utility(values,i)
            assert truthful>=-1e-12
            for bid in np.arange(0,8,.25):
                changed=list(values);changed[i]=bid
                assert utility(changed,i)<=truthful+1e-12
        allocation=a.slots(values)
        welfare=sum(values[r['bidder']]*r['click_probability'] for r in allocation)
        assert welfare==max(values[p[0]]+.4*values[p[1]] for p in itertools.permutations(range(3),2))
    with pytest.raises(ValueError):a.slots(mechanism='unknown')


def test_eligibility_budget_and_empty_paths():
    for query in ('跑鞋','运动 旅行','未知'):
        for region in ('上海','北京','成都'):
            for ad in a.candidates(query,region):
                assert ad['active'] and ad['budget']>0 and region in ad['regions']
    assert a.ledger(query='未知')['spend']==0
    assert a.ledger(budget=1)['spend']==0
    assert a.ledger(budget=2)['spend']>0


def test_feedback_budget_every_request_all_scenarios():
    for pattern,policy,budget in itertools.product(('burst','sparse','drift'),('none','gate','feedback'),(0.,.1,1.,8.)):
        # Zero budget is a valid empty allocation; no divide by zero.
        r=a.pacing(pattern,policy,budget=budget)
        s=[t['spend'] for t in r['trace']]
        assert all(0<=v<=budget+1e-12 for v in s)
        assert all(x<=y for x,y in zip(s,s[1:]))


def test_accounting_selection_and_allocation():
    b=a.billing();assert np.isclose(b['ctr']*b['cvr'],b['ctcvr'])
    c=a.contract_allocation(demand=(1,1));assert c['offline'][0]>c['greedy'][0]
    for d in range(1,8):
        c=a.conversion_delay(d);assert c['observed_cvr']<=c['eventual_cvr']
    r=a.attribution();assert sum(r['last_click'].values())==pytest.approx(sum(r['linear'].values()))
    for r in a.reserves():assert r['utility']>=0 and 0<=r['fill']<=1
    assert a.rtb(timeout=True)['lost_price_lower_bound'] is None


def test_ledger_idempotence_and_shared_budget_interference():
    r=a.deduplicate_charges([('one',.5),('one',.5),('two',.8)],1.)
    assert r['spend']==.5 and len(r['entries'])==1
    assert sum(x['advertiser_debit'] for x in r['entries'])==sum(x['publisher_credit'] for x in r['entries'])
    assert a.shared_budget_interference()['A']==2
    assert a.shared_budget_interference(('B','B','A','A'))['A']==0


def test_course_and_ledger_api():
    from fastapi.testclient import TestClient
    from reclab.service import app
    with TestClient(app,base_url="http://localhost") as client:
        course=client.get('/api/ads/course');assert course.status_code==200
        assert [u['id'] for u in course.json()['units']]==[f'A{i:02}' for i in range(1,19)]
        for budget in (0,1,2):
            response=client.post('/api/ads/workshop/ledger',json={'budget':budget});assert response.status_code==200
            result=response.json();assert result['spend']<=budget and result['spend']==result['publisher_income']
        assert client.post('/api/ads/workshop/ledger',json={'budget':-1}).status_code==422
        assert client.get('/api/code/ads_curriculum').status_code==200
