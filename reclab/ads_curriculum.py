"""Inspectable advertising mechanisms. All fixtures are constructed, never user logs."""
from itertools import product, permutations
import numpy as np


def billing(impressions=10000, clicks=200, conversions=8, spend=80., revenue=240.):
    if not 0 < conversions <= clicks <= impressions or spend <= 0:
        raise ValueError('Require positive, nested event counts and spend')
    ctr, cvr = clicks/impressions, conversions/clicks
    return dict(ctr=ctr, cvr=cvr, ctcvr=conversions/impressions,
                cpm=1000*spend/impressions, cpc=spend/clicks,
                cpa=spend/conversions, roas=revenue/spend,
                advertiser_margin=revenue-spend, publisher_receipt=spend)


def contract_allocation(supply=('flex','onlyA','onlyA'), demand=(1,2)):
    """A accepts both types, B only flex. Reward 10 fulfilled, 0 discarded."""
    eligible=lambda t,j: j==0 or (j==1 and t=='flex')
    def score(assign):
        count=[assign.count(j) for j in (0,1)]
        return 10*sum(min(n,d) for n,d in zip(count,demand))
    feasible=[]
    for assign in product((-1,0,1),repeat=len(supply)):
        if all(j==-1 or eligible(t,j) for t,j in zip(supply,assign)):
            counts=[assign.count(j) for j in (0,1)]
            if all(n<=d for n,d in zip(counts,demand)):
                feasible.append((score(assign),assign))
    best=max(feasible)
    counts=[0,0]; greedy=[]
    for t in supply:
        options=[j for j in (0,1) if eligible(t,j) and counts[j]<demand[j]]
        j=options[0] if options else -1
        greedy.append(j)
        if j>=0: counts[j]+=1
    return {'offline':best,'greedy':(score(tuple(greedy)),tuple(greedy)),
            'offline_shortfall':[max(0,d-best[1].count(j)) for j,d in enumerate(demand)],
            'inventory_counts':{'flex':supply.count('flex'),'onlyA':supply.count('onlyA')}}


def inventory():
    return [dict(id='a',keywords={'跑鞋','运动'},regions={'上海'},active=True,budget=3.,quality=.8,bid=2.,context={'运动'}),
            dict(id='b',keywords={'跑鞋'},regions={'上海','北京'},active=True,budget=2.,quality=.5,bid=2.5,context={'生活'}),
            dict(id='c',keywords={'跑鞋'},regions={'上海'},active=False,budget=3.,quality=.9,bid=4.,context={'运动'}),
            dict(id='d',keywords={'旅行'},regions={'北京'},active=True,budget=0.,quality=.6,bid=3.,context={'旅行'})]


def candidates(query='跑鞋', region='上海', topk=3):
    ads=inventory(); index={}
    for a in ads:
        for key in a['keywords']: index.setdefault(key,set()).add(a['id'])
    tokens=set(query.split()); recalled=set().union(*(index.get(t,set()) for t in tokens))
    eligible=[a for a in ads if a['id'] in recalled and region in a['regions'] and a['active'] and a['budget']>0]
    oracle=[a for a in ads if tokens & a['keywords'] and region in a['regions'] and a['active'] and a['budget']>0]
    assert {a['id'] for a in eligible}=={a['id'] for a in oracle}
    return sorted(eligible,key=lambda a:(-a['quality']*a['bid'],a['id']))[:topk]


def slots(bids=(7.,6.,1.), alpha=(1.,.4), mechanism='vcg'):
    """Homogeneous quality, separable position CTR alpha, bids in currency/click.
    payment is expected currency per request, not CPC. Zero reserve; stable ties.
    """
    bids=np.asarray(bids,float); alpha=np.asarray(alpha,float)
    if mechanism not in ('gfp','gsp','vcg') or len(bids)<len(alpha) or np.any(bids<0) or np.any(np.diff(alpha)>0) or np.any(alpha<=0):
        raise ValueError('Invalid auction')
    order=sorted(range(len(bids)),key=lambda i:(-bids[i],i)); winners=order[:len(alpha)]
    welfare=sum(alpha[k]*bids[i] for k,i in enumerate(winners)); rows=[]
    for k,i in enumerate(winners):
        if mechanism=='gfp': pay=alpha[k]*bids[i]
        elif mechanism=='gsp': pay=alpha[k]*(bids[order[k+1]] if k+1<len(order) else 0.)
        else:
            other=[j for j in order if j!=i][:len(alpha)]
            without=sum(alpha[t]*bids[j] for t,j in enumerate(other))
            pay=without-(welfare-alpha[k]*bids[i])
        rows.append(dict(bidder=i,slot=k,click_probability=float(alpha[k]),payment=float(pay),cpc=float(pay/alpha[k])))
    return rows


def rtb(bid=1.2, competitor=1., reserve=.8, mechanism='second', timeout=False):
    if mechanism not in ('first','second'): raise ValueError('Invalid mechanism')
    win=not timeout and bid>=max(competitor,reserve)
    return dict(win=win, payment=(bid if mechanism=='first' else max(reserve,competitor)) if win else 0.,
                observed_market_price=competitor if win else None,
                lost_price_lower_bound=bid if not win and not timeout and bid>=reserve else None,
                stages=['SSP request','ADX eligibility','DSP response','auction','win/loss notice','feedback'])


def targeting(expand=False):
    # Context is deliberately provided; no device ID, inferred sensitive label or join.
    contexts=['运动','生活','旅行','运动','科技']
    allowed={'运动','生活'} if expand else {'运动'}
    chosen=[i for i,c in enumerate(contexts) if c in allowed]
    relevant={0,3}
    return dict(selected=chosen,coverage=len(chosen)/len(contexts),precision=len(set(chosen)&relevant)/max(1,len(chosen)),fields=['page_context'])


def conversion_delay(cutoff=2):
    # Each tuple is (clicked, eventual_conversion, delay_days, stratum).
    logs=[(1,1,1,'high'),(1,1,5,'high'),(1,0,0,'high'),(1,0,0,'low')]+[(0,0,0,'low')]*6
    clicks=sum(x[0] for x in logs); eventual=sum(x[1] for x in logs)
    observed=sum(c and y and d<=cutoff for c,y,d,_ in logs)
    return dict(ctr=clicks/len(logs),eventual_cvr=eventual/clicks,eventual_ctcvr=eventual/len(logs),
                observed_cvr=observed/clicks,unmatured_positive=eventual-observed,
                clicked_high_share=sum(c and s=='high' for c,_,_,s in logs)/clicks,
                exposure_high_share=sum(s=='high' for *_,s in logs)/len(logs))


def constrained_bidding(budget=4., target_roas=2.):
    # Known constructed pCTCVR and conversion values; prices independent of bid.
    p=np.array([.01,.08,.02,.12,.03,.1]); value=p*50
    price=np.array([.4,1.5,.7,2.8,1.,2.])
    out=[]
    for name, bids in [('value',value),('roas_cap',value/target_roas)]:
        spent=reward=0.; decisions=[]
        for v,c,b in zip(value,price,bids):
            win=b>=c and spent+c<=budget+1e-12
            if win: spent+=c;reward+=v
            assert spent<=budget+1e-12
            decisions.append(bool(win))
        out.append(dict(policy=name,spend=spent,expected_value=reward,expected_roas=reward/spent if spent else None,win=decisions))
    return out


def pacing(pattern='burst', policy='feedback', budget=8., gain=.8):
    if pattern not in ('burst','sparse','drift') or policy not in ('none','gate','feedback'): raise ValueError('Invalid scenario')
    prices={'burst':[.2]*20+[.8]*20,'sparse':[1.5]*30+[.3]*10,'drift':[.3]*20+[1.1]*20}[pattern]
    spend=0.; multiplier=1.; trace=[]; value=0.
    for t,price in enumerate(prices):
        target=budget*(t+1)/len(prices)
        if policy=='feedback' and budget>0: multiplier=float(np.clip(multiplier+gain*(target-spend)/budget,.05,3.))
        bid=(.7*multiplier if policy=='feedback' else .7)
        allowed=policy!='gate' or spend+price<=target
        win=allowed and bid>=price and spend+price<=budget+1e-12
        if win: spend+=price;value+=1.
        trace.append(dict(t=t+1,target=target,spend=spend,multiplier=multiplier,won=bool(win)))
        assert -1e-12<=spend<=budget+1e-12
    return dict(pattern=pattern,policy=policy,spend=spend,value=value,trace=trace)


def attribution():
    paths=[(['search','display'],100.),(['display'],60.),(['search','search','display'],90.)]
    last={};linear={}
    for path,revenue in paths:
        last[path[-1]]=last.get(path[-1],0)+revenue
        for channel in path: linear[channel]=linear.get(channel,0)+revenue/len(path)
    assert abs(sum(last.values())-sum(linear.values()))<1e-9
    rng=np.random.default_rng(42); n=10000
    assignment=rng.integers(0,2,n); latent=rng.random(n)
    y0=latent<.10; y1=latent<.13; observed=np.where(assignment,y1,y0)
    return dict(last_click=last,linear=linear,randomized_difference=float(observed[assignment==1].mean()-observed[assignment==0].mean()),
                finite_population_effect=float((y1.astype(int)-y0).mean()),n=n,assignment_probability=.5)


def reserves():
    bids=np.random.default_rng(42).uniform(0,2,(2000,3)); order=np.sort(bids,axis=1)
    rows=[]
    for reserve in (0.,.5,1.,1.5,2.):
        win=order[:,-1]>=reserve; pay=np.where(win,np.maximum(reserve,order[:,-2]),0)
        utility=np.where(win,order[:,-1]-pay,0)
        assert np.all(utility>=-1e-12)
        rows.append(dict(reserve=reserve,fill=float(win.mean()),revenue=float(pay.mean()),utility=float(utility.mean())))
    return rows


def experiment_bias():
    # Group A/B have identical stratum rates; unequal mix creates aggregate difference.
    counts={'A':[(90,18),(10,1)],'B':[(10,2),(90,9)]}
    aggregate={k:sum(y for n,y in cells)/sum(n for n,y in cells) for k,cells in counts.items()}
    standardized={k:sum(y/n for n,y in cells)/2 for k,cells in counts.items()}
    # A rule is only a triage signal; legitimate burst is a false positive.
    traffic=[('bot',50,True),('human',1,False),('campaign_burst',40,False)]
    flags=[name for name,rate,_ in traffic if rate>20]
    return dict(aggregate=aggregate,standardized=standardized,flagged=flags,true_positive=1,false_positive=1,
                interference='Shared campaign budgets couple request buckets; use campaign/market randomization or model spillovers.')


def ledger(query='跑鞋',region='上海',budget=1.):
    pool=candidates(query,region); rows=[]; spent=0.
    if not pool: return dict(candidates=[],ledger=[],spend=0.,publisher_income=0.)
    winner=pool[0]; next_score=pool[1]['bid']*pool[1]['quality'] if len(pool)>1 else 0.
    cpc=next_score/winner['quality']
    # One explicitly constructed click; affordable eligibility is checked again at charging.
    click=True;payment=cpc if click and cpc<=budget else 0.
    if payment: spent+=payment; rows.append(dict(request='constructed-1',ad=winner['id'],event='click',payment=payment,receipt=payment))
    assert spent<=budget and abs(sum(r['payment'] for r in rows)-sum(r['receipt'] for r in rows))<1e-12
    return dict(candidates=[a['id'] for a in pool],ledger=rows,spend=spent,publisher_income=spent,
                probability_source='constructed quality, NOT Criteo production CTR',served=bool(rows))


def ctr_interaction_workshop(rows=10000, epochs=8):
    """Real sampled Criteo categorical LR versus FM; shared recommendation FM operator."""
    import torch
    import time
    from torch import nn
    from .learning.networks import fm_interaction
    from .ads import load_data, split_groups, vocabulary, encode, verify_sources, CAT
    from .probability import evaluate
    from .experiments import begin, finish
    from .paths import ROOT, read_json, atomic_json
    if rows!=10000 or epochs!=8: raise ValueError('Frozen teaching budget: 10k rows, 8 epochs')
    folder=begin('ads',{'workshop':'categorical_fm','rows':rows,'epochs':epochs,'seed':42,'evaluation':'select only'},read_json(ROOT/'docs/ads_sources.json'))
    try:
        verify_sources();frame,_=load_data(rows);masks,_=split_groups(frame)
        vocab=vocabulary(frame.loc[masks['train']],64)
        x=encode(frame,vocab)[:,13:].astype(np.int64)+np.arange(len(CAT))*65
        y=frame.label.to_numpy(dtype=np.float32)
        xt=torch.tensor(x);yt=torch.tensor(y)
        tr=torch.tensor(np.flatnonzero(masks['train'])); se=torch.tensor(np.flatnonzero(masks['select']))
        class Response(nn.Module):
            def __init__(self, interaction):
                super().__init__();self.linear=nn.Embedding(26*65,1);self.bias=nn.Parameter(torch.zeros(()));self.interaction=interaction
                nn.init.zeros_(self.linear.weight)
                if interaction: self.factors=nn.Embedding(26*65,4);nn.init.normal_(self.factors.weight,std=.01)
            def forward(self,x):
                logit=self.linear(x).sum((1,2))+self.bias
                return logit+fm_interaction(self.factors(x)) if self.interaction else logit
        results=[]; old_threads=torch.get_num_threads();torch.set_num_threads(2)
        try:
            for interaction in (False,True):
                started=time.perf_counter()
                torch.manual_seed(42);model=Response(interaction);optimizer=torch.optim.Adam(model.parameters(),lr=.02,weight_decay=1e-5)
                generator=torch.Generator().manual_seed(42);curve=[]
                for _ in range(epochs):
                    order=tr[torch.randperm(len(tr),generator=generator)]
                    for batch in order.split(256):
                        optimizer.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(model(xt[batch]),yt[batch]);loss.backward();optimizer.step()
                    with torch.no_grad():curve.append(float(nn.functional.binary_cross_entropy_with_logits(model(xt[tr]),yt[tr])))
                with torch.no_grad():p=model(xt[se]).sigmoid().numpy()
                name='fm' if interaction else 'linear'
                torch.save(model.state_dict(),folder/(name+'.pt'))
                np.savez_compressed(folder/(name+'.npz'),select_probability=p,select_row=se.numpy(),label=y[se.numpy()])
                results.append(dict(method=name,features=26,fit_seconds=time.perf_counter()-started,parameters=sum(p.numel() for p in model.parameters()),train_loss=curve,select=evaluate(y[se.numpy()],p)))
        finally: torch.set_num_threads(old_threads)
        atomic_json(folder/'vocabulary.json',vocab)
        result=dict(kind='workshop',rows=results,shapes={'raw':[rows,39],'categorical':[rows,26],'train':len(tr),'select':len(se)},
                    changed_variable='FM pair interactions; extra factor parameters explicitly counted; same rows/order/epochs/optimizer',
                    scope='Real sampled Criteo categorical-only select diagnostic, not independent audit or population CTR.',
                    limits='Capped train vocab, small prefix, one seed, numeric features omitted equally; no production selection or revenue claim.')
        finish(folder,result);return folder.name,result
    except Exception as exc:
        receipt=read_json(folder/'receipt.json');receipt.update(state='failed',error=str(exc));atomic_json(folder/'receipt.json',receipt);raise


def deduplicate_charges(events, budget):
    """Single-process mechanism; NOT a concurrent transactional billing service."""
    seen=set();ledger=[];spent=0.
    for event_id,amount in events:
        if amount<0: raise ValueError('Use a separate refund event type')
        if event_id in seen: continue
        seen.add(event_id)
        if spent+amount<=budget:
            spent+=amount;ledger.append(dict(event=event_id,advertiser_debit=amount,publisher_credit=amount))
        assert spent<=budget
    return dict(spend=spent,entries=ledger,unique_events=len(seen))


def shared_budget_interference(order=('A','A','B','B'),budget=2):
    spent=0;wins={'A':0,'B':0}
    for bucket in order:
        if spent+1<=budget:spent+=1;wins[bucket]+=1
    return wins
