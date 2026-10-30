"""Explicit synthetic second-price impression market, never logged revenue estimation."""
import numpy as np


def market(seed=42, requests=6000, advertisers=3):
    rng = np.random.default_rng(seed)
    # Later requests are more valuable. Every policy sees identical potential outcomes.
    time = np.arange(requests) / requests
    true_p = np.clip(rng.beta(2, 15, (requests, advertisers)) * (0.45 + 1.4*time[:, None]), .001, .7)
    estimate = np.clip(true_p * np.exp(rng.normal(0, .35, true_p.shape)), .001, .95)
    return {'true_p': true_p, 'estimate': estimate,
            'clicks': rng.random(true_p.shape) < true_p,
            'value': np.linspace(1.5, 2.5, advertisers),
            'external': rng.uniform(.025, .12, requests)}


def simulate(data, strategy='value', budget=60., multiplier=1., paced=False):
    if strategy not in ('flat', 'value') or not 0 < budget <= 10000 or not 0 < multiplier <= 10:
        raise ValueError('Invalid market controls')
    p = np.asarray(data['estimate'])
    n, advertisers = p.shape
    spent = np.zeros(advertisers)
    wins = np.zeros(advertisers, dtype=int)
    clicks = np.zeros(advertisers, dtype=int)
    expected_value = 0.
    realized_value = 0.
    trace = []
    for t in range(n):
        bids = multiplier * (p[t] * data['value'] if strategy == 'value' else np.full(advertisers, .12))
        # A bid must be fully affordable. CPC/value are synthetic dollar units;
        # bids and payments here are dollars PER IMPRESSION, not eCPM.
        allowance = budget*(t+1)/n if paced else budget
        eligible = bids <= np.maximum(0., allowance-spent) + 1e-12
        bids = np.where(eligible, bids, -np.inf)
        winner = int(np.argmax(bids))
        if bids[winner] >= data['external'][t]:
            others = np.delete(bids, winner)
            price = max(float(data['external'][t]), float(others.max(initial=-np.inf)))
            assert 0 <= price <= bids[winner] + 1e-12
            spent[winner] += price
            wins[winner] += 1
            clicks[winner] += int(data['clicks'][t, winner])
            expected_value += float(data['true_p'][t, winner]*data['value'][winner])
            realized_value += float(data['clicks'][t, winner]*data['value'][winner])
        assert (spent <= budget+1e-8).all()
        if paced:
            assert (spent <= allowance+1e-8).all()
        if (t+1) % max(1, n//20) == 0 or t == n-1:
            trace.append({'fraction': (t+1)/n, 'spend': spent.tolist()})
    return {'strategy': strategy, 'paced': paced, 'multiplier': multiplier,
            'budget_per_advertiser': budget, 'requests': n,
            'spend': float(spent.sum()), 'simulated_platform_revenue': float(spent.sum()),
            'expected_click_value': expected_value, 'realized_click_value': realized_value,
            'expected_advertiser_surplus': expected_value-float(spent.sum()),
            'wins': int(wins.sum()), 'clicks': int(clicks.sum()),
            'advertisers': [{'spend': float(s), 'wins': int(w), 'clicks': int(c)} for s,w,c in zip(spent,wins,clicks)],
            'budget_ok': bool((spent <= budget+1e-8).all()), 'trace': trace}


def comparisons(seeds=(41, 42, 43)):
    rows = []
    for seed in seeds:
        data = market(seed)
        for name, strategy, paced, multiplier in [('flat', 'flat', False, 1.), ('value', 'value', False, 1.),
                                                  ('value_paced', 'value', True, 1.), ('value_paced_halfbid', 'value', True, .5)]:
            rows.append({'name': name, 'seed': seed, **simulate(data, strategy, paced=paced, multiplier=multiplier)})
    return rows
