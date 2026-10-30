import numpy as np
import pytest
from reclab.search_lessons import split_training_queries,fit_ranker,rerank_rows


def test_query_group_split_is_disjoint():
    fit,select=split_training_queries([str(i) for i in range(10)])
    assert len(fit)==8 and len(select)==2 and not set(fit)&set(select)
    with pytest.raises(ValueError):split_training_queries(['same']*10)


def test_ranker_scaler_uses_only_training_rows_and_preserves_candidates():
    rows=[{'qid':'a','ids':[4,7,2], 'x':np.array([[0,0,0,0,0],[3,2,1,2,1],[1,0,1,0,1]],dtype=float),'y':np.array([0,1,0])},
          {'qid':'b','ids':[4,7,2], 'x':np.array([[0,0,0,0,0],[4,3,2,2,1],[1,0,1,0,1]],dtype=float),'y':np.array([0,1,0])}]
    model=fit_ranker(rows)
    assert np.allclose(model[0].mean_,np.concatenate([r['x'] for r in rows]).mean(0))
    pred=rerank_rows(model,rows)
    assert pred['a'][0]==7 and set(pred['a'])==set(rows[0]['ids'])
    single=fit_ranker(rows,columns=(0,)); assert single[0].mean_.shape==(1,)
    assert set(rerank_rows(single,rows,columns=(0,))['b'])==set(rows[1]['ids'])


def test_search_workshop_cannot_replace_serving_run(tmp_path,monkeypatch):
    from reclab import experiments
    from reclab.search import latest
    monkeypatch.setattr(experiments,'BASE',tmp_path)
    official=experiments.begin('search',{},{});experiments.finish(official,{'scope':'official'})
    exercise=experiments.begin('search',{'workshop':'linear_features'},{});experiments.finish(exercise,{'kind':'workshop'})
    receipt,result=latest()
    assert receipt['run_id']==official.name and result['scope']=='official'
    assert len(experiments.catalog())==2
