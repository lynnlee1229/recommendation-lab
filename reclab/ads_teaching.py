"""Small real-data workshops sharing the ads source, splits, evaluator and registry.

These use select only, do not replace the frozen 200k-run or select on audit.
"""
import time
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from .ads import NUM, CAT, FEATURES, load_data, split_groups, signed_log, encode, vocabulary, verify_sources
from .probability import evaluate
from .experiments import begin, finish
from .paths import ROOT, atomic_json, read_json


def workshop(kind, rows=40000):
    if kind not in ('linear_features','tree_capacity') or not 10000<=rows<=40000:
        raise ValueError('Unknown workshop or excessive row budget')
    config={'workshop':kind,'rows':rows,'seed':42,'evaluation':'select only; audit untouched'}
    folder=begin('ads',config,read_json(ROOT/'docs/ads_sources.json'))
    try:
        verify_sources()
        frame,_=load_data(rows)
        masks,groups=split_groups(frame)
        tr=masks['train'];se=masks['select'];y=frame.label.to_numpy(dtype=int)
        results=[]
        with threadpool_limits(limits=4):
            if kind=='linear_features':
                for categorical in (False,True):
                    numeric=make_pipeline(FunctionTransformer(signed_log,feature_names_out='one-to-one'),SimpleImputer(strategy='median',add_indicator=True),StandardScaler())
                    parts=[('numeric',numeric,NUM)]
                    if categorical:parts.append(('categorical',OneHotEncoder(max_categories=128,handle_unknown='infrequent_if_exist',dtype=np.float32),CAT))
                    model=make_pipeline(ColumnTransformer(parts,sparse_threshold=1.),LogisticRegression(C=1.,max_iter=300))
                    start=time.perf_counter();model.fit(frame.loc[tr,FEATURES],y[tr]);seconds=time.perf_counter()-start
                    p=model.predict_proba(frame.loc[se,FEATURES])[:,1]
                    name='numeric+categorical' if categorical else 'numeric_only'
                    np.savez_compressed(folder/(name+'.npz'),coef=model[-1].coef_,intercept=model[-1].intercept_,select_probability=p,select_row=np.flatnonzero(se))
                    results.append({'method':name,'fit_seconds':seconds,'features':model[-1].coef_.shape[1],'select':evaluate(y[se],p),'iterations':int(model[-1].n_iter_[0])})
            else:
                # Imported only in this explicit training workshop, never in serving.
                import lightgbm as lgb
                vocab=vocabulary(frame.loc[tr],255);x=encode(frame,vocab)
                atomic_json(folder/'vocabulary.json',vocab)
                for leaves in (7,15):
                    model=lgb.LGBMClassifier(n_estimators=80,learning_rate=.05,num_leaves=leaves,min_child_samples=100,reg_lambda=1.,n_jobs=4,verbosity=-1,random_state=42)
                    start=time.perf_counter();model.fit(x[tr],y[tr],categorical_feature=list(range(13,39)));seconds=time.perf_counter()-start
                    p=model.predict_proba(x[se])[:,1]
                    model.booster_.save_model(str(folder/f'leaves{leaves}.txt'))
                    np.savez_compressed(folder/f'leaves{leaves}.npz',select_probability=p,select_row=np.flatnonzero(se))
                    results.append({'method':f'leaves_{leaves}','fit_seconds':seconds,'features':39,'select':evaluate(y[se],p)})
        out={'kind':'workshop','scope':f'Chapter workshop {kind}: first {rows} real Criteo rows; same feature-group split; select-only development results, not independent audit.',
             'rows':results,'shapes':{'raw':[rows,39],'train':int(tr.sum()),'select':int(se.sum())},'changed_variable':kind,
             'limits':'Small prefix, correlated observations and anonymous fields; do not compare to 200k audit or choose production model from this workshop.'}
        finish(folder,out)
        return folder.name,out
    except Exception as exc:
        receipt=read_json(folder/'receipt.json');receipt.update(state='failed',error=type(exc).__name__+': '+str(exc));atomic_json(folder/'receipt.json',receipt);raise
