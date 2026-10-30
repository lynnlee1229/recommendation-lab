"""Real sampled Criteo CTR adapter using the shared lab registry and probability metrics."""
from dataclasses import dataclass, asdict
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
from .paths import ROOT, ARTIFACTS, atomic_json, read_json
from .experiments import begin, finish, catalog, result
from .probability import evaluate
from .auction import comparisons

NUM = [f'integer_feature_{i}' for i in range(1,14)]
CAT = [f'categorical_feature_{i}' for i in range(1,27)]
FEATURES = NUM + CAT


@dataclass(frozen=True)
class Config:
    rows: int = 200000
    seed: int = 42
    trees: tuple = (80, 160)
    category_limit: int = 255

    def __post_init__(self):
        if not 10000 <= self.rows <= 200000 or not self.trees or any(t < 1 or t > 300 for t in self.trees):
            raise ValueError('Invalid training budget')
        if not 2 <= self.category_limit <= 255:
            raise ValueError('Invalid category budget')


def verify_sources(download=False):
    from download_public_assets import fetch, sha
    spec = read_json(ROOT/'docs/ads_sources.json')
    for row in spec['assets']:
        if download:
            fetch(row)
        path = ROOT/row['destination']
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Frozen ads input mismatch')
    return spec


def load_data(rows=200000):
    file = pq.ParquetFile(ROOT/'data/ads/criteo-day15-part00079.parquet')
    batches, count = [], 0
    for batch in file.iter_batches(batch_size=10000):
        frame = batch.to_pandas()
        batches.append(frame.iloc[:rows-count])
        count += len(batches[-1])
        if count == rows:
            break
    frame = pd.concat(batches, ignore_index=True)
    if len(frame) != rows or not frame.label.isin([0,1]).all():
        raise ValueError('Invalid Criteo rows')
    frame[NUM] = frame[NUM].astype(float)
    frame[CAT] = frame[CAT].fillna('__MISSING__').astype(str)
    return frame, file.metadata.num_rows


def split_groups(frame):
    # Feature-only grouping prevents identical feature vectors crossing splits.
    # No timestamps/user IDs are disclosed: this is NOT temporal/user holdout.
    group = pd.util.hash_pandas_object(frame[FEATURES], index=False).to_numpy(dtype='uint64')
    bucket = group % 10000
    masks = {'train': bucket < 5500, 'select': (bucket >= 5500)&(bucket < 7000),
             'calibration': (bucket >= 7000)&(bucket < 8500), 'audit': bucket >= 8500}
    return masks, group


def signed_log(x):
    return np.sign(x)*np.log1p(np.abs(x))


def vocabulary(frame, limit):
    return {name: frame[name].value_counts().head(limit).index.tolist() for name in CAT}


def encode(frame, vocab):
    values = [frame[NUM].to_numpy(dtype=np.float32)]
    for name in CAT:
        mapping = {v: i+1 for i,v in enumerate(vocab[name])}
        values.append(frame[name].map(mapping).fillna(0).to_numpy(dtype=np.float32)[:,None])
    return np.concatenate(values, axis=1)


def logit(p):
    p = np.clip(p, 1e-7, 1-1e-7)
    return np.log(p/(1-p))


def paired_loss_interval(y, raw, calibrated, seed=42):
    # Paired IID row bootstrap; repeated vectors/users can violate independence.
    def loss(p):
        p=np.clip(p,1e-7,1-1e-7)
        return -(y*np.log(p)+(1-y)*np.log1p(-p))
    delta = loss(calibrated)-loss(raw)
    rng=np.random.default_rng(seed)
    means=[float(delta[rng.integers(len(y),size=len(y))].mean()) for _ in range(300)]
    return {'calibrated_minus_raw_logloss':float(delta.mean()), 'iid_row_bootstrap_95':np.quantile(means,[.025,.975]).tolist(),
            'replicates':300, 'limitation':'conditional on fitted models; repeated/related rows and one shard limit independence and generalization'}


def train(folder, config):
    import lightgbm as lgb
    start=time.perf_counter()
    frame, source_rows=load_data(config.rows)
    masks, groups=split_groups(frame)
    y=frame.label.to_numpy(dtype=int)
    for name,mask in masks.items():
        if mask.sum()<100 or len(np.unique(y[mask])) != 2:
            raise ValueError('Split too small or single-class: '+name)
    train_mask=masks['train']; select=masks['select']; cal=masks['calibration']; audit=masks['audit']
    rows=[]
    preds={}
    def report(name, probability, seconds, **extra):
        preds[name]=np.asarray(probability)
        row={'method':name,'fit_seconds':seconds,'select':evaluate(y[select], probability[select]),
             'audit':evaluate(y[audit],probability[audit]), **extra}
        rows.append(row)
        return row
    prior=np.full(len(y), y[train_mask].mean())
    report('train_prior',prior,0.)
    numeric=make_pipeline(FunctionTransformer(signed_log,feature_names_out="one-to-one"),SimpleImputer(strategy='median',add_indicator=True),StandardScaler())
    transforms=ColumnTransformer([('numeric',numeric,NUM),('categorical',OneHotEncoder(handle_unknown='infrequent_if_exist',max_categories=128,dtype=np.float32),CAT)],sparse_threshold=1.)
    lr=make_pipeline(transforms,LogisticRegression(C=1.,max_iter=300,solver='lbfgs',random_state=config.seed))
    tick=time.perf_counter()
    lr.fit(frame.loc[train_mask,FEATURES],y[train_mask])
    lr_seconds=time.perf_counter()-tick
    report('linear_onehot',lr.predict_proba(frame[FEATURES])[:,1],lr_seconds,iterations=int(lr[-1].n_iter_[0]))
    names=lr[0].get_feature_names_out()
    coef=lr[-1].coef_[0]
    # Local full coefficient table is inspectable; no hashed IDs in committed aggregate.
    pd.DataFrame({'feature':names,'coefficient':coef}).to_csv(folder/'linear_coefficients.csv',index=False)
    vocab=vocabulary(frame.loc[train_mask],config.category_limit)
    atomic_json(folder/'vocabulary.json',vocab)
    x=encode(frame,vocab)
    models=[]
    selection=[]
    for trees in config.trees:
        tick=time.perf_counter()
        model=lgb.LGBMClassifier(n_estimators=trees,learning_rate=.05,num_leaves=15,max_depth=-1,
                                min_child_samples=100,colsample_bytree=1.,subsample=1.,reg_lambda=1.,
                                n_jobs=4,random_state=config.seed,verbosity=-1)
        model.fit(x[train_mask],y[train_mask],categorical_feature=list(range(13,39)))
        seconds=time.perf_counter()-tick
        p=model.predict_proba(x)[:,1]
        method=f'gbdt_{trees}'
        report(method,p,seconds,trees=trees)
        selection.append((evaluate(y[select],p[select])['logloss'],trees,method))
        models.append(model)
    chosen=min(selection)
    selected=chosen[2]
    model=models[list(config.trees).index(chosen[1])]
    model.booster_.save_model(str(folder/'ctr_model.txt'))
    atomic_json(folder/'ctr_trees.json',model.booster_.dump_model())
    p=preds[selected]
    tick=time.perf_counter()
    sigmoid=LogisticRegression(C=1e6,solver='lbfgs',max_iter=200)
    sigmoid.fit(logit(p[cal])[:,None],y[cal])
    sigmoid_seconds=time.perf_counter()-tick
    tick=time.perf_counter()
    isotonic=IsotonicRegression(out_of_bounds='clip',y_min=0.,y_max=1.).fit(p[cal],y[cal])
    isotonic_seconds=time.perf_counter()-tick
    # These rows' select metrics are diagnostic only: base tree count was already chosen there.
    sp=sigmoid.predict_proba(logit(p)[:,None])[:,1]
    ip=isotonic.predict(p)
    report('selected_sigmoid',sp,sigmoid_seconds,base=selected)
    report('selected_isotonic',ip,isotonic_seconds,base=selected)
    calibration={'base':selected,'sigmoid_coef':float(sigmoid.coef_[0,0]),'sigmoid_intercept':float(sigmoid.intercept_[0]),
                 'isotonic_x':isotonic.X_thresholds_.tolist(),'isotonic_y':isotonic.y_thresholds_.tolist()}
    atomic_json(folder/'calibration.json',calibration)
    missing=frame[NUM].isna().sum(axis=1).to_numpy()
    threshold=float(np.median(missing[train_mask]))
    unseen=np.column_stack([~frame[name].isin(set(frame.loc[train_mask,name])).to_numpy() for name in CAT]).any(axis=1)
    segments=[]
    for name,mask in [('low_numeric_missing',missing<=threshold),('high_numeric_missing',missing>threshold),('any_unseen_category',unseen),('all_categories_seen',~unseen)]:
        mask=mask&audit
        if mask.any():
            segments.append({'segment':name,'raw':evaluate(y[mask],p[mask]),'sigmoid':evaluate(y[mask],sp[mask])})
    audit_ids=np.flatnonzero(audit)
    losses=-(y[audit]*np.log(np.clip(sp[audit],1e-7,1-1e-7))+(1-y[audit])*np.log(np.clip(1-sp[audit],1e-7,1-1e-7)))
    worst=audit_ids[np.argsort(-losses,kind='stable')[:8]]
    failures=[{'row':int(i),'label':int(y[i]),'raw':float(p[i]),'sigmoid':float(sp[i]),'numeric_missing':int(missing[i]),'any_unseen_category':bool(unseen[i])} for i in worst]
    split_metrics={name:{'n':int(mask.sum()),'positives':int(y[mask].sum()),'click_rate':float(y[mask].mean()),'unique_feature_groups':len(np.unique(groups[mask]))} for name,mask in masks.items()}
    # All row-level outputs stay ignored. Labels are actual clicked/not-clicked exposures.
    np.savez_compressed(folder/'predictions.npz',row=np.arange(len(y)),label=y,group=groups,split=np.select(list(masks.values()),list(range(4))),**preds)
    frame.iloc[worst].to_parquet(folder/'failure_rows.parquet',index=False)
    frame[FEATURES].iloc[audit_ids[:5]].to_json(folder/'example_inputs.json',orient='records')
    tick=time.perf_counter()
    simulations=comparisons()
    simulation_seconds=time.perf_counter()-tick
    return {'scope':f'Criteo official sampled click logs: first {len(frame):,} rows of one {source_rows:,}-row parquet shard (2015-02-15). Real CTR task on a subsampled distribution. Auction is a separate synthetic market.',
            'rows':rows,'selected_base':selected,'splits':split_metrics,'source_shard_rows':source_rows,
            'feature_duplicate_rows':len(frame)-len(np.unique(groups)),
            'conflicting_label_feature_groups':int(pd.DataFrame({'g':groups,'y':y}).groupby('g').y.nunique().gt(1).sum()),
            'segments':segments,'missing_threshold_train':threshold,'failures':failures,
            'calibration':calibration,'paired_sigmoid':paired_loss_interval(y[audit],p[audit],sp[audit],config.seed),
            'simulations':simulations,'simulation_seconds':simulation_seconds,'total_seconds':time.perf_counter()-start,
            'protocol':{'split':'feature-only pandas uint64 hash modulo 10000: train <5500; select <7000; calibration <8500; audit >=8500; no feature group overlap; NOT temporal or user split',
                        'selection':'Only select logloss chooses between 80/160 tree GBDTs; train-only encoders and vocab. Calibrators fitted on separate calibration split; all variants reported on audit; no audit selection.',
                        'features':'13 signed-log numeric + imputation/scaling and 26 capped onehot for LR; raw numeric + train top255 category codes for GBDT (other=0). No semantic feature/user IDs disclosed.',
                        'timing':'CPU wall time fit and end-to-end including preprocessing, probability evaluation, disk artifacts and simulation; excludes download; up to 4 threads.',
                        'limits':'One prefix of one shard/day; unknown parquet chronology; no user identities; repeated related observations possible. Positive/negative sampling rates undisclosed: calibrated sample probabilities are NOT population CTR. ECE uses 10 equal-width bins, not a proper scoring rule.',
                        'market':'Independent synthetic 3-advertiser, 6000-request, second-price PER-IMPRESSION auction with external competing bid; 60 currency units each; no external win => no sale. No real logs are used to estimate counterfactual revenue. Seed-paired 41/42/43; same potentials across policies; not tuned.'}}


def run(config=Config(),download=False):
    provenance=read_json(ROOT/'docs/ads_sources.json')
    folder=begin('ads',asdict(config),provenance)
    try:
        verify_sources(download)
        with threadpool_limits(limits=4):
            output=train(folder,config)
        finish(folder,output)
        print('COMPLETED',folder.name,'seconds',round(output['total_seconds'],2),flush=True)
        return folder,output
    except Exception as exc:
        receipt=read_json(folder/'receipt.json')
        receipt.update(state='failed',error=type(exc).__name__+': '+str(exc))
        atomic_json(folder/'receipt.json',receipt)
        raise


def latest():
    rows=[r for r in catalog() if r['domain']=='ads' and r['state']=='completed' and not r.get('config',{}).get('workshop') and not r.get('annotation',{}).get('superseded')]
    if not rows:
        raise FileNotFoundError('Run python -m reclab.ads --download')
    receipt=rows[-1]
    return receipt,result('ads',receipt['run_id'])


class Predictor:
    def __init__(self,run_id):
        self.run_id=run_id
        folder=ARTIFACTS/'experiments/ads'/run_id
        self.vocab=read_json(folder/'vocabulary.json')
        self.cal=read_json(folder/'calibration.json')
        # Native OpenMP model loading can crash in the PyTorch-serving process on
        # this Mac. Interpret the exact exported tree structure instead, without
        # loading a second native inference runtime or altering system settings.
        self.trees=read_json(folder/'ctr_trees.json')
        if not self.trees['objective'].startswith('binary sigmoid:1') or self.trees['num_tree_per_iteration'] != 1 or self.trees['average_output']:
            raise ValueError('Unsupported exported tree objective')
        def prepare(node):
            if 'leaf_value' in node: return
            if node['decision_type']=='==':
                node['_category_set']={int(v) for v in node['threshold'].split('||')}
            prepare(node['left_child']);prepare(node['right_child'])
        for tree in self.trees['tree_info']: prepare(tree['tree_structure'])

    def raw_probability(self,x):
        score=0.
        for tree in self.trees['tree_info']:
            node=tree['tree_structure']
            while 'leaf_value' not in node:
                value=float(x[node['split_feature']])
                if node['decision_type']=='==':
                    left=np.isfinite(value) and value>=0 and int(value) in node['_category_set']
                else:
                    if np.isnan(value) and node['missing_type']!='NaN': value=0.
                    missing=(node['missing_type']=='NaN' and np.isnan(value)) or (node['missing_type']=='Zero' and abs(value)<=1e-35)
                    left=node['default_left'] if missing else value<=float(node['threshold'])
                node=node['left_child'] if left else node['right_child']
            score+=node['leaf_value']
        return float(1/(1+np.exp(-np.clip(score,-700,700))))

    def predict(self,numeric,categorical):
        frame=pd.DataFrame([dict(zip(NUM,numeric))|dict(zip(CAT,[v if v is not None else '__MISSING__' for v in categorical]))])
        p=self.raw_probability(encode(frame,self.vocab)[0])
        z=np.clip(self.cal['sigmoid_coef']*logit(p)+self.cal['sigmoid_intercept'],-700,700)
        return {'run_id':self.run_id,'base':self.cal['base'],'raw':p,'sigmoid':float(1/(1+np.exp(-z))),
                'isotonic':float(np.interp(p,self.cal['isotonic_x'],self.cal['isotonic_y'])),
                'scope':'Probabilities on the sampled Criteo distribution only; unknown feature semantics; not population CTR or auction revenue.'}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    run(download=args.download)
