import json
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from threadpoolctl import threadpool_limits
from .config import ARTIFACTS
from .data import fingerprint,trajectory_groups
from .features import fusion_matrix, mask_features
from .llm_settings import get_settings

EXPERTS = ('numeric','image','text','maintenance')
SEED = 20261013

def raw_estimator(seed=SEED, quantile=None):
    kw = dict(max_iter=180,max_leaf_nodes=15,min_samples_leaf=25,
              learning_rate=.065,l2_regularization=4.,early_stopping=False,random_state=seed)
    if quantile is not None:
        kw.update(loss='quantile',quantile=quantile)
    return HistGradientBoostingRegressor(**kw)

class SafeHistogramRegressor(RegressorMixin, BaseEstimator):
    """Drop training columns with no observations; preserve other NaNs for native routing."""
    def __init__(self,seed=SEED,quantile=None):
        self.seed=seed
        self.quantile=quantile

    def fit(self,X,y,sample_weight=None):
        X=np.asarray(X,float)
        self.columns_=np.flatnonzero(np.isfinite(X).any(axis=0))
        self.n_features_in_=X.shape[1]
        self.model_=raw_estimator(self.seed,self.quantile)
        selected=X[:,self.columns_] if len(self.columns_) else np.zeros((len(X),1))
        self.model_.fit(selected,y,sample_weight=sample_weight)
        return self

    def predict(self,X):
        X=np.asarray(X,float)
        if X.shape[1]!=self.n_features_in_:
            raise ValueError('预测特征维度与训练不一致')
        selected=X[:,self.columns_] if len(self.columns_) else np.zeros((len(X),1))
        return self.model_.predict(selected)

def estimator(seed=SEED, quantile=None):
    return SafeHistogramRegressor(seed=seed,quantile=quantile)

def split_records(records, mode='device'):
    devices = np.array([r.device for r in records])
    mapping=trajectory_groups(records)
    clusters=np.array([mapping[r.device] for r in records])
    unique = np.unique(clusters)
    if len(unique)<10:
        raise ValueError('完整评估至少需要 10 台设备')
    if mode=='device':
        shuffled=np.random.default_rng(SEED).permutation(unique)
        ntest=max(2,int(len(unique)*.2)); ncal=max(2,int(len(unique)*.1))
        train=np.isin(clusters,shuffled[:len(unique)-ntest-ncal])
        cal=np.isin(clusters,shuffled[len(unique)-ntest-ncal:len(unique)-ntest])
        test=np.isin(clusters,shuffled[-ntest:])
        info={'mode':'unseen_trajectory_group','train_devices':sorted(np.unique(devices[train]).tolist()),
              'calibration_devices':sorted(np.unique(devices[cal]).tolist()),
              'test_devices':sorted(np.unique(devices[test]).tolist()),'seed':SEED}
    elif mode=='temporal':
        train=np.zeros(len(records),bool);cal=train.copy();test=train.copy();cuts={}
        for device in np.unique(devices):
            ix=np.where(devices==device)[0]
            events=sorted(set(records[i].next_inspection for i in ix))
            c1=events[max(0,int(len(events)*.65)-1)]
            c2=events[max(1,int(len(events)*.8)-1)]
            cuts[str(device)]=[c1,c2]
            for i in ix:
                r=records[i]
                train[i]=r.next_inspection<=c1
                cal[i]=r.time>=c1 and r.next_inspection<=c2
                test[i]=r.time>=c2
        info={'mode':'future_events','cutoffs_by_device':cuts,
              'purged_boundary_records':int((~(train|cal|test)).sum()),'seed':SEED}
    else:
        raise ValueError('未知划分方式')
    if np.any(train&cal) or np.any(train&test) or np.any(cal&test):
        raise AssertionError('划分重叠')
    groups=[set((r.device,r.next_inspection) for r,m in zip(records,mask) if m) for mask in (train,cal,test)]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)):
        raise AssertionError('目标巡检事件跨越分区')
    info['record_counts']={'train':int(train.sum()),'calibration':int(cal.sum()),'test':int(test.sum())}
    info['trajectory_group_count']=len(unique)
    info['trajectory_group_by_device']={str(k):v for k,v in mapping.items()}
    info['duplicate_device_groups']=[[d for d in sorted(mapping) if mapping[d]==g]
                                     for g in unique if sum(v==g for v in mapping.values())>1]
    info['trajectory_group_counts']={k:int(len(np.unique(clusters[m])))
                                     for k,m in [('train',train),('calibration',cal),('test',test)]}
    if mode=='device':
        sets=[set(clusters[m]) for m in (train,cal,test)]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    return train,cal,test,info

def event_weights(records):
    mapping=trajectory_groups(records)
    counts={}
    for r in records:
        key=(mapping[r.device],r.next_inspection)
        counts[key]=counts.get(key,0)+1
    w=np.array([1/counts[(mapping[r.device],r.next_inspection)] for r in records])
    return w/w.mean()

def expert_predictions(models, features):
    p=np.column_stack([models[k].predict(features[k]) for k in EXPERTS])
    p[features['quality'][:,0]==0,1]=np.nan
    p[features['quality'][:,2]==0,2]=np.nan
    return p

def subset(features, ix):
    return {k:v[ix] for k,v in features.items() if isinstance(v,np.ndarray)}

def augment_fusion(p, features, y, w):
    xs=[];ys=[];ws=[]
    for hide_image,hide_text,factor in [(False,False,.7),(True,False,.1),(False,True,.1),(True,True,.1)]:
        f=mask_features(features,hide_image,hide_text);pp=p.copy()
        if hide_image:pp[:,1]=np.nan
        if hide_text:pp[:,2]=np.nan
        xs.append(fusion_matrix(pp,f));ys.append(y);ws.append(w*factor)
    return np.vstack(xs),np.concatenate(ys),np.concatenate(ws)

def interval(model, matrix):
    a=model['lower'].predict(matrix);b=model['upper'].predict(matrix)
    lo=np.minimum(a,b)-model.get('interval_adjustment',0.)
    hi=np.maximum(a,b)+model.get('interval_adjustment',0.)
    return lo,hi

def metrics(y,p,devices,events,lo=None,hi=None):
    groups=np.unique(devices)
    by_device={str(d):float(mean_absolute_error(y[devices==d],p[devices==d])) for d in groups}
    out={'mae_adu':float(mean_absolute_error(y,p)),
         'rmse_adu':float(np.sqrt(mean_squared_error(y,p))), 'r2':float(r2_score(y,p)),
         'device_macro_mae_adu':float(np.mean(list(by_device.values()))),
         'by_device_mae_adu':by_device,'records':int(len(y))}
    if lo is not None:
        cover=(y>=lo)&(y<=hi)
        event_coverage=[bool(np.all(cover[events==e])) for e in np.unique(events)]
        out.update(interval_coverage=float(cover.mean()),interval_width_adu=float(np.mean(hi-lo)),
                   event_simultaneous_coverage=float(np.mean(event_coverage)))
    # Device cluster bootstrap, rather than pretending individual rows are independent.
    cluster=np.array([e.split(':')[0] for e in events])
    cg=np.unique(cluster)
    by_cluster={g:float(mean_absolute_error(y[cluster==g],p[cluster==g])) for g in cg}
    out['trajectory_macro_mae_adu']=float(np.mean(list(by_cluster.values())))
    out['trajectory_groups']=len(cg)
    rng=np.random.default_rng(SEED);draws=[]
    for _ in range(500):
        sample=rng.choice(cg,len(cg),replace=True)
        draws.append(float(np.mean([by_cluster[g] for g in sample])))
    out['trajectory_macro_mae_bootstrap_95']=np.quantile(draws,[.025,.975]).tolist()
    return out

def train(records,features,mode='device',progress=print):
    started=time.monotonic()
    train_mask,cal_mask,test_mask,split=split_records(records,mode)
    ix=np.flatnonzero(train_mask);cal_ix=np.flatnonzero(cal_mask);test_ix=np.flatnonzero(test_mask)
    y=np.array([r.target for r in records],float)
    devices=np.array([r.device for r in records])
    mapping=trajectory_groups(records)
    clusters=np.array([mapping[r.device] for r in records])
    events=np.array([f'{mapping[r.device]}:{r.next_inspection:g}' for r in records])
    weights=event_weights([records[i] for i in ix])
    oof=np.full((len(ix),len(EXPERTS)),np.nan)
    cv=GroupKFold(n_splits=4)
    fold_log=[]
    with threadpool_limits(limits=4):
        for fold,(fit,val) in enumerate(cv.split(ix,groups=clusters[ix])):
            fold_models={}
            for k in EXPERTS:
                progress(f'OOF fold {fold+1}/4: {k}')
                m=estimator(seed=SEED+fold)
                m.fit(features[k][ix[fit]],y[ix[fit]],sample_weight=weights[fit])
                fold_models[k]=m
            oof[val]=expert_predictions(fold_models,subset(features,ix[val]))
            fold_log.append({'fold':fold+1,'fit_devices':sorted(np.unique(devices[ix[fit]]).tolist()),
                             'validation_devices':sorted(np.unique(devices[ix[val]]).tolist()),
                             'fit_trajectory_groups':sorted(np.unique(clusters[ix[fit]]).tolist()),
                             'validation_trajectory_groups':sorted(np.unique(clusters[ix[val]]).tolist())})
        xmeta,ymeta,wmeta=augment_fusion(oof,subset(features,ix),y[ix],weights)
        model={'experts':{},'fusion':estimator(),'lower':estimator(quantile=.05),
               'upper':estimator(quantile=.95),'split':split,'oof_folds':fold_log,
               'fingerprint':fingerprint(records),'feature_version':'v1','interval_adjustment':0.,
               'semantic_namespace':get_settings().namespace}
        for k in ('fusion','lower','upper'):
            progress(f'Training fusion: {k}')
            model[k].fit(xmeta,ymeta,sample_weight=wmeta)
        for k in EXPERTS:
            progress(f'Training final expert: {k}')
            model['experts'][k]=estimator().fit(features[k][ix],y[ix],sample_weight=weights)
        cal_f=subset(features,cal_ix)
        cal_x=fusion_matrix(expert_predictions(model['experts'],cal_f),cal_f)
        lo,hi=interval(model,cal_x)
        scores=np.maximum.reduce([lo-y[cal_ix],y[cal_ix]-hi,np.zeros(len(cal_ix))])
        event_scores=np.array([np.max(scores[events[cal_ix]==e]) for e in np.unique(events[cal_ix])])
        # Conservative calibration on whole inspection events; empirical coverage only.
        q=min(1.,np.ceil((len(event_scores)+1)*.9)/len(event_scores))
        model['interval_adjustment']=float(np.quantile(event_scores,q,method='higher'))
        tf=subset(features,test_ix)
        xp=fusion_matrix(expert_predictions(model['experts'],tf),tf)
        pred=model['fusion'].predict(xp);lo,hi=interval(model,xp)
        report={'split':split,'oof_folds':fold_log,'dataset_fingerprint':model['fingerprint'],
                'target':'next scheduled inspection degradation; dataset label convention',
                'interval_note':'90% nominal event-calibrated interval; coverage is empirical, not a guarantee under correlated or shifted data.',
                'calibration_events':int(len(event_scores)),
                'interval_adjustment_adu':model['interval_adjustment'],
                'fusion':metrics(y[test_ix],pred,devices[test_ix],events[test_ix],lo,hi),
                'experts':{},'ablation':{},'stress_tests':{}}
        for k in EXPERTS:
            valid=np.ones(len(test_ix),bool)
            if k=='image':valid=tf['quality'][:,0]>0
            if k=='text':valid=tf['quality'][:,2]>0
            pp=model['experts'][k].predict(tf[k])
            report['experts'][k]=metrics(y[test_ix][valid],pp[valid],devices[test_ix][valid],events[test_ix][valid])
        raw=np.column_stack([features[k] for k in EXPERTS]+[features['quality']])
        progress('Training direct-feature-fusion baseline')
        direct=estimator().fit(raw[ix],y[ix],sample_weight=weights)
        report['ablation']['single_model_direct_fusion']=metrics(y[test_ix],direct.predict(raw[test_ix]),devices[test_ix],events[test_ix])
        model['direct_baseline']=direct
        avg=np.nanmean(expert_predictions(model['experts'],tf),axis=1)
        report['ablation']['fixed_expert_average']=metrics(y[test_ix],avg,devices[test_ix],events[test_ix])
        for name,hide_img,hide_txt in [('no_current_image',True,False),('no_current_text',False,True),('no_image_or_text',True,True)]:
            mf=mask_features(tf,hide_img,hide_txt);mx=fusion_matrix(expert_predictions(model['experts'],mf),mf)
            pp=model['fusion'].predict(mx);ml,mh=interval(model,mx)
            report['stress_tests'][name]=metrics(y[test_ix],pp,devices[test_ix],events[test_ix],ml,mh)
        for name,mask in [('naturally_missing_image',tf['quality'][:,0]==0),
                          ('repeated_text',tf['quality'][:,6]>0),('new_repair',tf['quality'][:,10]>0),
                          ('long_horizon',tf['numeric'][:,0]>=np.median(features['numeric'][ix,0]))]:
            if mask.sum()>2:
                report['stress_tests'][name]=metrics(y[test_ix][mask],pred[mask],devices[test_ix][mask],events[test_ix][mask],lo[mask],hi[mask])
        report['duration_seconds']=round(time.monotonic()-started,2)
        outdir=ARTIFACTS if mode=='device' else ARTIFACTS/'temporal'
        outdir.mkdir(parents=True,exist_ok=True)
        model_tmp=outdir/'model.joblib.tmp'
        joblib.dump(model,model_tmp)
        model_tmp.replace(outdir/'model.joblib')
        (outdir/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        predictions=[{'device':records[i].device,'index':records[i].index,'time':records[i].time,
                      'target':float(y[i]),'prediction':float(p),'lower':float(l),'upper':float(h)}
                      for i,p,l,h in zip(test_ix,pred,lo,hi)]
        (outdir/'test_predictions.json').write_text(json.dumps(predictions,indent=2))
    progress(f'Complete: MAE={report["fusion"]["mae_adu"]:.4f} adu')
    return model,report
