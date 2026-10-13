"""Frozen protocols, paired predictions, ablations and missing-modality stress tests."""
import argparse
from collections import defaultdict
from dataclasses import replace
import gzip
import hashlib
import json
import platform
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import KFold,GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from sg_agents.config import ROOT,ARTIFACTS
from sg_agents.data import load_dataset,trajectory_groups,fingerprint
from sg_agents.features import fusion_matrix,mask_features
from sg_agents.features import build_features
from sg_agents.modeling import estimator,EXPERTS,subset,expert_predictions,augment_fusion,interval,split_records
from sg_agents.provider import read_cache

SEEDS=(17,41,73)
VARIANTS=('full','without_maintenance_agent','without_freshness','without_llm_semantics',
          'without_modality_dropout','without_text_agent','without_image_agent')

def score(y,p):
    positive=np.abs(y)>1e-12
    return {'mae_adu':float(np.mean(np.abs(p-y))),'mse_adu2':float(np.mean((p-y)**2)),
            'rmse_adu':float(np.sqrt(np.mean((p-y)**2))),
            'mre_percent':float(100*np.mean(np.abs((p[positive]-y[positive])/y[positive]))),
            'mre_valid_records':int(positive.sum()),'records':len(y)}

def make_splits(records,protocol,seed):
    mapping=trajectory_groups(records);devices=np.array([r.device for r in records]);groups=np.array([mapping[r.device] for r in records])
    if protocol=='temporal':
        a,b,c,info=split_records(records,'temporal')
        yield 0,np.flatnonzero(a),np.flatnonzero(b),np.flatnonzero(c),info;return
    units=np.unique(groups if protocol=='grouped_cv' else devices)
    splitter=KFold(n_splits=5,shuffle=True,random_state=seed)
    for fold,(a,b) in enumerate(splitter.split(units)):
        outer_train=np.flatnonzero(np.isin(groups if protocol=='grouped_cv' else devices,units[a]))
        test=np.flatnonzero(np.isin(groups if protocol=='grouped_cv' else devices,units[b]))
        rng=np.random.default_rng(seed*100+fold)
        if protocol=='grouped_cv':
            ug=rng.permutation(np.unique(groups[outer_train]));cal_g=ug[:max(2,round(len(ug)*.125))]
            cal=outer_train[np.isin(groups[outer_train],cal_g)]
            train=outer_train[~np.isin(groups[outer_train],cal_g)]
        else:
            perm=rng.permutation(outer_train);count=max(1,int(len(perm)*.05));cal=perm[:count];train=perm[count:]
        overlap=[int(g) for g in sorted(set(groups[train])&set(groups[test]))]
        info={'protocol':protocol,'seed':seed,'fold':fold,'train_devices':sorted(np.unique(devices[train]).tolist()),
              'validation_devices':sorted(np.unique(devices[cal]).tolist()),'test_devices':sorted(np.unique(devices[test]).tolist()),
              'train_groups':sorted(np.unique(groups[train]).tolist()),'validation_groups':sorted(np.unique(groups[cal]).tolist()),
              'test_groups':sorted(np.unique(groups[test]).tolist()),'duplicate_groups_crossing_train_test':overlap,
              'counts':{'train':len(train),'validation':len(cal),'test':len(test)},
              'diagnostic_only':protocol=='paper_device_cv'}
        if protocol=='grouped_cv':assert not overlap and not(set(groups[train])&set(groups[cal]))
        yield fold,train,cal,test,info

def transformed(features,variant):
    f={k:v.copy() for k,v in features.items() if isinstance(v,np.ndarray)}
    active=list(EXPERTS)
    if variant=='without_maintenance_agent':
        active.remove('maintenance');f['quality'][:,[7,10]]=np.nan
    elif variant=='without_freshness':
        f['quality'][:,4:10]=np.nan;f['maintenance'][:,4:10]=np.nan
    elif variant=='without_llm_semantics':
        f['text'][:,2:8]=np.nan;f['quality'][:,3]=0
    elif variant=='without_text_agent':
        active.remove('text');f=mask_features(f,False,True)
    elif variant=='without_image_agent':
        active.remove('image');f=mask_features(f,True,False)
    return f,active

def weighted_events(records,ix,groups,balanced=True):
    if not balanced:return np.ones(len(ix))
    keys=[(groups[i],records[i].next_inspection) for i in ix];counts=defaultdict(int)
    for k in keys:counts[k]+=1
    w=np.array([1/counts[k] for k in keys]);return w/w.mean()

def predict_experts(models,f,active):
    p=np.full((len(f['numeric']),4),np.nan)
    for k in active:p[:,EXPERTS.index(k)]=models[k].predict(f[k])
    p[f['quality'][:,0]==0,1]=np.nan;p[f['quality'][:,2]==0,2]=np.nan
    return p

def fit_steam(records,features,y,train,cal,groups,seed,variant='full',balanced=True):
    start=time.monotonic();f,active=transformed(features,variant)
    weights=weighted_events(records,train,groups,balanced)
    cv_groups=groups[train] if balanced else np.array([records[i].device for i in train])
    oof=np.full((len(train),4),np.nan);folds=[]
    for a,b in GroupKFold(n_splits=4).split(train,groups=cv_groups):
        models={k:estimator(seed).fit(f[k][train[a]],y[train[a]],sample_weight=weights[a]) for k in active}
        oof[b]=predict_experts(models,subset(f,train[b]),active)
        folds.append({'fit_groups':sorted(np.unique(groups[train[a]]).tolist()),'validation_groups':sorted(np.unique(groups[train[b]]).tolist())})
        if balanced:assert not set(groups[train[a]])&set(groups[train[b]])
    tf=subset(f,train)
    if variant=='without_modality_dropout':x=fusion_matrix(oof,tf);yy=y[train];ww=weights
    else:x,yy,ww=augment_fusion(oof,tf,y[train],weights)
    fusion=estimator(seed).fit(x,yy,sample_weight=ww)
    models={k:estimator(seed).fit(f[k][train],y[train],sample_weight=weights) for k in active}
    result={'models':models,'fusion':fusion,'features':f,'active':active,'oof':folds,'training_seconds':time.monotonic()-start}
    if variant=='full':
        lower=estimator(seed,quantile=.05).fit(x,yy,sample_weight=ww)
        upper=estimator(seed,quantile=.95).fit(x,yy,sample_weight=ww)
        cf=subset(f,cal);cx=fusion_matrix(predict_experts(models,cf,active),cf)
        a=lower.predict(cx);b=upper.predict(cx);lo=np.minimum(a,b);hi=np.maximum(a,b)
        residual=np.maximum.reduce([lo-y[cal],y[cal]-hi,np.zeros(len(cal))])
        keys=np.array([f'{groups[i]}:{records[i].next_inspection}' for i in cal])
        es=np.array([residual[keys==k].max() for k in np.unique(keys)])
        q=min(1.,np.ceil((len(es)+1)*.9)/len(es))
        result.update(lower=lower,upper=upper,adjustment=float(np.quantile(es,q,method='higher')))
    return result

def steam_predict(run,features,ix):
    f=subset(features,ix);p=predict_experts(run['models'],f,run['active']);x=fusion_matrix(p,f)
    pred=run['fusion'].predict(x)
    if 'lower' in run:
        a=run['lower'].predict(x);b=run['upper'].predict(x);lo=np.minimum(a,b)-run['adjustment'];hi=np.maximum(a,b)+run['adjustment']
    else:lo=hi=None
    return pred,lo,hi,p

def record_stress(records,features,run,y,test,groups,seed):
    rng=np.random.default_rng(seed);rows=[];clean,lo,hi,_=steam_predict(run,features,test)
    tf=subset(features,test)
    for rate in (0.,.1,.3,.5,.7,1.):
        for modality in ('image','text','both'):
            f={k:v.copy() for k,v in tf.items()};selected=rng.random(len(test))<rate
            if modality in ('image','both'):f['image'][selected,2:]=np.nan;f['quality'][selected,:2]=0
            if modality in ('text','both'):f['text'][selected,2:]=np.nan;f['quality'][selected,2:4]=0
            pp=predict_experts(run['models'],f,run['active']);x=fusion_matrix(pp,f);pred=run['fusion'].predict(x)
            row={'type':'random_modality_loss','modality':modality,'rate':rate,**score(y[test],pred)}
            if 'lower' in run:
                a=run['lower'].predict(x);b=run['upper'].predict(x);low=np.minimum(a,b)-run['adjustment'];high=np.maximum(a,b)+run['adjustment']
                row.update(interval_coverage=float(np.mean((y[test]>=low)&(y[test]<=high))),interval_width_adu=float(np.mean(high-low)))
            rows.append(row)
    # Deterministic outages on the last part of each device's available test sequence.
    for rate in (.3,.5,.7):
        f={k:v.copy() for k,v in tf.items()};selected=np.zeros(len(test),bool)
        for d in sorted({records[i].device for i in test}):
            where=np.array([j for j,i in enumerate(test) if records[i].device==d]);selected[where[-max(1,int(len(where)*rate)):]]=True
        f['image'][selected,2:]=np.nan;f['quality'][selected,:2]=0
        pp=predict_experts(run['models'],f,run['active']);pred=run['fusion'].predict(fusion_matrix(pp,f))
        rows.append({'type':'continuous_curve_outage','modality':'image','rate':rate,**score(y[test],pred)})
    # Existing stale evidence and repair/horizon slices, no test labels used to choose thresholds.
    slices={'natural_image_missing':tf['quality'][:,0]==0,'reused_text':tf['quality'][:,6]>0,
            'new_repair':tf['quality'][:,10]>0,'horizon_0_10':tf['numeric'][:,0]<=10,
            'horizon_10_30':(tf['numeric'][:,0]>10)&(tf['numeric'][:,0]<=30),'horizon_over_30':tf['numeric'][:,0]>30}
    for name,m in slices.items():
        if m.any():
            row={'type':'observed_slice','slice':name,**score(y[test][m],clean[m])}
            if lo is not None:row.update(interval_coverage=float(np.mean((y[test][m]>=lo[m])&(y[test][m]<=hi[m]))),interval_width_adu=float(np.mean(hi[m]-lo[m])))
            rows.append(row)
    return rows

def delayed_report_features(records,features,lag):
    histories=defaultdict(list);changed=[];source_ages=[];source_repair=[]
    for r in records:
        history=histories[r.device];source=next((p for p in reversed(history) if p.time<=r.time-lag),None)
        changed.append(replace(r,text=source.text if source else ''))
        source_ages.append(r.time-source.time if source else 0.)
        source_repair.append(float(bool(source and r.last_repair>source.time)))
        history.append(r)
    image_cache={r.uid:(features['image'][i,2:],features['image_meta'][i]) for i,r in enumerate(records)}
    f=build_features(changed,image_cache=image_cache)
    f['quality'][:,4]+=source_ages;f['quality'][:,5]+=source_ages
    f['quality'][:,7]=source_repair
    f['maintenance'][:,4]+=source_ages;f['maintenance'][:,5]+=source_ages;f['maintenance'][:,7]=source_repair
    return f

def run(args):
    out=ARTIFACTS/'research'/('runs' if args.run_name=='main' else 'runs_'+args.run_name);out.mkdir(parents=True,exist_ok=True)
    records=load_dataset();data=joblib.load(ARTIFACTS/'features.joblib')
    if data['fingerprint']!=fingerprint(records):raise ValueError('Prepare matching dataset features first')
    features=data['features'];y=np.array([r.target for r in records],float);mapping=trajectory_groups(records)
    groups=np.array([mapping[r.device] for r in records]);nn_inputs=None
    if not args.skip_neural:
        from .paper_baseline import prepare_inputs,fit_predict
        nn_inputs=prepare_inputs(records,features,ARTIFACTS/'research'/'glove_vocab.npz')
    delayed={lag:delayed_report_features(records,features,lag) for lag in (10,25,50)}
    manifest={'project':'SteamWeave','dataset_fingerprint':data['fingerprint'],'seeds':args.seeds,
              'neural_seeds':args.neural_seeds,'neural_protocols':args.neural_protocols,'max_epochs':args.epochs,
              'patience':args.patience,'protocols':args.protocols,'python':platform.python_version(),
              'platform':platform.platform(),'semantic_cache_sha256':hashlib.sha256(json.dumps(read_cache(),sort_keys=True).encode()).hexdigest(),
              'note':'All hyperparameters fixed before new-suite evaluation. Diagnostic protocol is not an independence claim.'}
    manifest_path=out.parent/('manifest.json' if args.run_name=='main' else 'manifest_'+args.run_name+'.json')
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        keys=('dataset_fingerprint','seeds','neural_seeds','neural_protocols','max_epochs','patience','protocols','semantic_cache_sha256')
        if any(previous.get(k)!=manifest.get(k) for k in keys):
            raise ValueError('Existing run has a different configuration; choose a new --run-name to preserve results')
    manifest_path.write_text(json.dumps(manifest,indent=2))
    with threadpool_limits(limits=2):
        for protocol in args.protocols:
            seeds=args.seeds if protocol!='paper_device_cv' else args.seeds[:1]
            for seed in seeds:
                for fold,train,cal,test,info in make_splits(records,protocol,seed):
                    dest=out/protocol/f'seed_{seed}'/f'fold_{fold}';dest.mkdir(parents=True,exist_ok=True)
                    label=f'{protocol}/seed={seed}/fold={fold}'
                    complete=dest/'complete.json'
                    if complete.exists():print('Resume skip',label,flush=True);continue
                    print('START',label,'records',len(train),len(cal),len(test),flush=True)
                    (dest/'split.json').write_text(json.dumps(info,indent=2))
                    results={};predictions={};balanced=protocol!='paper_device_cv'
                    variants=VARIANTS if protocol=='grouped_cv' and seed==args.seeds[0] else ('full',)
                    for variant in variants:
                        print(label,'SteamWeave',variant,flush=True)
                        fitted=fit_steam(records,features,y,train,cal,groups,seed,variant,balanced)
                        pred,lo,hi,pp=steam_predict(fitted,fitted['features'],test)
                        results[variant]={**score(y[test],pred),'training_seconds':fitted['training_seconds']}
                        if lo is not None:
                            results[variant].update(interval_coverage=float(np.mean((y[test]>=lo)&(y[test]<=hi))),interval_width_adu=float(np.mean(hi-lo)))
                        predictions[variant]=pred
                        if protocol=='grouped_cv' and seed==args.seeds[0] and variant in ('full','without_modality_dropout'):
                            stress=record_stress(records,features,fitted,y,test,groups,seed+fold)
                            for lag,lag_features in delayed.items():
                                dlf,_=transformed(lag_features,variant);dp,_,_,_=steam_predict(fitted,dlf,test)
                                stress.append({'type':'synthetic_report_delay','lag_atu':lag,**score(y[test],dp)})
                            (dest/('stress.json' if variant=='full' else 'stress_without_modality_dropout.json')).write_text(json.dumps(stress,indent=2))
                        if variant=='full':
                            full_run=fitted;predictions['lower']=lo;predictions['upper']=hi
                            average=np.nanmean(pp,axis=1);results['fixed_expert_average']=score(y[test],average);predictions['fixed_expert_average']=average
                            (dest/'nested_oof.json').write_text(json.dumps(fitted['oof'],indent=2))
                    weights=weighted_events(records,train,groups,balanced)
                    raw=np.column_stack([features[k] for k in EXPERTS]+[features['quality']])
                    started=time.monotonic();direct=estimator(seed).fit(raw[train],y[train],sample_weight=weights)
                    pred=direct.predict(raw[test]);results['direct_feature_fusion']={**score(y[test],pred),'training_seconds':time.monotonic()-started};predictions['direct_feature_fusion']=pred
                    started=time.monotonic();rf=RandomForestRegressor(n_estimators=100,random_state=seed,n_jobs=2).fit(features['numeric'][train],y[train],sample_weight=weights)
                    pred=rf.predict(features['numeric'][test]);results['rf_numeric_100']={**score(y[test],pred),'training_seconds':time.monotonic()-started};predictions['rf_numeric_100']=pred
                    started=time.monotonic();scale=StandardScaler().fit(features['numeric'][train]);svr=SVR(C=100,kernel='rbf').fit(scale.transform(features['numeric'][train]),y[train],sample_weight=weights)
                    pred=svr.predict(scale.transform(features['numeric'][test]));results['svr_numeric_c100']={**score(y[test],pred),'training_seconds':time.monotonic()-started};predictions['svr_numeric_c100']=pred
                    if nn_inputs is not None and seed in args.neural_seeds and protocol in args.neural_protocols:
                        result=fit_predict(nn_inputs,y,train,cal,test,seed+fold,epochs=args.epochs,patience=args.patience,
                             device=args.device,progress=lambda text:print(label,text,flush=True))
                        pred=result.pop('prediction');predictions['paper_dnn_reconstruction']=pred
                        results['paper_dnn_reconstruction']={**score(y[test],pred),'training_seconds':result['training_seconds']}
                        (dest/'neural_training.json').write_text(json.dumps(result,indent=2))
                    np.savez_compressed(dest/'predictions.npz',indices=test,target=y[test],**predictions)
                    (dest/'metrics.json').write_text(json.dumps(results,indent=2))
                    complete.write_text(json.dumps({'status':'complete','label':label,'models':list(results)}))
                    print('DONE',label,'full MAE',results['full']['mae_adu'],flush=True)
    print('SUITE COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--protocols',nargs='+',choices=['grouped_cv','paper_device_cv','temporal'],default=['grouped_cv','paper_device_cv','temporal'])
    p.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS))
    p.add_argument('--neural-seeds',type=int,nargs='+',default=[17])
    p.add_argument('--neural-protocols',nargs='+',default=['grouped_cv','temporal'])
    p.add_argument('--epochs',type=int,default=80);p.add_argument('--patience',type=int,default=12)
    p.add_argument('--device',choices=['mps','cpu','cuda'],default='mps');p.add_argument('--skip-neural',action='store_true')
    p.add_argument('--run-name',default='main')
    run(p.parse_args())
