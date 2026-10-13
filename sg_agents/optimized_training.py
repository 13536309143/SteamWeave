"""Deploy a training-selected forecasting head while retaining the original checkpoint."""
import json,shutil,time
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from .config import ARTIFACTS
from .data import fingerprint,trajectory_groups
from .features import mask_features
from .llm_settings import get_settings
from .modeling import split_records,subset,metrics,expert_predictions,fusion_matrix,interval,train
from .optimization import VERSION,causal_curve_history,matrix,select_head,weights_for


def train_optimized(records,features,mode='device',progress=print):
    started=time.monotonic();out=ARTIFACTS if mode=='device' else ARTIFACTS/'temporal'
    path=out/'model.joblib';report_path=out/'metrics.json'
    if not path.exists():train(records,features,mode,progress)
    legacy=joblib.load(path);report=json.loads(report_path.read_text())
    fp=fingerprint(records)
    if legacy['fingerprint']!=fp or legacy.get('semantic_namespace','minimax-m27-legacy-v1')!=get_settings().namespace:
        raise ValueError('Rebuild the legacy checkpoint using the same dataset and extraction provider first')
    train_mask,cal_mask,test_mask,split=split_records(records,mode)
    if legacy['split']!=split:raise ValueError('Legacy checkpoint split differs; retrain for requested split first')
    ix=np.flatnonzero(train_mask);cal=np.flatnonzero(cal_mask);test=np.flatnonzero(test_mask)
    y=np.array([r.target for r in records]);mapping=trajectory_groups(records)
    groups=np.array([mapping[r.device] for r in records]);devices=np.array([r.device for r in records])
    events=np.array([f'{mapping[r.device]}:{r.next_inspection:g}' for r in records])
    history=features.get('curve_history')
    if history is None:history=causal_curve_history(records,features);features['curve_history']=history
    with threadpool_limits(limits=2):
        head,selection=select_head(records,features,history,y,ix,groups,seed=20261013,temporal=mode=='temporal',
            progress=lambda n,s:progress(f'Inner validation {n}: MAE {s:.4f}'))
        xx=matrix(features,history,head.candidate.view)
        cf=subset(features,cal);cx=fusion_matrix(expert_predictions(legacy['experts'],cf),cf)
        pcal=np.where(cf['quality'][:,0]>0,head.predict(xx[cal]),legacy['fusion'].predict(cx))
        residual=np.abs(y[cal]-pcal);scores=np.array([residual[events[cal]==e].max() for e in np.unique(events[cal])])
        q=min(1.,np.ceil((len(scores)+1)*.9)/len(scores));radius=float(np.quantile(scores,q,method='higher'))
        tf=subset(features,test);legacy_x=fusion_matrix(expert_predictions(legacy['experts'],tf),tf)
        pp=legacy['fusion'].predict(legacy_x);ll,lh=interval(legacy,legacy_x)
        prediction=np.where(tf['quality'][:,0]>0,head.predict(xx[test]),pp);lo=prediction-radius;hi=prediction+radius
        original_metrics=metrics(y[test],pp,devices[test],events[test],ll,lh)
        optimized_metrics=metrics(y[test],prediction,devices[test],events[test],lo,hi)
        stress={}
        for name,im,txt in [('no_current_image',True,False),('no_current_text',False,True),('no_image_or_text',True,True)]:
            masked=mask_features(tf,im,txt);mx=fusion_matrix(expert_predictions(legacy['experts'],masked),masked)
            p=np.where(masked['quality'][:,0]>0,head.predict(matrix(masked,history[test],head.candidate.view)),legacy['fusion'].predict(mx))
            stress[name]=metrics(y[test],p,devices[test],events[test],p-radius,p+radius)
        for name,mask in [('naturally_missing_image',tf['quality'][:,0]==0),('repeated_text',tf['quality'][:,6]>0),
                         ('new_repair',tf['quality'][:,10]>0),('long_horizon',tf['numeric'][:,0]>=np.median(features['numeric'][ix,0]))]:
            if mask.sum()>2:stress[name]=metrics(y[test][mask],prediction[mask],devices[test][mask],events[test][mask],lo[mask],hi[mask])
    backup=out/'baselines'/'legacy-v1';backup.mkdir(parents=True,exist_ok=True)
    if not (backup/'model.joblib').exists():
        original=dict(legacy);original.pop('optimized_head',None);joblib.dump(original,backup/'model.joblib')
        shutil.copyfile(report_path,backup/'metrics.json')
    legacy['optimized_head']={'model':head,'version':VERSION,'selection':selection,'interval_adjustment':radius,
                              'routing':'legacy_when_curve_missing',
                              'interval_note':'Absolute-error event-max calibration on held-out groups; empirical coverage only'}
    report.update(fusion=optimized_metrics,legacy_fusion=original_metrics,stress_tests=stress,
                  prediction_method=VERSION,optimization_selection=selection,
                  expert_note='Current curve available: optimized geometry/history head; curve missing: original augmented expert fusion.',
                  interval_adjustment_adu=radius,calibration_events=len(scores),duration_seconds=round(time.monotonic()-started,2))
    tmp=out/'model.joblib.tmp';joblib.dump(legacy,tmp);tmp.replace(path)
    tmp=out/'metrics.json.tmp';tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2));tmp.replace(report_path)
    predictions=[{'device':records[i].device,'index':records[i].index,'time':records[i].time,'target':float(y[i]),
                  'prediction':float(p),'lower':float(l),'upper':float(h)} for i,p,l,h in zip(test,prediction,lo,hi)]
    (out/'test_predictions.json').write_text(json.dumps(predictions,indent=2))
    progress(f'Optimized {head.candidate.name}: MAE={optimized_metrics["mae_adu"]:.4f}; original={original_metrics["mae_adu"]:.4f}')
    return legacy,report
