from dataclasses import replace
import numpy as np
from sg_agents.data import Record
from sg_agents.optimization import Candidate,ForecastHead,causal_curve_history,curve_shape,matrix,development_splits
from sg_agents.features import mask_features
from threadpoolctl import threadpool_limits
import pytest


def fixture():
    records=[Record(1,i+1,float(i*10),10.,float(i*10),0.,0.,'report','',float(i)) for i in range(6)]
    x=np.linspace(0,1,32)
    curves=np.array([np.r_[.2*(x-.4)**2+.1+i*.01,np.zeros(8)] for i in range(6)])
    f={'numeric':np.array([[r.horizon,r.since_repair,0,0] for r in records]),'image':np.c_[np.zeros((6,2)),curves],
       'quality':np.ones((6,12)),'text':np.zeros((6,72)),'maintenance':np.zeros((6,17))}
    return records,f


def test_curve_shape_preserves_quadratic_geometry():
    _,f=fixture();s=curve_shape(f['image'][:,2:])
    np.testing.assert_allclose(s[0,:5],[.2,-.16,.132,.4,.1],atol=1e-8)
    assert s[0,5]<1e-8
    assert np.isnan(curve_shape(np.full((1,40),np.nan))).all()


def test_curve_history_is_strictly_causal_and_ignores_targets():
    rows,f=fixture();h=causal_curve_history(rows,f)
    assert np.isnan(h[0,:20]).all()
    for n in range(1,len(rows)+1):
        prefix={k:v[:n] for k,v in f.items()}
        np.testing.assert_allclose(causal_curve_history(rows[:n],prefix)[-1],h[n-1],equal_nan=True)
    np.testing.assert_allclose(causal_curve_history([replace(r,target=-9999) for r in rows],f),h,equal_nan=True)
    modified={k:v.copy() for k,v in f.items()};modified['image'][3,2:]+=1000
    np.testing.assert_allclose(causal_curve_history(rows,modified)[:4],h[:4],equal_nan=True)


def test_curve_history_resets_before_using_current_repair_curve():
    rows,f=fixture();rows=[replace(r,since_repair=r.time-30) if r.time>=30 else r for r in rows]
    h=causal_curve_history(rows,f)
    assert np.isnan(h[3,:20]).all() and h[3,-3]==0
    assert h[4,-3]==1
    np.testing.assert_allclose(h[4,:10],curve_shape(f['image'][3:4,2:])[0])


def test_current_modality_mask_does_not_restore_current_curve():
    rows,f=fixture();h=causal_curve_history(rows,f);masked=mask_features(f,True,True)
    x=matrix(masked,h,'history')
    assert np.isnan(x[:,4:14]).all()
    assert np.isnan(x[:,-3:]).all()
    np.testing.assert_allclose(h,causal_curve_history(rows,f),equal_nan=True)
    # Past observed curves remain available when only the current modality is hidden.
    assert np.isfinite(x[1,23:33]).all()


def test_training_selector_and_imputation_do_not_use_prediction_rows():
    x=np.c_[np.arange(80.),np.full(80,np.nan)];y=np.arange(80.)*.3
    with threadpool_limits(limits=2):
        head=ForecastHead(Candidate('test','hgb',iterations=10,minimum=5)).fit(x,y)
        assert head.columns_.tolist()==[0]
        np.testing.assert_allclose(head.predict([[20,np.nan]]),head.predict([[20,9999]]))
        with pytest.raises(ValueError):head.predict([[20]])


def test_forward_inner_validation_purges_label_horizon():
    rows=[Record(d,i+1,float(i*10),15.,float(i*10),0,0,'','',1) for d in (1,2) for i in range(30)]
    train=np.arange(len(rows));groups=np.array([r.device for r in rows])
    for fit,val in development_splits(rows,train,groups,temporal=True):
        assert not set(fit)&set(val)
        for d in (1,2):
            assert max(rows[i].next_inspection for i in fit if rows[i].device==d)<=min(rows[i].time for i in val if rows[i].device==d)


def test_serving_uses_optimized_head_and_its_interval():
    from sklearn.dummy import DummyRegressor
    from sg_agents.agents import predict_record
    from sg_agents.data import reconstruct
    rows,f=fixture();f['curve_history']=causal_curve_history(rows,f)
    f['state']=reconstruct(rows);f['image_meta']=[{'available':True,'coverage':1.} for _ in rows]
    f['semantic_evidence']=[None]*len(rows)
    candidate=Candidate('test','hgb','history',iterations=10,minimum=2)
    with threadpool_limits(limits=2):
        head=ForecastHead(candidate).fit(matrix(f,f['curve_history'],'history'),np.arange(6.))
    constant=lambda value:DummyRegressor(strategy='constant',constant=value).fit([[0]], [value])
    model={'experts':{k:constant(7.) for k in ('numeric','image','text','maintenance')},'fusion':constant(7.),
        'lower':constant(0.),'upper':constant(20.),'split':{'test_devices':[1]},
        'optimized_head':{'model':head,'version':'curve-history-v2','interval_adjustment':3.}}
    p=predict_record(rows[3],f,3,model)
    assert p['prediction_method']=='curve-history-v2' and p['forecast_head']=='test'
    assert p['legacy_prediction_adu']==7.
    assert len(p['trace'])==7
    np.testing.assert_allclose(p['interval_90_adu'],[p['prediction_adu']-3,p['prediction_adu']+3])
    assert any('诊断' in s for s in p['warnings'])

    masked=predict_record(rows[3],f,3,model,hide_image=True)
    assert masked['prediction_adu']==7.
    assert masked['forecast_head']=='legacy_missing_modality_fusion'
