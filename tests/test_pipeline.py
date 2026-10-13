from dataclasses import replace
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from sg_agents.data import Record,load_dataset,reconstruct,trajectory_groups
from sg_agents.features import curve_features,build_features,mask_features
from sg_agents.modeling import split_records
from sg_agents.provider import decode_json,validate_extraction,ProviderError

def row(index,time,since,text='old',target=9999):
    return Record(1,index,time,50,since,0,0,text,'',target)

def test_reconstruction_ignores_future_and_labels():
    rows=[row(1,0,0),row(2,10,10),row(3,20,0),row(4,30,10,'new')]
    full=reconstruct(rows)
    for n in range(1,len(rows)+1):
        assert reconstruct(rows[:n])[rows[n-1].uid]==full[rows[n-1].uid]
    assert full['1:3']['new_repair']
    assert full['1:3']['repair_since_text']
    assert full['1:4']['text_age']==0
    assert reconstruct([replace(r,target=-r.target) for r in rows])==full

def test_text_repeat_is_not_exact_generation_time():
    rows=[row(1,0,0,'A'),row(2,10,10,'B'),row(3,20,20,'A')]
    s=reconstruct(rows)['1:3']
    assert s['text_first_seen_at']==0
    assert s['text_run_start']==20
    assert s['text_time_unknown']

def test_axes_only_image_is_missing(tmp_path):
    im=Image.new('L',(194,325),255);d=ImageDraw.Draw(im);d.rectangle((43,25,175,288),outline=0)
    path=tmp_path/'blank.tiff';im.save(path)
    values,meta=curve_features(str(path))
    assert not meta['available'] and meta['reason']=='blank_plot'
    assert np.isnan(values).all()
    d.line([(50,180),(70,195),(90,210),(110,195),(145,150)],fill=0,width=1);im.save(path)
    values,meta=curve_features(str(path))
    assert meta['available'] and np.isfinite(values).sum()>10

def test_dataset_alignment_and_split_no_event_leakage():
    rows=load_dataset()
    assert len(rows)==7626 and len({r.device for r in rows})==50
    for mode in ('device','temporal'):
        masks=split_records(rows,mode)[:3]
        groups=[set((r.device,r.next_inspection) for r,m in zip(rows,mask) if m) for mask in masks]
        assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    assert split_records(rows)[3]['record_counts']['test']>0

def test_duplicate_trajectories_never_cross_device_splits():
    rows=load_dataset();mapping=trajectory_groups(rows)
    assert len(set(mapping.values()))==30
    assert all(mapping[d]==mapping[d+30] for d in range(1,21))
    masks=split_records(rows,'device')[:3]
    sets=[{mapping[r.device] for r,m in zip(rows,mask) if m} for mask in masks]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    assert trajectory_groups([replace(r,target=None) for r in rows])==mapping

def test_feature_builder_does_not_read_targets():
    a=[row(1,0,0),row(2,10,0,'flushed')]
    b=[replace(r,target=None) for r in a]
    fa,fb=build_features(a),build_features(b)
    for name in ('numeric','image','text','maintenance','quality'):
        np.testing.assert_allclose(fa[name],fb[name],equal_nan=True)
    masked=mask_features(fa,True,True)
    assert np.isnan(masked['image'][:,2:]).all()
    assert np.isnan(masked['text'][:,2:]).all()
    assert not masked['quality'][:,0:4].any()

def test_minimax_json_and_evidence_validation():
    assert decode_json('<think>ignore [1]</think>\n```json\n{"ok":true}\n```')=={'ok':True}
    text='the fouling is moderate. we do mechanical cleaning'
    valid={'severity_before':2,'severity_after':None,'maintenance':'mechanical','urgent':False,'evidence':'fouling is moderate'}
    assert validate_extraction(valid,text)['source']=='minimax'
    import pytest
    with pytest.raises(ProviderError):validate_extraction(dict(valid,evidence='made up'),text)

def test_training_handles_entirely_missing_curve_columns():
    from sg_agents.modeling import estimator
    from threadpoolctl import threadpool_limits
    x=np.c_[np.arange(80),np.full(80,np.nan),np.ones(80)]
    with threadpool_limits(limits=2):
        m=estimator().fit(x,np.arange(80)*.2)
        assert 1 not in m.columns_
        assert np.isfinite(m.predict([[30,np.nan,1]])).all()

def test_record_api_does_not_send_ground_truth_to_explainer(monkeypatch):
    import asyncio
    from sg_agents import agents
    captured={}
    async def fake(system,user,max_tokens,**kwargs):
        captured['context']=user
        return '{"summary":"ok","evidence":[],"limitations":[]}',{}
    monkeypatch.setattr(agents,'call_model',fake)
    p={k:None for k in ('observed_at','next_inspection_at','horizon_atu','prediction_adu','interval_90_adu','experts','maintenance','text_semantics','warnings','input')}
    p['evaluation_target_adu']=999999
    asyncio.run(agents.explain(p))
    assert '999999' not in captured['context']
    assert 'evaluation_target' not in captured['context']
