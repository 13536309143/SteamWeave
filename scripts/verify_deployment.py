"""Exercise the running project's real data, predictions, uploads and provider."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys
import httpx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def verify(with_minimax=False):
    checks=[]
    with httpx.Client(base_url='http://127.0.0.1:8787',timeout=120) as client:
        def get(path):
            r=client.get(path);r.raise_for_status();return r.json()
        health=get('/api/health');assert health['model_ready'];checks.append('model_ready')
        summary=get('/api/summary');assert summary['records']==7626 and summary['devices']==50
        assert summary['minimax_reports']>=summary['unique_reports'];checks.append('complete_semantic_cache')
        assert summary['trajectory_groups']==30
        assert summary['report']['split']['trajectory_group_counts']=={'train':21,'calibration':3,'test':6}
        for fold in summary['report']['oof_folds']:
            assert not set(fold['fit_trajectory_groups']) & set(fold['validation_trajectory_groups'])
        checks.append('duplicate_trajectory_isolation')
        devices=get('/api/devices');d=next(v['device'] for v in devices if v['partition']=='test')
        records=get(f'/api/devices/{d}/records');r=next(v for v in records if v['image_available'] and v['index']>5)
        request={'device':d,'index':r['index']}
        response=client.post('/api/predict',json=request);response.raise_for_status();prediction=response.json()
        assert prediction['evaluation_partition']=='test'
        assert len(prediction['trace'])==7 and prediction['text_semantics']['source']=='minimax'
        assert prediction['interval_90_adu'][0]<=prediction['interval_90_adu'][1];checks.append('heldout_record_prediction')
        for image,text in [(True,False),(False,True),(True,True)]:
            out=client.post('/api/predict',json=dict(request,hide_image=image,hide_text=text));out.raise_for_status()
            p=out.json()
            if image:assert p['experts'][1]['status']=='unavailable'
            if text:assert p['experts'][2]['status']=='unavailable'
            assert isinstance(p['prediction_adu'],float)
        checks.append('missing_modality_paths')
        history=get(f'/api/devices/{d}/history?through={r["time"]}')
        assert all(v['time']<=r['time'] for v in history);checks.append('history_cutoff')
        image=client.get(f'/api/records/{d}/{r["index"]}/image');assert image.status_code==200 and image.content.startswith(b'\x89PNG')
        export=client.get(f'/api/export/{d}');export.raise_for_status()
        rows=list(csv.DictReader(io.StringIO(export.content.decode('utf-8-sig'))));assert len(rows)==len(records);checks.append('csv_export')
        src=prediction['input']
        data={'device':d,'time':r['time'],'horizon':src['horizon'],'since_repair':src['since_repair'],
              'chemical_count':src['chemical_count'],'mechanical_count':src['mechanical_count'],'text':src['text']}
        custom=client.post('/api/predict/custom',data=data,files={'curve':('curve.png',image.content,'image/png')})
        custom.raise_for_status();custom=custom.json();assert custom['evaluation_partition']=='custom'
        assert custom['semantic_status']=='completed' and custom['image']['available'];checks.append('custom_upload_prediction')
        denied=client.post('/api/provider/check',headers={'Origin':'https://unrelated.example'})
        assert denied.status_code==403;checks.append('same_origin_management')
        assert client.get('/static/../.env').status_code!=200;checks.append('credential_not_served')
        if with_minimax:
            out=client.post('/api/explain',json=request);out.raise_for_status();audit=out.json()
            assert audit['llm_status']=='completed' and audit['explanation']['summary'];checks.append('live_minimax_explanation')
        report={'checks':checks,'sample_device':d,'sample_index':r['index'],
                'sample_prediction_adu':prediction['prediction_adu'],
                'test_mae_adu':summary['report']['fusion']['mae_adu']}
        (ROOT/'artifacts'/'deployment_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--with-minimax',action='store_true');a=p.parse_args();verify(a.with_minimax)
