import asyncio
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
from .config import ROOT, ARTIFACTS, MODEL, BASE_URL
from .data import load_dataset, Record, fingerprint,trajectory_groups
from .features import build_features
from .agents import predict_record, explain
from .provider import connection_check, ensure_texts, read_cache, ProviderError
from .llm_settings import get_settings,PROVIDERS

class Runtime:
    def __init__(self):
        self.rows=[];self.features=None;self.model=None;self.by_uid={};self.report={}
        self.lock=threading.RLock();self.job=None;self.model_warning=None

    def load(self):
        with self.lock:
            rows=load_dataset();fp=fingerprint(rows)
            cache_path=ARTIFACTS/'features.joblib'
            f=None
            if cache_path.is_file():
                cached=joblib.load(cache_path)
                if cached['fingerprint']==fp and cached.get('semantic_namespace','minimax-m27-legacy-v1')==get_settings().namespace:f=cached['features']
            if f is None:f=build_features(rows)
            model_path=ARTIFACTS/'model.joblib'
            model=joblib.load(model_path) if model_path.exists() else None
            if model and model['fingerprint']!=fp:
                raise ValueError('数据集已变更，请重新训练，不能沿用旧模型')
            self.model_warning=None
            if model and model.get('semantic_namespace','minimax-m27-legacy-v1')!=get_settings().namespace:
                model=None
                self.model_warning='提取供应商与训练语义缓存不一致，请重新训练；只更换解释 API 无需重训'
            self.rows=rows;self.features=f;self.model=model
            self.by_uid={r.uid:i for i,r in enumerate(rows)}
            metrics=ARTIFACTS/'metrics.json'
            self.report=json.loads(metrics.read_text()) if metrics.exists() else {}

    def lookup(self,device,index):
        key=f'{device}:{index}'
        if key not in self.by_uid:raise HTTPException(404,'找不到观察记录')
        return self.by_uid[key]

    def require_model(self):
        if self.model is None:raise HTTPException(409,self.model_warning or '模型尚未训练完成')

runtime=Runtime()

@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(runtime.load)
    yield

app=FastAPI(title='SteamWeave · Maintenance-aware Multimodal Prognostics',version='1.1.0',lifespan=lifespan)
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')

@app.middleware('http')
async def same_origin_mutations(request, call_next):
    origin=request.headers.get('origin')
    if request.method not in ('GET','HEAD','OPTIONS') and origin:
        expected=f'{request.url.scheme}://{request.headers.get("host", "")}'
        if origin!=expected:
            return JSONResponse({'detail':'本地管理操作仅允许来自当前界面'},status_code=403)
    return await call_next(request)

@app.get('/')
def home():return FileResponse(ROOT/'static'/'index.html')

@app.get('/api/health')
def health():
    status_path=ARTIFACTS/'provider_status.json'
    provider=json.loads(status_path.read_text()) if status_path.exists() else {'connected':False,'model':MODEL}
    if provider.get('provider','minimax')!=get_settings().provider:
        provider={'connected':False,'model':get_settings().model,'provider':get_settings().provider}
    return {'status':'ok','model_ready':runtime.model is not None,'records':len(runtime.rows),
            'minimax':provider,'provider':get_settings().public(),'audit_provider':get_settings(audit=True).public(),
            'credential_configured':get_settings().public()['credential_configured'],'model_warning':runtime.model_warning}

@app.get('/api/providers')
def providers():
    return {'active_extraction':get_settings().provider,'active_audit':get_settings(audit=True).provider,
            'providers':[get_settings(p).public() for p in PROVIDERS]}

@app.post('/api/provider/check')
async def check(provider:str|None=None):
    try:return await connection_check(provider)
    except ValueError as e:raise HTTPException(422,str(e)) from None
    except ProviderError as e:raise HTTPException(502,str(e)) from None

@app.get('/api/summary')
def summary():
    q=runtime.features['quality']
    temporal=ARTIFACTS/'temporal'/'metrics.json'
    temporal_report=json.loads(temporal.read_text()) if temporal.is_file() else None
    return {'records':len(runtime.rows),'devices':len({r.device for r in runtime.rows}),
            'trajectory_groups':len(set(trajectory_groups(runtime.rows).values())),
            'image_available':int(q[:,0].sum()),'image_missing':int((q[:,0]==0).sum()),
            'unique_reports':len({r.text for r in runtime.rows}),'minimax_reports':len(read_cache()),
            'model_ready':runtime.model is not None,'report':runtime.report,'temporal_report':temporal_report,
            'citation':'Zhe Yang, Piero Baraldi, Enrico Zio. A Multi-Branch Deep Neural Network Model for Failure Prognostics Based on Multimodal Data.'}

@app.get('/api/devices')
def devices():
    out=[]
    for d in sorted({r.device for r in runtime.rows}):
        rows=[r for r in runtime.rows if r.device==d]
        partition='untrained'
        if runtime.model:
            split=runtime.model['split']
            partition='test' if d in split['test_devices'] else 'calibration' if d in split['calibration_devices'] else 'train'
        out.append({'device':d,'records':len(rows),'time_range':[rows[0].time,rows[-1].time],'partition':partition})
    return out

@app.get('/api/devices/{device}/records')
def records(device:int):
    return [{'index':r.index,'time':r.time,'horizon':r.horizon,'next_inspection':r.next_inspection,
             'image_available':runtime.features['image_meta'][i]['available'],
             'new_repair':runtime.features['state'][r.uid]['new_repair']}
            for i,r in enumerate(runtime.rows) if r.device==device]

@app.get('/api/devices/{device}/history')
def history(device:int,through:float|None=None):
    runtime.require_model()
    rows=[]
    for i,r in enumerate(runtime.rows):
        if r.device==device and (through is None or r.time<=through):
            p=predict_record(r,runtime.features,i,runtime.model)
            rows.append({'time':r.time,'index':r.index,'prediction':p['prediction_adu'],
                         'lower':p['interval_90_adu'][0],'upper':p['interval_90_adu'][1],
                         'target':r.target,'new_repair':p['maintenance']['new_repair'],
                         'image_available':p['image']['available']})
    return rows

@app.get('/api/records/{device}/{index}/image')
def image(device:int,index:int):
    r=runtime.rows[runtime.lookup(device,index)]
    if not r.image_path:raise HTTPException(404,'图片不存在')
    out=io.BytesIO()
    with Image.open(r.image_path) as im:im.convert('RGB').save(out,format='PNG')
    return StreamingResponse(io.BytesIO(out.getvalue()),media_type='image/png')

class PredictionRequest(BaseModel):
    device:int=Field(ge=1)
    index:int=Field(ge=1)
    hide_image:bool=False
    hide_text:bool=False
    provider:str|None=None

@app.post('/api/predict')
def predict(req:PredictionRequest):
    runtime.require_model();i=runtime.lookup(req.device,req.index);r=runtime.rows[i]
    p=predict_record(r,runtime.features,i,runtime.model,req.hide_image,req.hide_text)
    # Added only after inference, never sent to the explanation provider.
    p['evaluation_target_adu']=r.target
    return p

@app.post('/api/explain')
async def explanation(req:PredictionRequest):
    runtime.require_model();i=runtime.lookup(req.device,req.index);r=runtime.rows[i]
    p=await asyncio.to_thread(predict_record,r,runtime.features,i,runtime.model,req.hide_image,req.hide_text)
    try:
        cfg=get_settings(req.provider,audit=True)
        result,usage=await explain(p,provider=cfg.provider)
        return {'llm_status':'completed','model':cfg.model,'provider':cfg.provider,'explanation':result,'usage':usage}
    except ValueError as e:raise HTTPException(422,str(e)) from None
    except ProviderError as e:raise HTTPException(502,str(e)) from None

@app.post('/api/predict/custom')
async def custom(device:int=Form(...),time:float=Form(...),horizon:float=Form(...),
                 since_repair:float=Form(...),chemical_count:float=Form(...),
                 mechanical_count:float=Form(...),text:str=Form(''),
                 curve:UploadFile|None=File(None)):
    runtime.require_model()
    if device<1 or not all(np.isfinite(v) and v>=0 for v in (time,horizon,since_repair,chemical_count,mechanical_count)):
        raise HTTPException(422,'设备 ID 应为正整数，数值应为有限非负数')
    if chemical_count%1 or mechanical_count%1:
        raise HTTPException(422,'清洗次数应为整数')
    if len(text)>10000:raise HTTPException(413,'报告过长，最多 10000 字符')
    data=await curve.read(10*1024*1024+1) if curve else None
    if data and len(data)>10*1024*1024:raise HTTPException(413,'曲线图片最多 10 MB')
    temp=None
    try:
        if data:
            try:
                im=Image.open(io.BytesIO(data))
                if im.width*im.height>20_000_000:
                    raise HTTPException(413,'曲线图片尺寸过大')
                im.verify()
            except (OSError,ValueError):raise HTTPException(422,'无法读取曲线图片') from None
            temp=tempfile.NamedTemporaryFile(suffix='.tiff',delete=False)
            temp.write(data);temp.close()
        semantic_status='not_needed'
        if text:
            try:
                await ensure_texts([text],batch_size=1,concurrency=1)
                semantic_status='completed'
            except ProviderError:
                semantic_status='unavailable'
        r=Record(device,0,time,horizon,since_repair,chemical_count,mechanical_count,text,
                 temp.name if temp else '',None)
        past=[row for row in runtime.rows if row.device==device and row.time<time]
        # Feature reconstruction contains input histories, never historical labels.
        ff=await asyncio.to_thread(build_features,past+[r])
        p=await asyncio.to_thread(predict_record,r,ff,len(past),runtime.model)
        p['evaluation_partition']='custom';p['semantic_status']=semantic_status;p['semantic_provider']=get_settings().provider
        return p
    finally:
        if temp:Path(temp.name).unlink(missing_ok=True)

@app.get('/api/export/{device}')
def export(device:int):
    rows=history(device)
    if not rows:raise HTTPException(404,'设备不存在')
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return StreamingResponse(iter([('\ufeff'+out.getvalue()).encode('utf-8')]),media_type='text/csv',
                             headers={'Content-Disposition':f'attachment; filename=device_{device}_predictions.csv'})

@app.post('/api/train')
def training():
    with runtime.lock:
        if runtime.job and runtime.job.poll() is None:raise HTTPException(409,'训练正在进行')
        log=ARTIFACTS/'training.log'
        with log.open('w') as out:
            runtime.job=subprocess.Popen([sys.executable,'-u','-m','sg_agents.cli','train'],cwd=ROOT,
                        stdout=out,stderr=subprocess.STDOUT,env=os.environ.copy())
        return {'status':'started','pid':runtime.job.pid}

@app.get('/api/train/status')
def training_status():
    log=ARTIFACTS/'training.log'
    lines=log.read_text(errors='replace').splitlines()[-18:] if log.exists() else []
    code=runtime.job.poll() if runtime.job else None
    return {'running':bool(runtime.job and code is None),'exit_code':code,'log':lines}

@app.post('/api/reload')
async def reload_model():
    if runtime.job and runtime.job.poll() is None:raise HTTPException(409,'请等待训练完成再加载')
    try:await asyncio.to_thread(runtime.load)
    except ValueError as e:raise HTTPException(409,str(e)) from None
    return {'model_ready':runtime.model is not None}
