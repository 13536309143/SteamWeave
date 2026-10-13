import argparse
import asyncio
import json
import joblib
import fcntl
from .config import ARTIFACTS
from .data import load_dataset,fingerprint,trajectory_groups
from .features import build_features
from .provider import ensure_texts, connection_check, read_cache, text_key
from .modeling import train
from .llm_settings import get_settings

def main():
    parser=argparse.ArgumentParser(description='蒸汽发生器多 agent 预测')
    parser.add_argument('command',choices=['prepare','train','check-api'])
    parser.add_argument('--split',choices=['device','temporal'],default='device')
    parser.add_argument('--offline',action='store_true',help='不请求模型 API；缺少语义缓存时显式使用缺失语义特征')
    parser.add_argument('--provider',choices=['minimax','openai','qwen','custom'])
    args=parser.parse_args()
    if args.provider and args.command!='check-api':
        __import__('os').environ['LLM_PROVIDER']=args.provider
    if args.command=='check-api':
        print(json.dumps(asyncio.run(connection_check(args.provider)),ensure_ascii=False,indent=2));return
    lock=(ARTIFACTS/'.pipeline.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('已有数据准备或训练任务正在运行，请等待完成。') from None
    rows=load_dataset()
    if not args.offline:
        asyncio.run(ensure_texts([r.text for r in rows],progress=lambda d,t,e:print(f'{get_settings().provider} 文本 {d}/{t}，失败 {e}',flush=True)))
    print('Extracting image and causal history features...',flush=True)
    f=build_features(rows,progress=lambda d,t:print(f'图像特征 {d}/{t}',flush=True))
    cache=read_cache()
    groups=trajectory_groups(rows)
    summary={'records':len(rows),'devices':len({r.device for r in rows}),
             'trajectory_groups':len(set(groups.values())),
             'duplicate_device_groups':[[d for d in sorted(groups) if groups[d]==g]
                 for g in sorted(set(groups.values())) if sum(v==g for v in groups.values())>1],
             'image_available':int(f['quality'][:,0].sum()),
             'image_missing':int((f['quality'][:,0]==0).sum()),
             'unique_reports':len({r.text for r in rows}),
             'minimax_reports':len({text_key(r.text) for r in rows if text_key(r.text) in cache}),
             'fingerprint':fingerprint(rows)}
    (ARTIFACTS/'dataset_summary.json').write_text(json.dumps(summary,indent=2))
    feature_tmp=ARTIFACTS/'features.joblib.tmp'
    joblib.dump({'fingerprint':summary['fingerprint'],'features':f,'semantic_namespace':get_settings().namespace},feature_tmp)
    feature_tmp.replace(ARTIFACTS/'features.joblib')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if args.command=='train':train(rows,f,mode=args.split,progress=lambda s:print(s,flush=True))

if __name__=='__main__':main()
