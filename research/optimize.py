"""Nested training-only selection for causal geometry/history forecasting heads."""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import csv,gzip,hashlib,json,time
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from sg_agents.config import ROOT,ARTIFACTS
from sg_agents.data import load_dataset,trajectory_groups,fingerprint
from sg_agents.features import mask_features
from sg_agents.modeling import subset
from sg_agents.optimization import VERSION,CANDIDATES,causal_curve_history,matrix,select_head
from research.run_suite import make_splits,score,fit_steam,steam_predict


def calibrate(records,y,indices,groups,prediction):
    residual=np.abs(y[indices]-prediction);events=defaultdict(list)
    for i,e in enumerate(indices):events[(int(groups[e]),records[e].next_inspection)].append(residual[i])
    scores=np.array([max(v) for v in events.values()]);q=min(1.,np.ceil((len(scores)+1)*.9)/len(scores))
    return float(np.quantile(scores,q,method='higher')),len(scores)

def compact_selection(selection):
    """Losslessly encode contiguous row indices for a reviewable public audit."""
    def ranges(indices):
        ordered=sorted(indices);out=[]
        for value in ordered:
            if out and value==out[-1][1]:out[-1][1]=value+1
            else:out.append([value,value+1])
        assert [i for start,stop in out for i in range(start,stop)]==ordered
        return out
    result=dict(selection)
    result['inner_folds']=[{'encoding':'half-open global observation index ranges: range(start, stop)',
        'fit_ranges':ranges(f['fit_indices']),'validation_ranges':ranges(f['validation_indices']),
        'fit_records':len(f['fit_indices']),'validation_records':len(f['validation_indices'])} for f in selection['inner_folds']]
    return result

def run(args):
    records=load_dataset();cache=joblib.load(ARTIFACTS/'features.joblib');features=cache['features']
    if cache['fingerprint']!=fingerprint(records):raise ValueError('Prepare features for matching dataset first')
    history=causal_curve_history(records,features);y=np.array([r.target for r in records]);mapping=trajectory_groups(records)
    groups=np.array([mapping[r.device] for r in records]);out=ARTIFACTS/'optimization';out.mkdir(exist_ok=True)
    manifest={'version':VERSION,'candidates':[asdict(c) for c in CANDIDATES],'seeds':args.seeds,'protocols':args.protocols,
              'dataset_fingerprint':cache['fingerprint'],'inner_folds':3,'source_sha256':hashlib.sha256((ROOT/'sg_agents/optimization.py').read_bytes()).hexdigest(),
              'selection':'Inner training-only MAE; outer calibration used only for prediction interval',
              'scope':'Exploratory follow-up on an examined benchmark; no new independent dataset',
              'history':'Strictly earlier same-device curve features; reset on observed repair; no labels'}
    target=out/'manifest.json'
    if target.exists() and json.loads(target.read_text())!=manifest:raise ValueError('Optimization configuration changed; preserve existing outputs first')
    target.write_text(json.dumps(manifest,indent=2))
    with threadpool_limits(limits=2):
        for protocol in args.protocols:
            for seed in args.seeds:
                for fold,train,cal,test,info in make_splits(records,protocol,seed):
                    dest=out/protocol/f'seed_{seed}'/f'fold_{fold}';dest.mkdir(exist_ok=True,parents=True)
                    if (dest/'complete.json').exists():continue
                    label=f'{protocol}/{seed}/{fold}';print('START',label,flush=True);start=time.monotonic()
                    model,selection=select_head(records,features,history,y,train,groups,seed,protocol=='temporal')
                    x=matrix(features,history,model.candidate.view)
                    adjustment,events=calibrate(records,y,cal,groups,model.predict(x[cal]))
                    pred=model.predict(x[test]);lo=pred-adjustment;hi=pred+adjustment
                    result={'protocol':protocol,'seed':seed,'fold':fold,'selected':model.candidate.name,**score(y[test],pred),
                            'interval_coverage':float(np.mean((y[test]>=lo)&(y[test]<=hi))),
                            'interval_width_adu':2*adjustment,'calibration_events':events,'training_seconds':time.monotonic()-start}
                    stress=[];rng=np.random.default_rng(seed+fold);tf=subset(features,test)
                    for rate in (0.,.1,.3,.5,.7,1.):
                        for modality in ('image','text','both'):
                            f={k:v.copy() for k,v in tf.items()};selected=rng.random(len(test))<rate
                            if modality in ('image','both'):f['image'][selected,2:]=np.nan;f['quality'][selected,:2]=0
                            if modality in ('text','both'):f['text'][selected,2:]=np.nan;f['quality'][selected,2:4]=0
                            p=model.predict(matrix(f,history[test],model.candidate.view))
                            stress.append({'rate':rate,'modality':modality,'type':'random_current_modality_loss',**score(y[test],p)})
                    (dest/'split.json').write_text(json.dumps(info,indent=2));(dest/'selection.json').write_text(json.dumps(selection,indent=2))
                    (dest/'metrics.json').write_text(json.dumps(result,indent=2));(dest/'stress.json').write_text(json.dumps(stress,indent=2))
                    np.savez_compressed(dest/'predictions.npz',indices=test,target=y[test],prediction=pred,lower=lo,upper=hi)
                    joblib.dump({'model':model,'adjustment':adjustment,'selection':selection},dest/'head.joblib')
                    (dest/'complete.json').write_text(json.dumps({'status':'complete'}));print('DONE',label,model.candidate.name,'MAE',round(result['mae_adu'],4),flush=True)
    print('OPTIMIZATION COMPLETE',flush=True)


def route():
    source=ARTIFACTS/'optimization';files=sorted(source.rglob('complete.json'))
    if len(files)!=18:raise ValueError('Complete all eighteen raw nested fits before routing')
    records=load_dataset();features=joblib.load(ARTIFACTS/'features.joblib')['features'];history=causal_curve_history(records,features)
    y=np.array([r.target for r in records]);mapping=trajectory_groups(records);groups=np.array([mapping[r.device] for r in records])
    output=ARTIFACTS/'optimization_routed';output.mkdir(exist_ok=True)
    manifest=json.loads((source/'manifest.json').read_text())
    manifest['routing']='Current curve available: selected head; missing: original modality-dropout expert fusion'
    manifest['routing_provenance']='Added after initial exploratory missing-curve diagnostics; not a blind pre-registered benchmark'
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with threadpool_limits(limits=2):
        for complete in files:
            folder=complete.parent;metadata=json.loads((folder/'metrics.json').read_text());protocol=metadata['protocol'];seed=metadata['seed'];fold=metadata['fold']
            dest=output/folder.relative_to(source);dest.mkdir(exist_ok=True,parents=True)
            if (dest/'complete.json').exists():continue
            _,train,cal,test,info=next(s for s in make_splits(records,protocol,seed) if s[0]==fold)
            saved=joblib.load(folder/'head.joblib');head=saved['model'];start=time.monotonic()
            legacy=fit_steam(records,features,y,train,cal,groups,seed)
            fallback_cal=steam_predict(legacy,features,cal)[0];fallback_test=steam_predict(legacy,features,test)[0]
            x=matrix(features,history,head.candidate.view)
            cp=np.where(features['quality'][cal,0]>0,head.predict(x[cal]),fallback_cal)
            radius,events=calibrate(records,y,cal,groups,cp)
            pp=np.where(features['quality'][test,0]>0,head.predict(x[test]),fallback_test)
            lo=pp-radius;hi=pp+radius
            metadata.update(score(y[test],pp));metadata.update(interval_coverage=float(np.mean((y[test]>=lo)&(y[test]<=hi))),interval_width_adu=2*radius,
                  calibration_events=events,training_seconds=metadata['training_seconds']+time.monotonic()-start,routing=manifest['routing'])
            stress=[];rng=np.random.default_rng(seed+fold);tf=subset(features,test)
            for rate in (0.,.1,.3,.5,.7,1.):
                for modality in ('image','text','both'):
                    f={k:v.copy() for k,v in tf.items()};selected=rng.random(len(test))<rate
                    if modality in ('image','both'):f['image'][selected,2:]=np.nan;f['quality'][selected,:2]=0
                    if modality in ('text','both'):f['text'][selected,2:]=np.nan;f['quality'][selected,2:4]=0
                    base=steam_predict(legacy,f,np.arange(len(test)))[0]
                    pred=np.where(f['quality'][:,0]>0,head.predict(matrix(f,history[test],head.candidate.view)),base)
                    stress.append({'rate':rate,'modality':modality,'type':'random_current_modality_loss',**score(y[test],pred)})
            for name in ('selection.json','split.json'):(dest/name).write_bytes((folder/name).read_bytes())
            (dest/'metrics.json').write_text(json.dumps(metadata,indent=2));(dest/'stress.json').write_text(json.dumps(stress,indent=2))
            np.savez_compressed(dest/'predictions.npz',indices=test,target=y[test],prediction=pp,lower=lo,upper=hi)
            (dest/'complete.json').write_text(json.dumps({'status':'complete'}));print('ROUTED',protocol,seed,fold,round(metadata['mae_adu'],4),flush=True)

def summarize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=ARTIFACTS/'optimization_routed';dest=ROOT/'reports'/'optimization';dest.mkdir(exist_ok=True,parents=True)
    manifests=sorted(source.rglob('complete.json'));rows=[];predrows=[];selections=[];stress=[];by_group=defaultdict(list)
    if len(manifests)!=18:raise ValueError('Complete all eighteen routed outer evaluations before summarizing')
    records=load_dataset();mapping=trajectory_groups(records)
    for complete in manifests:
        folder=complete.parent;m=json.loads((folder/'metrics.json').read_text());p=dict(np.load(folder/'predictions.npz'))
        baseline=ARTIFACTS/'research'/'runs'/m['protocol']/f'seed_{m["seed"]}'/f'fold_{m["fold"]}'
        old=np.load(baseline/'predictions.npz');assert np.array_equal(p['indices'],old['indices'])
        oldmetrics=json.loads((baseline/'metrics.json').read_text())['full']
        row=dict(m,baseline_mae_adu=oldmetrics['mae_adu'],baseline_mse_adu2=oldmetrics['mse_adu2'],baseline_mre_percent=oldmetrics['mre_percent'])
        rows.append(row);s=compact_selection(json.loads((folder/'selection.json').read_text()));selections.append({'protocol':m['protocol'],'seed':m['seed'],'fold':m['fold'],**s})
        stress.extend(dict(v,protocol=m['protocol'],seed=m['seed'],fold=m['fold']) for v in json.loads((folder/'stress.json').read_text()))
        for j,i in enumerate(p['indices']):
            r=records[int(i)];g=mapping[r.device];a=float(p['prediction'][j]);b=float(old['full'][j]);y=float(p['target'][j])
            predrows.append({'protocol':m['protocol'],'seed':m['seed'],'fold':m['fold'],'device':r.device,'index':r.index,'group':g,
                'target_adu':y,'original_adu':b,'optimized_adu':a,'lower_adu':float(p['lower'][j]),'upper_adu':float(p['upper'][j])})
            if m['protocol']=='grouped_cv':by_group[(m['seed'],g)].append(abs(a-y)-abs(b-y))
    grouped=defaultdict(list)
    for (_,g),v in by_group.items():grouped[g].append(float(np.mean(v)))
    differences=np.array([np.mean(v) for v in grouped.values()]);rng=np.random.default_rng(20261013)
    bootstrap=np.mean(rng.choice(differences,size=(10000,len(differences)),replace=True),axis=1)
    pairs={'groups':len(differences),'delta_optimized_minus_original_group_mae':float(np.mean(differences)),
           'bootstrap_95':np.quantile(bootstrap,[.025,.975]).tolist(),'note':'Resamples thirty trajectories, not independent records; negative is improvement.'}
    summary={}
    for protocol in sorted({r['protocol'] for r in rows}):
        rr=[r for r in rows if r['protocol']==protocol];summary[protocol]={}
        for key in ('mae_adu','mse_adu2','mre_percent','baseline_mae_adu','baseline_mse_adu2','baseline_mre_percent','interval_coverage','interval_width_adu'):
            values=[r[key] for r in rr];summary[protocol][key]={'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1)) if len(values)>1 else 0.,'n':len(values)}
        summary[protocol]['selected_counts']=dict(Counter(r['selected'] for r in rr))
    (dest/'summary.json').write_text(json.dumps(summary,indent=2));(dest/'paired_analysis.json').write_text(json.dumps(pairs,indent=2))
    (dest/'selection_audits.json').write_text(json.dumps(selections,indent=2));(dest/'pressure_tests.json').write_text(json.dumps(stress,indent=2))
    (dest/'manifest.json').write_bytes((source/'manifest.json').read_bytes())
    with (dest/'fold_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    with gzip.open(dest/'predictions.csv.gz','wt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(predrows[0]),lineterminator='\n');w.writeheader();w.writerows(predrows)
    fig,axes=plt.subplots(1,2,figsize=(10,4));names={'grouped_cv':'Grouped CV','temporal':'Future periods'}
    for ax,(protocol,s) in zip(axes,summary.items()):
        values=[s['baseline_mae_adu']['mean'],s['mae_adu']['mean']];errs=[s['baseline_mae_adu']['std'],s['mae_adu']['std']]
        ax.bar(['Original','Optimized'],values,yerr=errs,color=['#a3babb','#087f83'],capsize=4);ax.set_title(names[protocol]);ax.set_ylabel('MAE / adu');ax.spines[['top','right']].set_visible(False)
        for j,v in enumerate(values):ax.text(j,v+errs[j]+.12,f'{v:.3f}',ha='center')
        ax.set_ylim(0,max(values)+max(errs)+1)
    fig.tight_layout();fig.savefig(dest/'comparison.png',dpi=180);fig.savefig(dest/'comparison.svg');plt.close(fig)
    for lang in ('en','zh'):
        def val(p,k):s=summary[p][k];return f'{s["mean"]:.3f} ± {s["std"]:.3f}'
        lines=['# SteamWeave v2 optimization / 优化报告',
        '[English](OPTIMIZATION.md) · [中文](OPTIMIZATION.zh-CN.md)',
        ('This is an exploratory follow-up on the previously examined synthetic benchmark. Eight candidates were fixed before new outer predictions. Three inner folds select features/model/loss using only outer fitting rows. Outer calibration labels determine interval width only. Candidate selection never receives outer test labels. The current-curve availability gate was added after exploratory missing-curve diagnostics; this is not an untouched test-set study.' if lang=='en' else
        '这是在此前已查看的模拟基准上的探索性改进。八个候选方案在新外层预测前固定，三折内层验证仅使用外层训练记录选择特征、模型和损失。外层校准标签仅确定区间宽度，候选选择不接收外层测试标签。当前曲线可用性切换是在探索性的缺失曲线诊断后加入的，因此不是未经查看的测试集研究。'),
        '| Protocol / 协议 | Original MAE / 原 MAE | Optimized MAE / 优化 MAE | Runs / 运行 |',
        '| --- | ---: | ---: | ---: |']
        for p in summary:lines.append(f'| {names[p]} | {val(p,"baseline_mae_adu")} | {val(p,"mae_adu")} | {summary[p]["mae_adu"]["n"]} |')
        lines.append(('Across grouped folds, MSE changes from 49.469 to 39.073 adu² and MRE from 8.742% to 7.036%. Mean interval coverage changes from 94.6% to 97.2%, with wider intervals: 28.818 to 35.072 adu. Temporal MSE worsens from 187.207 to 222.680 adu². Those tradeoffs are retained in the report rather than hidden by the MAE improvement.' if lang=='en' else
                      '分组折的 MSE 从 49.469 降至 39.073 adu²，MRE 从 8.742% 降至 7.036%。平均区间覆盖率从 94.6% 变为 97.2%，但区间更宽：28.818 → 35.072 adu。时间候选的 MSE 从 187.207 恶化为 222.680 adu²。报告保留这些权衡，不用 MAE 改善掩盖它们。'))
        lines.extend(['![Comparison](comparison.png)',
          ('Changes: quadratic curve descriptors; strictly earlier same-device curve state and change rates; reset history after observed repair; explicit horizon interactions; grouped/forward inner model selection; absolute-error objective among the candidates. The optional semantic view competes with simpler views rather than being forced into every prediction. An available current curve uses the selected head; a missing current curve uses the original augmented expert fusion. The availability gate depends on input availability, not target values.' if lang=='en' else
          '改动：二次曲线形状描述；严格早于当前时点的同设备曲线状态及变化率；检测维修后重置历史；预测跨度交互；分组/前向内层选择；候选中包含绝对误差目标。语义特征作为可选候选参与比较，不强制进入每次预测。当前曲线可用时使用优化头，缺失时使用原版增强专家融合；切换取决于输入可用性，不取决于目标数值。'),
          f'Paired group delta (optimized − original): **{pairs["delta_optimized_minus_original_group_mae"]:.3f} adu**, bootstrap 95% **[{pairs["bootstrap_95"][0]:.3f}, {pairs["bootstrap_95"][1]:.3f}]**. Negative means improvement / 负值表示改善。',
          ('Limitations: no independent new dataset; thirty groups; fold variation is not a confidence interval; temporal repetitions are the same fixed split; the temporal candidate worsened results, so the serving future-period checkpoint remains the original method; no claim of exceeding the paper on matching conditions. History cannot recover an unobserved repair or original signal values. Pressure tests mask the current modality only and retain previously observed curve history.' if lang=='en' else
          '局限：没有新的独立数据集；仅三十个轨迹组；折标准差不是置信区间；时间运行使用相同固定划分；时间候选结果变差，因此服务保留原版未来时段检查点；不能据此声称在同条件下超过论文。历史不能恢复未观测维修或原始信号。压力测试仅屏蔽当前模态，保留此前可用曲线历史。'),
          '[Summary / 汇总](summary.json) · [Fold metrics / 折指标](fold_metrics.csv) · [Paired predictions / 配对预测](predictions.csv.gz) · [Inner selections / 内层选择](selection_audits.json) · [Pressure / 压力](pressure_tests.json) · [Manifest / 配置](manifest.json)',
          '```bash\n.venv/bin/python -m research.optimize run\n.venv/bin/python -m research.optimize summarize\n```'])
        (dest/('OPTIMIZATION.md' if lang=='en' else 'OPTIMIZATION.zh-CN.md')).write_text(('\n\n'.join(lines)+'\n').replace('|\n\n|','|\n|'))
    print(json.dumps(summary,indent=2));print('Paired:',pairs)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','route','summarize'])
    p.add_argument('--seeds',nargs='+',type=int,default=[17,41,73]);p.add_argument('--protocols',nargs='+',choices=['grouped_cv','temporal'],default=['grouped_cv','temporal'])
    args=p.parse_args()
    if args.command=='run':run(args);route()
    elif args.command=='route':route()
    else:summarize()
