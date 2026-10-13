"""Build public result tables, paired group analyses and standalone scientific figures."""
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sg_agents.config import ROOT,ARTIFACTS
from sg_agents.data import load_dataset,trajectory_groups

LABELS={
 'full':'SteamWeave · full','direct_feature_fusion':'Direct feature fusion',
 'fixed_expert_average':'Fixed expert average','rf_numeric_100':'RF · numeric (100 trees)',
 'svr_numeric_c100':'SVR · numeric (C=100)','paper_dnn_reconstruction':'DNN reconstruction · budgeted',
 'without_maintenance_agent':'No maintenance expert','without_freshness':'No freshness metadata',
 'without_llm_semantics':'No LLM semantics','without_modality_dropout':'No modality dropout',
 'without_text_agent':'No current text expert','without_image_agent':'No current curve expert'}
MAIN=['full','direct_feature_fusion','fixed_expert_average','rf_numeric_100','svr_numeric_c100','paper_dnn_reconstruction']
ABLATIONS=[k for k in LABELS if k.startswith('without_')]

def stats(values):
    a=np.asarray(values,float)
    return {'mean':float(a.mean()),'std':float(a.std(ddof=1)) if len(a)>1 else 0.,'n':len(a)}

def paired_groups(runs,model,mapping,records):
    samples=defaultdict(list)
    for protocol,seed,fold,metrics,predictions,_ in runs:
        if protocol!='grouped_cv' or model not in metrics:continue
        ids=predictions['indices'];y=predictions['target'];groups=np.array([mapping[records[i].device] for i in ids])
        full_err=np.abs(predictions['full']-y);other_err=np.abs(predictions[model]-y)
        for g in np.unique(groups):samples[int(g)].append(float(np.mean(other_err[groups==g])-np.mean(full_err[groups==g])))
    differences=np.array([np.mean(samples[g]) for g in sorted(samples)])
    rng=np.random.default_rng(20261013)
    boots=np.mean(rng.choice(differences,size=(10000,len(differences)),replace=True),axis=1)
    p=1. if np.allclose(differences,0) else float(wilcoxon(np.round(differences,10),alternative='two-sided',method='auto').pvalue)
    return {'model':model,'groups':len(differences),'delta_other_minus_full_adu':float(differences.mean()),
            'bootstrap_95':np.quantile(boots,[.025,.975]).tolist(),'signed_rank_p':p,
            'group_differences':{str(g):float(np.mean(samples[g])) for g in sorted(samples)},
            'sign_convention':'Positive means the comparator has higher error than full SteamWeave.'}

def table(summary,protocol,models=MAIN):
    rows=['| Model | MAE / adu | MSE / adu² | MRE / % | Folds / runs |','| --- | ---: | ---: | ---: | ---: |']
    for model in models:
        m=summary[protocol].get(model)
        if not m:continue
        def cell(k):return f'{m[k]["mean"]:.3f} ± {m[k]["std"]:.3f}'
        rows.append(f'| {LABELS[model]} | {cell("mae_adu")} | {cell("mse_adu2")} | {cell("mre_percent")} | {m["mae_adu"]["n"]} |')
    return '\n'.join(rows)

def plot_results(summary,pairs,stress,records,runs,dest):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.labelcolor':'#315460','text.color':'#23424d','axes.edgecolor':'#c1d1d1','xtick.color':'#617b83','ytick.color':'#617b83',
        'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none'})
    def save(fig,name):
        fig.savefig(dest/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(dest/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    models=[k for k in MAIN if k in summary['grouped_cv']]
    fig,ax=plt.subplots(figsize=(10.5,4.3));values=[summary['grouped_cv'][k]['mae_adu']['mean'] for k in models]
    errors=[summary['grouped_cv'][k]['mae_adu']['std'] for k in models]
    yy=np.arange(len(models));ax.barh(yy,values,xerr=errors,color=['#087f83' if k=='full' else '#9eb8bb' for k in models],height=.6,capsize=3)
    ax.set_yticks(yy,[LABELS[k] for k in models]);ax.invert_yaxis();ax.set_xlabel('MAE / adu · mean ± fold standard deviation')
    ax.set_title('Shared duplicate-aware outer test groups',loc='left',fontweight='bold',pad=18)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();save(fig,'model_comparison')
    variants=[k for k in ABLATIONS if k in pairs]
    fig,ax=plt.subplots(figsize=(10.5,4.2));d=[pairs[k]['delta_other_minus_full_adu'] for k in variants]
    ci=np.array([pairs[k]['bootstrap_95'] for k in variants]);yy=np.arange(len(variants))
    ax.errorbar(d,yy,xerr=np.array([np.asarray(d)-ci[:,0],ci[:,1]-np.asarray(d)]),fmt='o',color='#147d7a',capsize=4)
    ax.axvline(0,color='#bfa17b',ls='--',lw=1);ax.set_yticks(yy,[LABELS[k] for k in variants]);ax.invert_yaxis()
    ax.set_xlabel('Comparator − full MAE / adu · paired trajectory bootstrap 95% interval')
    ax.set_title('Ablations can help or hurt',loc='left',fontweight='bold',pad=18);ax.grid(axis='x',alpha=.15)
    fig.tight_layout();save(fig,'ablation_paired')
    fig,axes=plt.subplots(1,3,figsize=(13.2,3.8),sharey=True)
    for ax,modality in zip(axes,['image','text','both']):
        for variant,color,linestyle in [('full','#087f83','-'),('without_modality_dropout','#c78b4e','--')]:
            subset=[v for v in stress if v['variant']==variant and v['type']=='random_modality_loss' and v['modality']==modality]
            rates=sorted({v['rate'] for v in subset});means=[np.mean([v['mae_adu'] for v in subset if v['rate']==rate]) for rate in rates]
            ax.plot(np.array(rates)*100,means,'o',color=color,ls=linestyle,label=LABELS[variant],ms=4)
        ax.set_title({'image':'Current curve loss','text':'Current text loss','both':'Both modalities lost'}[modality]);ax.set_xlabel('Artificial loss rate / %');ax.grid(alpha=.15)
    axes[0].set_ylabel('MAE / adu');axes[0].legend(fontsize=8);fig.tight_layout();save(fig,'missing_modality_robustness')
    fig,axes=plt.subplots(1,2,figsize=(11,3.7))
    protocols=['paper_device_cv','grouped_cv','temporal'];v=[summary[p]['full']['mae_adu']['mean'] for p in protocols]
    axes[0].bar(['Directory-ID\ndiagnostic','Trajectory-group\nCV','Future-period\ntest'],v,color=['#a1b4bb','#147d7a','#c78b4e'])
    axes[0].set_ylabel('MAE / adu');axes[0].set_title('Protocols answer different questions',loc='left',fontweight='bold');axes[0].grid(axis='y',alpha=.15);axes[0].set_axisbelow(True)
    nominal=.9;coverage=[summary[p]['full']['interval_coverage']['mean'] for p in protocols]
    axes[1].bar(['ID diagnostic','Grouped CV','Temporal'],np.array(coverage)*100,color=['#a1b4bb','#147d7a','#c78b4e'])
    axes[1].axhline(nominal*100,color='#5d727d',ls='--',lw=1,label='90% nominal');axes[1].set_ylabel('Empirical interval coverage / %');axes[1].set_ylim(0,105);axes[1].legend(fontsize=8);axes[1].grid(axis='y',alpha=.15);axes[1].set_axisbelow(True)
    fig.tight_layout();save(fig,'protocols_and_coverage')
    data=next(p for protocol,seed,fold,metrics,p,_ in runs if protocol=='grouped_cv' and seed==17 and fold==0)
    y=data['target'];p=data['full'];fig,axes=plt.subplots(1,2,figsize=(10.5,4.1))
    axes[0].scatter(y,p,s=10,c='#087f83',alpha=.25,edgecolors='none');limit=max(y.max(),p.max());axes[0].plot([0,limit],[0,limit],color='#b3a28f',ls='--');axes[0].set_xlabel('Target / adu');axes[0].set_ylabel('Forecast / adu');axes[0].set_title('Held-out predictions · seed 17, fold 0')
    error=p-y;axes[1].hist(error,bins=40,color='#98b9b8',edgecolor='white');axes[1].axvline(0,c='#147d7a',lw=1.5);axes[1].set_xlabel('Forecast − target / adu');axes[1].set_ylabel('Observations');axes[1].set_title('Residual distribution')
    fig.tight_layout();save(fig,'predictions_and_residuals')
    delays=[v for v in stress if v['type']=='synthetic_report_delay' and v['variant']=='full'];lags=sorted({v['lag_atu'] for v in delays})
    fig,ax=plt.subplots(figsize=(7,3.3));ax.plot(lags,[np.mean([v['mae_adu'] for v in delays if v['lag_atu']==lag]) for lag in lags],'o-',c='#147d7a');ax.set_xlabel('Synthetic report retrieval delay / atu');ax.set_ylabel('MAE / adu');ax.set_title('Delayed evidence sensitivity',loc='left',fontweight='bold');ax.grid(alpha=.15);fig.tight_layout();save(fig,'report_delay')

def main():
    report_dir=ROOT/'reports';figures=report_dir/'figures';figures.mkdir(exist_ok=True,parents=True)
    source=ARTIFACTS/'research'/'runs';files=sorted(source.rglob('complete.json'))
    if len(files)!=23:raise ValueError(f'Expected 23 completed outer evaluations, found {len(files)}')
    records=load_dataset();mapping=trajectory_groups(records);runs=[];buckets=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    stress=[];histories=[];export_rows=[]
    for complete in files:
        folder=complete.parent;protocol=folder.parents[1].name;seed=int(folder.parent.name.split('_')[1]);fold=int(folder.name.split('_')[1])
        metrics=json.loads((folder/'metrics.json').read_text());preds=dict(np.load(folder/'predictions.npz'))
        info=json.loads((folder/'split.json').read_text());runs.append((protocol,seed,fold,metrics,preds,info))
        for model,values in metrics.items():
            for key,value in values.items():
                if isinstance(value,(int,float)):buckets[protocol][model][key].append(value)
            export_rows.append({'protocol':protocol,'seed':seed,'fold':fold,'model':model,**values})
        for variant,name in [('full','stress.json'),('without_modality_dropout','stress_without_modality_dropout.json')]:
            p=folder/name
            if p.exists():
                ss=json.loads(p.read_text())
                if not any(v['type']=='synthetic_report_delay' for v in ss):raise ValueError('Re-run research.extend_stress to complete report-delay outputs')
                stress.extend(dict(v,variant=variant,seed=seed,fold=fold) for v in ss)
        if (folder/'neural_training.json').exists():
            histories.append({'protocol':protocol,'seed':seed,'fold':fold,**json.loads((folder/'neural_training.json').read_text())})
    summary={p:{m:{k:stats(v) for k,v in vals.items()} for m,vals in models.items()} for p,models in buckets.items()}
    for protocol,models in summary.items():
        for model,m in models.items():
            seed_values=defaultdict(list);group_values=defaultdict(list)
            for pp,seed,fold,metrics,preds,_ in runs:
                if pp!=protocol or model not in metrics:continue
                seed_values[seed].append(metrics[model]['mae_adu'])
                ids=preds['indices'];g=np.array([mapping[records[i].device] for i in ids]);err=np.abs(preds[model]-preds['target'])
                for gg in np.unique(g):group_values[(seed,int(gg))].append(float(err[g==gg].mean()))
            m['seed_mean_mae_adu']={str(s):float(np.mean(v)) for s,v in seed_values.items()}
            m['seed_mean_mae_std']=stats(list(m['seed_mean_mae_adu'].values()))
            m['trajectory_macro_mae_adu']=float(np.mean([np.mean(v) for v in group_values.values()]))
    pairs={m:paired_groups(runs,m,mapping,records) for m in MAIN[1:]+ABLATIONS}
    order=sorted(ABLATIONS,key=lambda k:pairs[k]['signed_rank_p']);last=0.
    for j,k in enumerate(order):
        last=max(last,min(1.,pairs[k]['signed_rank_p']*(len(order)-j)));pairs[k]['holm_adjusted_p']=last
    meta=json.loads((ARTIFACTS/'research'/'manifest.json').read_text())
    meta.update(completed_outer_evaluations=len(runs),completed_model_evaluations=len(export_rows),pressure_test_cases=len(stress),
                neural_fits=len(histories),neural_budget_cap_fits=sum(h['epochs_run']==h['assumptions']['max_epochs'] for h in histories))
    for name,data in [('summary.json',summary),('paired_analysis.json',pairs),('pressure_tests.json',stress),('neural_training_histories.json',histories),('experiment_manifest.json',meta)]:
        (report_dir/name).write_text(json.dumps(data,ensure_ascii=False,indent=2))
    with (report_dir/'fold_metrics.csv').open('w',newline='') as f:
        columns=['protocol','seed','fold','model','mae_adu','mse_adu2','rmse_adu','mre_percent','records','training_seconds','interval_coverage','interval_width_adu']
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(export_rows)
    with gzip.open(report_dir/'predictions.csv.gz','wt',newline='') as f:
        columns=['protocol','seed','fold','model','device','index','trajectory_group','observed_at','next_inspection_at','target_adu','prediction_adu','lower_adu','upper_adu']
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for protocol,seed,fold,metrics,preds,_ in runs:
            for model in metrics:
                for j,i in enumerate(preds['indices']):
                    r=records[int(i)];row={'protocol':protocol,'seed':seed,'fold':fold,'model':model,'device':r.device,'index':r.index,
                         'trajectory_group':mapping[r.device],'observed_at':r.time,'next_inspection_at':r.next_inspection,
                         'target_adu':float(preds['target'][j]),'prediction_adu':float(preds[model][j]),
                         'lower_adu':float(preds['lower'][j]) if model=='full' else '',
                         'upper_adu':float(preds['upper'][j]) if model=='full' else ''}
                    writer.writerow(row)
    raw=report_dir/'run_manifests';raw.mkdir(exist_ok=True)
    for complete in files:
        folder=complete.parent;rel=folder.relative_to(source);dest=raw/rel;dest.mkdir(parents=True,exist_ok=True)
        for name in ('split.json','nested_oof.json','complete.json'):
            if (folder/name).exists():shutil.copyfile(folder/name,dest/name)
    plot_results(summary,pairs,stress,records,runs,figures)
    primary=summary['grouped_cv']['full']['mae_adu']['mean'];temporal=summary['temporal']['full']['mae_adu']['mean']
    ablation_summary={'grouped_cv':dict(summary['grouped_cv'])}
    ablation_summary['grouped_cv']['full']={key:stats([metrics['full'][key] for protocol,seed,fold,metrics,_,_ in runs if protocol=='grouped_cv' and seed==17]) for key in ('mae_adu','mse_adu2','mre_percent')}
    diag=summary['paper_device_cv']['full']['mae_adu']['mean']
    ptable=['| Comparator removed | Δ MAE / adu | Bootstrap 95% | Holm p |','| --- | ---: | --- | ---: |']
    for k in ABLATIONS:
        p=pairs[k];ptable.append(f'| {LABELS[k]} | {p["delta_other_minus_full_adu"]:.3f} | [{p["bootstrap_95"][0]:.3f}, {p["bootstrap_95"][1]:.3f}] | {p["holm_adjusted_p"]:.4g} |')
    pair_table='\n'.join(ptable)
    english=f'''# SteamWeave experiment report

[中文](EXPERIMENTS.zh-CN.md) · [Protocol](../docs/EXPERIMENT_PROTOCOL.md) · [Raw metrics](fold_metrics.csv) · [Paired predictions](predictions.csv.gz)

## Scope and result provenance

Completed **{len(runs)} outer evaluations**, **{len(export_rows)} model evaluations**, **{len(stress)} pressure-test cases** and **{len(histories)} neural fits**. Grouped CV and temporal supervised comparisons use seeds 17/41/73. Neural reconstruction uses only seed 17. Six ablations and pressure tests use the complete five grouped folds of seed 17. Standard deviations below describe fold/run variation, not confidence intervals.

The original paper reports MAE 3.20 adu, MSE 45.57 adu² and MRE 6.76% for its three-branch model ([Table 3](https://www.researchgate.net/publication/349102644_A_multi-branch_deep_neural_network_model_for_failure_prognostics_based_on_multimodal_data)). This literature row is not merged into the local ranking: original splits, data realization, optimization and hardware are not fully matched. The local DNN is an architecture reconstruction with disclosed training choices, not the authors' model checkpoint.

## Primary grouped comparison

{table(summary,'grouped_cv')}

![Comparison](figures/model_comparison.png)

The full system's fold-mean MAE is **{primary:.3f} adu**. Group-macro MAE is {summary['grouped_cv']['full']['trajectory_macro_mae_adu']:.3f} adu. All methods share held-out input-trajectory groups and development boundaries. Neural numbers use five fits versus fifteen for the multiseed classical/system comparisons.

## Ablations and paired uncertainty

{table(ablation_summary,'grouped_cv',['full']+ABLATIONS)}

{pair_table}

Positive delta means removing the component worsens MAE; negative delta means the simpler comparator is better. Confidence intervals resample thirty trajectory groups. Signed-rank tests are paired on group-level errors; Holm correction covers the six ablations. Inference assumes the groups are adequate sampling units and should not be generalized beyond this synthetic copy.

![Ablations](figures/ablation_paired.png)

Current report content and LLM semantic features do not have an assumed accuracy benefit. The no-text ablation retains original numeric and earlier timing/maintenance context; it is not the elimination of all text-derived history. No test-set ranking is used to silently replace the production model.

## Missing evidence and delayed reports

![Missing evidence](figures/missing_modality_robustness.png)

Loss masks are paired between the full model and the no-dropout ablation. Artificial loss rates cover 0–100%, with current curve, text or both removed. Trailing contiguous curve outages cover 30/50/70%. Natural missing curves, reused reports, new repairs and horizon bins are reported separately in [pressure_tests.json](pressure_tests.json).

![Delayed reports](figures/report_delay.png)

Report delays of 10/25/50 atu use earlier observed text only, adjusting the observable age context. They are synthetic perturbations. The local data audit found zero naturally observed repair-after-text-repeat examples, so such interaction effects are not claimed as empirically established from natural data.

## Future periods and diagnostic directory-ID splits

{table(summary,'temporal')}

Temporal MAE is **{temporal:.3f} adu**. Three seeded full-model fits have identical point results because the configured histogram learner is deterministic on this fixed temporal partition; they are not independent datasets.

{table(summary,'paper_device_cv')}

Directory-ID diagnostic MAE is **{diag:.3f} adu**. Identical input trajectories can cross training/test boundaries here, and random record validation differs from grouped development. The lower error is not an independence-controlled generalization result, nor a reproduction of the paper's original data realization.

![Protocols and coverage](figures/protocols_and_coverage.png)

Coverage is empirical and widths are recorded in the JSON/CSV outputs. Calibration is event-based but observations remain dependent. Training times exclude one-time API/feature preparation; cross-hardware speedup relative to the published CPU experiment is not inferred.

## Neural reconstruction budget

The image/TextCNN/numeric dimensions follow the consulted architecture description; missing optimizer, preprocessing and schedule choices are disclosed in each training history. Of {len(histories)} fits, **{meta['neural_budget_cap_fits']} reached the 80-epoch cap**. Validation selects the best checkpoint within that budget. These results measure this implementation and budget, not the optimized performance ceiling of the paper's approach. Extra neural seeds and higher budgets remain follow-up work.

## Reproducibility artifacts

- [Frozen manifest](experiment_manifest.json), [data audit](data_audit.json), [paper reference](paper_reference.json).
- [Summary](summary.json), [fold-level metrics](fold_metrics.csv), [paired analysis](paired_analysis.json).
- [All unperturbed paired predictions](predictions.csv.gz), [split/inner-fold manifests](run_manifests/).
- [Pressure conditions and metrics](pressure_tests.json), [neural epoch histories](neural_training_histories.json).
- PNG and editable SVG figures in [figures/](figures/).

## Interpretation

Multimodal availability, report timing, maintenance context and split design should be studied independently. A component's role in the architecture is not proof of its predictive value. This experiment supports reproducible comparisons on a synthetic benchmark with duplicate-aware accounting, not a claim of industrial deployment accuracy or universal superiority over the literature.
'''
    chinese=f'''# SteamWeave 实验报告

[English](EXPERIMENTS.md) · [实验协议](../docs/EXPERIMENT_PROTOCOL.md) · [原始指标](fold_metrics.csv) · [配对预测](predictions.csv.gz)

## 范围与结果来源

实际完成 **{len(runs)} 次外层评估、{len(export_rows)} 个模型评估、{len(stress)} 个压力测试条件、{len(histories)} 次深度网络拟合**。轨迹组与时间实验的监督模型采用 17/41/73 三个种子；深度重建仅使用种子 17。六项消融和压力测试覆盖种子 17 的完整五折。下表标准差表示折/运行差异，不是置信区间。

原论文三分支模型报告 MAE 3.20 adu、MSE 45.57 adu²、MRE 6.76%（[表 3](https://www.researchgate.net/publication/349102644_A_multi-branch_deep_neural_network_model_for_failure_prognostics_based_on_multimodal_data)）。此文献值不并入本地模型排名：原始划分、数据实现、优化和硬件未完整匹配。本地 DNN 为已披露训练选择的结构重建，不是作者检查点。

## 主要轨迹组比较

{table(summary,'grouped_cv')}

![模型对比](figures/model_comparison.png)

完整系统折平均 MAE 为 **{primary:.3f} adu**，轨迹组宏平均 MAE 为 {summary['grouped_cv']['full']['trajectory_macro_mae_adu']:.3f} adu。模型使用同一留出轨迹组与开发边界；深度重建为五次拟合，三种子系统/传统基线为十五次。

## 消融与配对不确定性

{table(ablation_summary,'grouped_cv',['full']+ABLATIONS)}

{pair_table}

正差值表示移除模块使误差增加；负差值表示简化方案更好。置信区间按三十个轨迹组重采样，符号秩检验基于逐组配对误差，六项消融使用 Holm 校正。统计推断依赖轨迹组作为抽样单位的合理性，不能外推为真实工业结论。

![消融](figures/ablation_paired.png)

当前报告内容和 LLM 语义不预设准确率收益。无当前文本专家仍保留原始数值、过去时间与维护上下文，不能解读为删除全部文本历史。系统不会根据测试集排名自动替换界面模型。

## 缺失证据与报告延迟

![缺失证据](figures/missing_modality_robustness.png)

完整模型与无模态增强模型使用配对缺失掩码。当前曲线、文本、两者同时缺失覆盖 0–100%；连续曲线中断覆盖每台设备测试记录末段的 30/50/70%。自然缺失、旧文本、新维修、预测跨度分组见 [pressure_tests.json](pressure_tests.json)。

![报告延迟](figures/report_delay.png)

10/25/50 atu 延迟只读取较早观察的文本，并调整观测年龄。这属于合成扰动。本地审计未发现自然的“连续沿用文本之后发生维修”样本，因此不声称这种交互已通过自然样本验证。

## 未来时段与设备 ID 诊断

{table(summary,'temporal')}

时间外推 MAE 为 **{temporal:.3f} adu**。完整模型三个种子的点预测相同，原因是固定时间分区上的直方图学习器为确定性实现；三次运行不等于三个独立数据集。

{table(summary,'paper_device_cv')}

设备 ID 诊断 MAE 为 **{diag:.3f} adu**。这个协议可能让重复轨迹跨越训练与测试，随机记录验证也不同于轨迹组开发边界。较低误差不属于独立性受控的泛化结果，也不等同于原论文数据实现的数值复现。

![协议与覆盖率](figures/protocols_and_coverage.png)

覆盖率为实测值，区间宽度保存在 JSON/CSV 中。按事件校准不消除观测之间的相关性。拟合时间排除一次性 API/特征准备，不据此声称相对原论文 CPU 实验的跨硬件加速比。

## 深度重建的预算

图像、TextCNN、数值分支尺寸来自已核对的结构说明；优化器、预处理和计划等补充选择保存于训练历史。{len(histories)} 次拟合中，**{meta['neural_budget_cap_fits']} 次到达 80 轮上限**，并在该预算内按验证集选最佳检查点。这衡量当前实现与预算，不代表论文方法经充分优化后的性能上限。更多深度随机种子与更高预算仍属于后续工作。

## 复现材料

- [固定配置](experiment_manifest.json)、[数据审计](data_audit.json)、[文献参考](paper_reference.json)。
- [汇总](summary.json)、[逐折指标](fold_metrics.csv)、[配对分析](paired_analysis.json)。
- [全部未扰动配对预测](predictions.csv.gz)、[外层与内层划分](run_manifests/)。
- [压力条件与指标](pressure_tests.json)、[深度训练轮次记录](neural_training_histories.json)。
- [figures/](figures/) 中包含 PNG 与可编辑 SVG。

## 解释范围

模态可用性、文本时间、维护上下文和划分方法需要分别研究。架构中的职责不等于预测收益的证明。本实验为含重复轨迹的模拟基准提供可复核比较，不声称真实工业精度或对文献方法的普遍优越性。
'''
    (report_dir/'EXPERIMENTS.md').write_text(english);(report_dir/'EXPERIMENTS.zh-CN.md').write_text(chinese)
    print(json.dumps({'outer_evaluations':len(runs),'model_evaluations':len(export_rows),'pressure_cases':len(stress),
             'grouped_mae':primary,'temporal_mae':temporal,'diagnostic_mae':diag,'neural_cap_fits':meta['neural_budget_cap_fits']},indent=2))

if __name__=='__main__':main()
