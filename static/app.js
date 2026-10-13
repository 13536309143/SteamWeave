'use strict';
const $=s=>document.querySelector(s);
const state={devices:[],records:[],summary:null,current:null,requestId:0,trainingTimer:null};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(n,d=2)=>Number.isFinite(n)?n.toLocaleString('zh-CN',{maximumFractionDigits:d,minimumFractionDigits:d}):'—';
const names={numeric:'数值专家',image:'曲线专家',text:'文本专家',maintenance:'维护时序专家',fusion:'完整融合模型',single_model_direct_fusion:'单模型直接融合',fixed_expert_average:'专家固定平均',no_current_image:'屏蔽当前曲线',no_current_text:'屏蔽当前文本',no_image_or_text:'同时屏蔽曲线和文本',naturally_missing_image:'自然缺失曲线',repeated_text:'旧文本重复',new_repair:'新维修事件',long_horizon:'较长预测跨度'};
const providerName=p=>({minimax:'MiniMax',openai:'OpenAI GPT',qwen:'通义千问',custom:'自定义接口'}[p]||p);
const partitions={test:'留出测试设备',calibration:'区间校准设备',train:'训练设备',custom:'新观察',untrained:'尚未训练'};
function error(message){$('#errorBanner').textContent=message;$('#errorBanner').hidden=false;}
function clearError(){$('#errorBanner').hidden=true;}
async function api(path,options={}){
 const response=await fetch(path,options);
 if(!response.ok){let data;try{data=await response.json();}catch{data={detail:response.statusText};}throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));}
 return response.json();
}
const post=(path,data)=>api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
const busy=(el,on,text)=>{if(on){el.dataset.original=el.textContent;el.textContent=text;el.disabled=true;}else{el.textContent=el.dataset.original||el.textContent;el.disabled=false;}};
function request(){return {device:Number($('#deviceSelect').value),index:Number($('#recordSelect').value),hide_image:$('#hideImage').checked,hide_text:$('#hideText').checked};}
function dl(entries){return entries.map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');}
function table(entries){return `<table><thead><tr><th>模型 / 场景</th><th>MAE / adu</th><th>RMSE / adu</th><th>设备平均 MAE</th><th>区间覆盖率</th><th>记录数</th></tr></thead><tbody>${entries.map(([name,m])=>`<tr><td>${esc(names[name]||name)}</td><td>${fmt(m.mae_adu)}</td><td>${fmt(m.rmse_adu)}</td><td>${fmt(m.device_macro_mae_adu)}</td><td>${m.interval_coverage!=null?fmt(m.interval_coverage*100,1)+'%':'—'}</td><td>${fmt(m.records,0)}</td></tr>`).join('')}</tbody></table>`;}
async function loadSummary(){
 const [s,h,providers]=await Promise.all([api('/api/summary'),api('/api/health'),api('/api/providers')]);state.summary=s;
 const oldProvider=$('#auditProvider').value;
 $('#auditProvider').innerHTML=providers.providers.map(p=>`<option value="${p.provider}">${esc({minimax:'MiniMax',openai:'OpenAI GPT',qwen:'通义千问',custom:'自定义接口'}[p.provider])}${p.credential_configured?'':' · 未配置 Key'}</option>`).join('');
 $('#auditProvider').value=oldProvider||providers.active_audit;
 $('#deviceCount').textContent=fmt(s.devices,0);$('#recordCount').textContent=fmt(s.records,0);$('#imageCount').textContent=fmt(s.image_available,0);$('#missingCount').textContent=`${fmt(s.image_missing,0)} 张空白或不可用曲线`;
 $('#trajectoryCount').textContent=`${s.trajectory_groups} 个输入轨迹组 · 重复轨迹合组`;
 $('#mae').textContent=fmt(s.report.fusion?.mae_adu);$('#connectionStatus').innerHTML=`${s.model_ready?'模型已就绪':'模型训练中'}<small>${h.minimax.connected?providerName(h.provider.provider)+' 接口已验证':providerName(h.provider.provider)+' 尚未验证'}</small>`;
 $('#connectionDot').classList.toggle('bad',!s.model_ready);$('#footerModel').textContent=h.provider.model||'';
 $('#providerDetails').innerHTML=dl([['供应商',providerName(h.provider.provider)],['模型',h.provider.model],['接口地址',h.provider.base_url||'—'],['后端 Key',h.credential_configured?'已配置（不展示）':'未配置'],['唯一报告缓存',`${s.minimax_reports} / ${s.unique_reports}`],['最近连接验证',h.minimax.checked_at?new Date(h.minimax.checked_at).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai'}):'尚未验证']]);
 $('#citation').textContent=`数据集引用：${s.citation}`;
 const r=s.report;
 if(r.fusion){
  const sp=r.split,gc=sp.trajectory_group_counts;$('#splitDescription').textContent=`${gc.train} 个训练轨迹组、${gc.calibration} 个校准组、${gc.test} 个测试组（分别含 ${sp.train_devices.length}、${sp.calibration_devices.length}、${sp.test_devices.length} 个设备目录）。测试设备：${sp.test_devices.join('、')}。`;
  $('#evaluationMetrics').innerHTML=[['MAE / adu',fmt(r.fusion.mae_adu)],['RMSE / adu',fmt(r.fusion.rmse_adu)],['实测区间覆盖率',fmt(r.fusion.interval_coverage*100,1)+'%'],['平均区间宽度 / adu',fmt(r.fusion.interval_width_adu)]].map(([label,value])=>`<div><span>${label}</span><strong>${value}</strong></div>`).join('');
  $('#comparisonTable').innerHTML=table([['fusion',r.fusion],...Object.entries(r.experts),...Object.entries(r.ablation)]);$('#stressTable').innerHTML=table(Object.entries(r.stress_tests));
 }else{$('#splitDescription').textContent='模型尚未训练完成，评估结果将在训练后生成。';}
 const temporal=s.temporal_report;
 if(temporal?.fusion){
  const counts=temporal.split.record_counts;
  $('#temporalDescription').textContent=`按每台设备的目标巡检事件先后切分：${counts.train} 条训练、${counts.calibration} 条校准、${counts.test} 条未来测试记录。训练标签须在截止时点前已获得；此结果与新设备泛化评估回答不同问题。`;
  $('#temporalTable').innerHTML=table([['fusion',temporal.fusion],...Object.entries(temporal.ablation)]);
 }
}
async function loadDevices(){state.devices=await api('/api/devices');$('#deviceSelect').innerHTML=state.devices.map(d=>`<option value="${d.device}">SG ${String(d.device).padStart(2,'0')} · ${partitions[d.partition]}</option>`).join('');const test=state.devices.find(d=>d.partition==='test');if(test)$('#deviceSelect').value=test.device;await loadRecords();}
async function loadRecords(){const device=Number($('#deviceSelect').value);state.records=await api(`/api/devices/${device}/records`);$('#recordSelect').innerHTML=state.records.map(r=>`<option value="${r.index}">t = ${fmt(r.time,0)} atu · 距巡检 ${fmt(r.horizon,0)} atu${!r.image_available?' · 曲线缺失':''}${r.new_repair?' · 维修变化':''}</option>`).join('');const useful=state.records.find(r=>r.index>5&&r.image_available);if(useful)$('#recordSelect').value=useful.index;if(state.summary.model_ready)await runPrediction();}
function renderPrediction(p){
 state.current=p;$('#prediction').textContent=fmt(p.prediction_adu);$('#interval').textContent=`标称 90% 区间 ${fmt(p.interval_90_adu[0])}–${fmt(p.interval_90_adu[1])} adu`;$('#horizon').textContent=`跨度 ${fmt(p.horizon_atu,0)} atu → 巡检 t=${fmt(p.next_inspection_at,0)}`;
 $('#partition').textContent=partitions[p.evaluation_partition]||p.evaluation_partition;
 $('#targetNote').textContent=`数据集目标：${fmt(p.evaluation_target_adu)} adu。${p.evaluation_partition==='train'?'此设备参与训练，当前误差不能代表泛化性能。':'标签仅用于结果对照，不进入本次推理或模型解释。'}`;
 $('#warnings').innerHTML=p.warnings.map(w=>`<p>· ${esc(w)}</p>`).join('');
 $('#curveImage').src=`/api/records/${p.device}/${p.index}/image`;$('#imageTag').textContent=p.image.available?'曲线可用':'曲线未使用';$('#imageTag').classList.toggle('warning',!p.image.available);
 $('#reportText').textContent=p.input.text||'当前文本已屏蔽或为空';
 $('#numericEvidence').innerHTML=dl([['距下一次巡检',fmt(p.input.horizon,0)+' atu'],['距最近维修',fmt(p.input.since_repair,0)+' atu'],['化学清洗次数',fmt(p.input.chemical_count,0)],['机械清洗次数',fmt(p.input.mechanical_count,0)],['反推最近维修时点',fmt(p.maintenance.last_repair_at,0)+' atu'],['文本连续沿用时间',fmt(p.maintenance.text_age,0)+' atu']]);
 const sem=p.text_semantics;const severity=['正常','低','中等','高','很高'];
 $('#semanticEvidence').textContent=sem?`${sem.source} 提取 · 维护：${{none:'无明确维护',mechanical:'机械清洗',chemical:'化学清洗',unknown:'不明确'}[sem.maintenance]} · 维护前：${sem.severity_before==null?'未明确':severity[sem.severity_before]} · 维护后：${sem.severity_after==null?'未明确':severity[sem.severity_after]}。原文证据：${sem.evidence}`:'暂无可用语义提取结果';
 $('#agentTrace').innerHTML=p.trace.map((a,i)=>`<div class="agent-row"><span class="agent-icon">${String(i+1).padStart(2,'0')}</span><div><strong>${esc(a.agent)}</strong><p>${esc(Array.isArray(a.evidence)?a.evidence[0]:a.evidence||(a.status==='unavailable'?'当前模态不可用':'专家模型已完成推理'))}</p></div><span class="agent-value">${a.prediction_adu!=null?fmt(a.prediction_adu)+' adu':a.status==='unavailable'?'未使用':'完成'}</span></div>`).join('');
 $('#explanation').innerHTML='';$('#llmTag').textContent='按需调用';$('#exportLink').href=`/api/export/${p.device}`;
}
function chart(rows){
 if(!rows.length){$('#historyChart').innerHTML='<div class="loading">暂无历史记录</div>';return;}
 const W=720,H=245,L=43,R=14,T=12,B=32;const maxX=Math.max(1,...rows.map(r=>r.time));const vals=rows.flatMap(r=>[r.prediction,r.target,r.lower,r.upper]);const minY=Math.min(0,...vals),maxY=Math.max(1,...vals);const x=v=>L+(W-L-R)*v/maxX,y=v=>T+(H-T-B)*(maxY-v)/(maxY-minY||1);
 const line=key=>rows.map(r=>`${x(r.time).toFixed(2)},${y(r[key]).toFixed(2)}`).join(' ');const area=line('upper')+' '+[...rows].reverse().map(r=>`${x(r.time)},${y(r.lower)}`).join(' ');
 let svg=`<svg viewBox="0 0 ${W} ${H}" aria-label="当前观察及以前的预测与标签"><polygon points="${area}" fill="#0d8c8b" opacity=".07"/>`;
 for(let i=0;i<5;i++){const v=minY+(maxY-minY)*i/4;svg+=`<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}" stroke="#e9efeb"/><text x="${L-8}" y="${y(v)+3}" text-anchor="end" fill="#95a5a2" font-size="9">${fmt(v,0)}</text>`;}
 for(let i=0;i<5;i++){const t=maxX*i/4;svg+=`<text x="${x(t)}" y="${H-10}" text-anchor="middle" fill="#95a5a2" font-size="9">${fmt(t,0)}</text>`;}
 rows.filter(r=>r.new_repair).forEach(r=>{svg+=`<line x1="${x(r.time)}" y1="${T}" x2="${x(r.time)}" y2="${H-B}" stroke="#dda060" stroke-dasharray="3 5" opacity=".5"/>`;});
 svg+=`<polyline points="${line('target')}" fill="none" stroke="#a4b0b6" stroke-width="1.3" stroke-dasharray="5 4"/><polyline points="${line('prediction')}" fill="none" stroke="#098a87" stroke-width="2"/>`;
 rows.forEach(r=>{svg+=`<circle cx="${x(r.time)}" cy="${y(r.prediction)}" r="3" fill="#098a87"><title>观察 t=${r.time} atu；下一巡检预测 ${fmt(r.prediction)} adu；标签 ${fmt(r.target)} adu</title></circle>`;});
 svg+='<text x="14" y="10" fill="#95a5a2" font-size="9">adu</text><text x="705" y="235" fill="#95a5a2" font-size="9">atu</text></svg>';$('#historyChart').innerHTML=svg;
}
async function runPrediction(){
 clearError();const id=++state.requestId;const req=request();const button=$('#predictButton');busy(button,true,'专家协作中…');
 try{const p=await post('/api/predict',req);if(id!==state.requestId)return;renderPrediction(p);$('#historyChart').innerHTML='<div class="loading">加载设备历史…</div>';const rows=await api(`/api/devices/${req.device}/history?through=${p.observed_at}`);if(id===state.requestId)chart(rows);}catch(e){if(id===state.requestId)error(e.message);}finally{if(id===state.requestId)busy(button,false);}
}
document.querySelectorAll('[data-tab]').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('active',b===button));document.querySelectorAll('.tab-panel').forEach(p=>p.hidden=p.id!==`tab-${button.dataset.tab}`);$('#pageTitle').textContent=button.querySelector('span').textContent;}));
$('#deviceSelect').addEventListener('change',()=>loadRecords().catch(e=>error(e.message)));$('#recordSelect').addEventListener('change',runPrediction);$('#predictButton').addEventListener('click',runPrediction);
$('#explainButton').addEventListener('click',async()=>{const btn=$('#explainButton');clearError();if(!state.current)return;const id=state.requestId;busy(btn,true,'模型正在审核证据…');try{const r=await post('/api/explain',{...request(),provider:$('#auditProvider').value});if(id!==state.requestId)return;const ex=r.explanation;$('#explanation').innerHTML=`<p>${esc(ex.summary)}</p><div class="section-label">支持证据</div><ul>${ex.evidence.map(s=>`<li>${esc(s)}</li>`).join('')}</ul><div class="section-label">解释边界</div><ul>${ex.limitations.map(s=>`<li>${esc(s)}</li>`).join('')}</ul>`;$('#llmTag').textContent=r.provider+' 已完成';}catch(e){error(e.message);$('#llmTag').textContent='调用失败';}finally{busy(btn,false);}});
$('#customForm').addEventListener('submit',async event=>{event.preventDefault();clearError();const button=$('#customButton');busy(button,true,'提取报告并预测…');try{const r=await api('/api/predict/custom',{method:'POST',body:new FormData(event.target)});$('#customResult').classList.remove('empty-state');$('#customResult').innerHTML=`<div class="prediction-number"><strong>${fmt(r.prediction_adu)}</strong><span>adu</span></div><p class="body-text">标称 90% 区间：${fmt(r.interval_90_adu[0])}–${fmt(r.interval_90_adu[1])} adu</p><p class="muted">下一次巡检 t=${fmt(r.next_inspection_at,0)} atu</p><div class="warnings">${r.warnings.map(s=>`<p>· ${esc(s)}</p>`).join('')}</div><div class="agent-trace">${r.experts.map(a=>`<div class="agent-row"><span class="agent-icon">◉</span><strong>${esc(a.agent)}</strong><span class="agent-value">${a.prediction_adu==null?'未使用':fmt(a.prediction_adu)+' adu'}</span></div>`).join('')}</div><p class="muted">语义提取：${r.semantic_status==='completed'?esc(providerName(r.semantic_provider))+' 已完成':r.semantic_status==='unavailable'?'接口不可用，已显式降级':'报告为空'}</p>`;}catch(e){error(e.message);}finally{busy(button,false);}});
$('#checkApiButton').addEventListener('click',async()=>{const b=$('#checkApiButton');busy(b,true,'检查连接…');clearError();try{await post('/api/provider/check',{});await loadSummary();}catch(e){error(e.message);}finally{busy(b,false);}});
$('#reloadButton').addEventListener('click',async()=>{clearError();try{await post('/api/reload',{});await loadSummary();await loadDevices();}catch(e){error(e.message);}});
async function pollTraining(){try{const r=await api('/api/train/status');$('#trainingLog').textContent=r.log.join('\n')||'正在开始训练…';if(!r.running&&state.trainingTimer){clearInterval(state.trainingTimer);state.trainingTimer=null;$('#trainButton').disabled=false;if(r.exit_code===0){await post('/api/reload',{});await loadSummary();await loadDevices();}else if(r.exit_code!==null)error('训练失败，请查看日志。');}}catch(e){error(e.message);}}
$('#trainButton').addEventListener('click',async()=>{clearError();try{await post('/api/train',{});$('#trainButton').disabled=true;state.trainingTimer=setInterval(pollTraining,3000);await pollTraining();}catch(e){error(e.message);}});
(async()=>{try{await loadSummary();await loadDevices();await pollTraining();}catch(e){error(e.message);$('#connectionStatus').textContent='连接失败';}})();
