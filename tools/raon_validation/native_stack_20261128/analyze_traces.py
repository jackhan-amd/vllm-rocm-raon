"""Audit actual kernels and classify GPU time; never equate sums with wall time."""
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
root=Path(sys.argv[1] if len(sys.argv)>1 else '/mnt/hammerspace/jackhan/raon-composed-native-20261128')
def grid(event):
    value=event.get('args',{}).get('grid',[])
    if isinstance(value,str):value=[int(x) for x in re.findall(r'\d+',value)]
    return value
def category(event):
    name=event['name'];lower=name.lower()
    if 'raon_rda_projection_ratio' in lower:return 'native_rda_projection_ratio'
    if 'raon_moe_mhc_finish' in lower:return 'native_moe_mhc_finish'
    if 'raon_projection_mhc' in lower:return 'native_attention_projection_mhc'
    if 'mfma_moe1_' in lower:return 'native_moe_gemm1'
    if 'mfma_moe2_' in lower:return 'native_moe_gemm2'
    if 'raon_mxfp8_route_prepare' in lower:return 'native_route_prepare'
    if any(x in lower for x in ('sd_conv_recurrence_norm','sd_vsplit_recurrence','sd_common_prologue','sd_vsplit_norm','raon_rda_ar_fusion')):return 'native_rda_core'
    if 'log_sigmoid' in lower:return 'rda_logsigmoid'
    if 'topk_topp' in lower or 'topp_split' in lower:return 'sampling_mask'
    if 'gumbel' in lower:return 'sampling_gumbel'
    if 'rejection' in lower or 'recovery' in lower:return 'speculative_rejection'
    if 'selector' in lower or 'dflash' in lower or 'draft_logits' in lower:return 'draft_control'
    if 'hgemm_bf16_skinny' in lower:
        match=re.search(r'_t(\d+)x(\d+)x(\d+)',lower);shape=grid(event)
        if match and shape:
            n=int(match[2])*shape[0]
            if n==2176:return 'native_bf16_shared_gateup'
            if n==259840:return 'native_bf16_lm_heads_combined'
            if n==40:return 'native_bf16_rda_write_projection'
            if n==1152:return 'native_bf16_mla_input'
            if n==5120:return 'native_bf16_rda_or_mla_gate'
            if n==2560:return 'native_bf16_n2560_ambiguous_K'
        return 'native_bf16_other'
    if 'mhc' in lower:return 'native_mhc_projection_or_gate'
    if lower.startswith('head_'):return 'native_mhc_chart_aggregate'
    if lower.startswith('mix_'):return 'native_mhc_mix'
    if 'mla' in lower:return 'mla_attention'
    if 'sort' in lower or 'quant' in lower:return 'routing_or_quantization'
    if 'expfunctor' in lower:return 'torch_exp'
    if 'cast' in lower or 'copy' in lower:return 'torch_copy_or_cast'
    if 'vectorized_elementwise' in lower:return 'torch_other_elementwise'
    if 'gemm' in lower or 'cijk' in lower:return 'other_gemm'
    return 'unclassified'

rows=[]
folders=list((root/'traces').glob('*'))
if root.name=='raon-composed-native-20261128':
    repair=root.parent/'raon-composed-port-repair-20261128/traces'
    names={f.name for f in folders if list((f/'profiles').glob('*_rank0.*.pt.trace.json*'))}
    folders += [f for f in repair.glob('*') if f.name not in names]
for folder in sorted(folders):
    paths=list((folder/'profiles').glob('*_rank0.*.pt.trace.json*'))
    if not paths:continue
    assert len(paths)==1,paths
    path=paths[0]
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as file:data=json.load(file)
    events=data['traceEvents'];kernels=[e for e in events if e.get('cat')=='kernel' and e.get('dur',0)>0]
    groups=defaultdict(lambda:dict(count=0,gpu_us=0.));names=defaultdict(lambda:dict(count=0,gpu_us=0.,grids=Counter()))
    for event in kernels:
        group=groups[category(event)];group['count']+=1;group['gpu_us']+=event['dur']
        item=names[event['name']];item['count']+=1;item['gpu_us']+=event['dur'];item['grids'][str(grid(event))]+=1
    total=sum(e['dur'] for e in kernels)
    intervals=sorted((e['ts'],e['ts']+e['dur']) for e in kernels)
    active=0.;gaps=[]
    if intervals:
        start,end=intervals[0]
        for lo,hi in intervals[1:]:
            if lo>end:
                active+=end-start;gaps.append(lo-end);start,end=lo,hi
            else:end=max(end,hi)
        active+=end-start;span=intervals[-1][1]-intervals[0][0]
    else:span=0.
    for value in groups.values():value['summed_kernel_time_share']=value['gpu_us']/total if total else None
    def top_names():
        return [dict(name=name,count=v['count'],gpu_us=v['gpu_us'],grids=dict(v['grids']))
            for name,v in sorted(names.items(),key=lambda item:-item[1]['gpu_us'])[:24]]
    rows.append(dict(case=folder.name,raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        gpu_kernel_events=len(kernels),summed_kernel_us=total,gpu_activity_union_us=active,gpu_kernel_span_us=span,
        gap_us_within_kernel_span=span-active,largest_gaps_us=sorted(gaps,reverse=True)[:12],
        categories=dict(sorted(groups.items(),key=lambda item:-item[1]['gpu_us'])),top_kernels=top_names(),
        profile_steps=[e['name'] for e in events if e.get('name','').startswith('ProfilerStep#')],
        sample_gpu_event=kernels[0] if kernels else None))
result=dict(status='GPU_EVENTS_PRESENT' if rows and all(r['gpu_kernel_events'] for r in rows) else 'REVIEW_REQUIRED',rows=rows,
    limitations=['Summed kernel durations can overlap streams; percentages are not wall time or GPU utilization.',
        'N is inferred only for known skinny GEMM grids. K/target-vs-draft ownership is left ambiguous where not recorded.',
        'Gaps inside the GPU event span can include host scheduling and dependencies; they are not attributed to a CPU cause here.',
        'These separate traces were collected after all official speed windows and do not replace their TPS.'])
(root/'trace-analysis.json').write_text(json.dumps(result,indent=2))
for r in rows:print(json.dumps(dict(case=r['case'],kernels=r['gpu_kernel_events'],sum_us=r['summed_kernel_us'],
    union_us=r['gpu_activity_union_us'],span_us=r['gpu_kernel_span_us'],categories=r['categories'],top_kernels=r['top_kernels'][:5])),flush=True)
