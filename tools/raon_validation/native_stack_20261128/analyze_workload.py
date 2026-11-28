"""Read-only workload diagnosis from existing responses and metric deltas."""
import collections
import json
import math
from pathlib import Path
import re
import statistics as st
import sys

ROOT=Path(sys.argv[1] if len(sys.argv)>1 else '/mnt/hammerspace/jackhan/raon-public-target-20261128')
REPAIR=Path('/mnt/hammerspace/jackhan/raon-public-jit-repair-20261128')
PRIOR=Path('/mnt/hammerspace/jackhan/raon-common-epilogue-20261128')

def load(path):
    try:return json.loads(path.read_bytes())
    except (OSError,ValueError):return {}

def metrics(path):
    out=collections.defaultdict(float)
    for line in path.read_text().splitlines():
        found=re.match(r'^(vllm:[a-zA-Z0-9_:]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)(?:\s|$)',line)
        if found:out[found[1]]+=float(found[2])
    return out

def quantile(values,q):
    values=sorted(values)
    index=(len(values)-1)*q
    lo=math.floor(index);hi=math.ceil(index)
    return values[lo]+(values[hi]-values[lo])*(index-lo)

def dist(values):
    if not values:return None
    return dict(mean=st.mean(values),p50=quantile(values,.5),p90=quantile(values,.9),
                p99=quantile(values,.99),min=min(values),max=max(values),
                cv=st.pstdev(values)/st.mean(values) if st.mean(values) else None)

rows=[]
for dataset in ('gsm8k','math500','humaneval','mbpp','mt-bench'):
    for c in (1,8,32):
        for sd in ('off','on'):
            name=f'{dataset}-{sd}-c{c}'
            base=PRIOR/f'integration-speed/projection-{sd}-c{c}/gsm8k' if dataset=='gsm8k' and len(sys.argv)==1 else ROOT/f'results/{name}/measurement'
            repaired=REPAIR/f'results/{name}/measurement'
            if len(sys.argv)==1 and load(repaired/'result.json').get('status')=='VALID_SINGLE_MEASUREMENT':base=repaired
            result=load(base/'result.json')
            if not result:continue
            requests=[json.loads(line) for line in (base/'requests.jsonl').open()]
            requests=[r for r in requests if r['phase']=='measurement']
            usages=[r['usage'] for r in requests]
            inputs=[u['prompt_tokens'] for u in usages]
            outputs=[u['completion_tokens'] for u in usages]
            reasoning=[u.get('completion_tokens_details',{}).get('reasoning_tokens') for u in usages]
            specs=[(r.get('raw_response',{}).get('metrics') or {}).get('speculative_decoding',{}) for r in requests]
            acceptance_groups=[]
            if sd=='on':
                for label,lower,upper in (('short_0_512',0,512),('medium_513_2048',513,2048),('long_2049_4096',2049,4096)):
                    group=[r for r in requests if lower<=r['usage']['completion_tokens']<=upper]
                    if group:
                        acceptance_groups.append(dict(group=label,count=len(group),
                            mean_request_al=st.mean(r['mapped_meta_info']['spec_accept_length'] for r in group),
                            output_token_share=sum(r['usage']['completion_tokens'] for r in group)/sum(outputs)))
            finishes=collections.Counter(r['raw_response']['choices'][0]['finish_reason'] for r in requests)
            before=metrics(base/'metrics-before.txt');after=metrics(base/'metrics-after.txt')
            delta={key:after[key]-before[key] for key in after}
            def metric(name):return delta.get('vllm:'+name,0.)
            drafts=metric('spec_decode_num_drafts_total');accepted=metric('spec_decode_num_accepted_tokens_total')
            valid=result['status']=='VALID_SINGLE_MEASUREMENT'
            wall=result['exact_measured_wall_seconds']
            inference=metric('request_inference_time_seconds_sum')
            prefill=metric('request_prefill_time_seconds_sum')
            decode=metric('request_decode_time_seconds_sum')
            log=(base/'server-measured.log').read_text(errors='replace')
            tuned=collections.Counter(re.findall(r'shape is M:(\d+), N:(\d+), K:(\d+)',log))
            rows.append(dict(case=name,dataset=dataset,sd=sd,c=c,path=str(base),
                status=result['status'],timing_valid=valid,requests=len(requests),
                unique_prompts=len({r['messages_sha256'] for r in requests}),
                input_tokens=dist(inputs),output_tokens=dist(outputs),finish_reasons=dict(finishes),
                capped_4096=sum(v==4096 for v in outputs),
                reasoning_token_fraction=sum(reasoning)/sum(outputs) if all(v is not None for v in reasoning) else None,
                reasoning_tokens=dist(reasoning) if all(v is not None for v in reasoning) else None,
                output_tps=result['output_tps'],wall_seconds=wall if valid else None,
                mean_request_al=result['upstream_mean_request_acceptance_length'],
                token_cycle_weighted_al=1+accepted/drafts if drafts else None,
                acceptance_by_output_length=acceptance_groups,
                response_spec_metric_schema=list(specs[0]) if specs and isinstance(specs[0],dict) else [],
                draft_cycles=drafts,accepted_draft_tokens=accepted,
                mean_client_inflight=sum(r['request_seconds'] for r in requests)/wall if valid else None,
                mean_engine_running_requests=inference/wall if valid else None,
                prefill_share_request_running=prefill/inference if valid and inference else None,
                decode_share_request_running=decode/inference if valid and inference else None,
                mean_decode_ms_per_sequence_cycle=1000*decode/drafts if valid and drafts else None,
                mean_decode_ms_per_output_token=1000*decode/sum(outputs) if valid else None,
                cache_hit_fraction=metric('prefix_cache_hits_total')/metric('prefix_cache_queries_total') if metric('prefix_cache_queries_total') else None,
                preemptions=metric('num_preemptions_total'),
                generation_counter_delta=metric('generation_tokens_total'),
                response_completion_sum=sum(outputs),
                top_untuned_gemm_log_shapes=[dict(m=int(k[0]),n=int(k[1]),k=int(k[2]),log_occurrences=v) for k,v in tuned.most_common(5)],
                caveats=['Untuned-shape logs are not invocation counts or GPU time',
                    'Running/prefill/decode metrics are request elapsed-time sums, not kernel-time shares',
                    'mean_engine_running_requests is an elapsed-time proxy; no scheduler timeline reconstructed']))

payload=dict(status='SAVED_ARTIFACT_DIAGNOSIS',new_gpu_work=0,rows=rows,
    limitations=['Current target final Raon-mini; public reference preview on B300',
        'Exact public sample selection, output length distribution, and per-concurrency AL aggregation unavailable',
        'Timing-invalid originals have token/finish statistics only; timing interpretation excluded'])
(ROOT/'dataset-effects.json').write_text(json.dumps(payload,indent=2))
for row in rows:
    if row['sd']=='on' or row['c']==32:
        print(json.dumps({key:row[key] for key in ('case','status','output_tps','input_tokens','output_tokens','reasoning_token_fraction',
            'finish_reasons','capped_4096','mean_request_al','token_cycle_weighted_al','mean_engine_running_requests',
            'prefill_share_request_running','mean_decode_ms_per_sequence_cycle','cache_hit_fraction','preemptions')}))
