# SPDX-License-Identifier: Apache-2.0
"""CPU-only probability reconstruction, gated by saved actual SD probabilities."""
import json
from pathlib import Path
import statistics
import torch
torch.set_num_threads(4)
root=Path('/work');status=json.loads((root/'results/status.json').read_bytes())
assert status['status']=='COMPLETE'

def probabilities(logits,history,presence):
    logits=logits.float().clone()
    if presence and history:logits[torch.tensor(sorted(set(history)),dtype=torch.long)]-=presence
    # Match the current Triton filter's exact-k, low-token-index duplicate
    # budget. The older PyTorch value-threshold reference retains all ties.
    threshold=logits.topk(20).values[-1]
    higher=(logits>threshold).nonzero().flatten()
    boundary=(logits==threshold).nonzero().flatten()[:20-len(higher)]
    ids=torch.cat((higher,boundary))
    values,order=logits[ids].sort(descending=True,stable=True);ids=ids[order]
    cumulative=values.softmax(-1).cumsum(-1)
    mask=(cumulative-values.softmax(-1))>=torch.tensor(.95,dtype=torch.float32);mask[0]=False
    values[mask]=-float('inf')
    output=torch.zeros_like(logits);output[ids]=values.softmax(-1)
    return output

groups=[];all_rows=[]
for case in status['cases']:
    folder=root/'results'/f'gpu-{case["rank"]}'
    policy=json.loads((folder/'trace-eos-policy.json').read_bytes());assert policy['ignore_eos'] is False
    prompt=json.loads((folder/'live-response.json').read_bytes())['prompt_token_ids']
    rows=[]
    for record in case['comparisons']:
        step,row=record['step'],record['row']
        sd=torch.load(folder/f'capture-{step:04d}.pt',map_location='cpu',weights_only=False)
        ar=torch.load(folder/f'ar-{step:04d}-{row}.pt',map_location='cpu',weights_only=False)
        fresh=torch.load(folder/f'score-{step:04d}-{row}.pt',map_location='cpu',weights_only=False)
        tokens=sd['prefix']+sd['input_ids'][:row+1].tolist()
        assert ar['prefix']==tokens==fresh['prefix']
        history=tokens[len(prompt):];presence=1.5 if case['dataset']=='mt-bench' else 0.
        p=probabilities(sd['logits'][row],history,presence)
        a=probabilities(ar['logits'][-1],history,presence)
        f=probabilities(fresh['logits'][-1],history,presence)
        ids,q=sd['candidate_ids'][row],sd['q'][row]
        oracle_error=float((p[ids]-sd['live_p'][row]).abs().max())
        overlap=lambda distribution:float(torch.minimum(distribution[ids],q).sum())
        item=dict(step=step,row=row,oracle_max_abs=oracle_error,
            sd_overlap=overlap(p),ar_overlap=overlap(a),fresh_overlap=overlap(f),
            sd_ar_processed_tv=float((p-a).abs().sum()/2),ar_fresh_processed_tv=float((a-f).abs().sum()/2))
        rows.append(item)
    group=dict(rank=case['rank'],pair=case['pair'],dataset=case['dataset'],ordinal=case['ordinal'],rows=rows,
        n=len(rows),oracle_max_abs=max(r['oracle_max_abs'] for r in rows),
        sd_overlap_mean=statistics.mean(r['sd_overlap'] for r in rows),ar_overlap_mean=statistics.mean(r['ar_overlap'] for r in rows),
        fixed_prefix_ar_minus_sd_overlap_mean=statistics.mean(r['ar_overlap']-r['sd_overlap'] for r in rows),
        first_position_ar_minus_sd_overlap_mean=statistics.mean(r['ar_overlap']-r['sd_overlap'] for r in rows if r['row']==0),
        processed_tv_sd_ar_mean=statistics.mean(r['sd_ar_processed_tv'] for r in rows))
    groups.append(group);all_rows.extend(rows)
max_error=max(r['oracle_max_abs'] for r in all_rows)
result=dict(status='ORACLE_VALIDATED' if max_error<=1e-5 else 'MASK_RECONSTRUCTION_REVIEW_REQUIRED',
    n=len(all_rows),oracle_max_abs=max_error,groups=groups,
    fixed_prefix_ar_minus_sd_overlap_mean=statistics.mean(r['ar_overlap']-r['sd_overlap'] for r in all_rows),
    limitation='Fixed realized draft prefixes and the same cached q. This is scorer overlap, not a counterfactual AL prediction or a new accuracy evaluation.')
destination=root/'processed-overlap-analysis.json'
if destination.exists():
    import shutil
    prior=root/'processed-overlap-value-threshold-analysis.json'
    if not prior.exists():shutil.copyfile(destination,prior)
destination.write_text(json.dumps(result,indent=2))
print(json.dumps({**result,'groups':[{k:v for k,v in g.items() if k!='rows'} for g in groups]}),flush=True)
