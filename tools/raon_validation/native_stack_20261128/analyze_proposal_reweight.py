"""CPU-only fixed-prefix proposal calibration; not a counterfactual AL score."""
import json
from pathlib import Path
import statistics
import torch
torch.set_num_threads(2)
root=Path('/work');status=json.loads((root/'results/status.json').read_bytes());assert status['status']=='COMPLETE'
oracle=json.loads((root/'processed-overlap-analysis.json').read_bytes());assert oracle['status']=='ORACLE_VALIDATED'
groups=[]
for case in status['cases']:
    folder=root/'results'/f'gpu-{case["rank"]}'
    prompt_len=len(json.loads((folder/'live-response.json').read_bytes())['prompt_token_ids'])
    changes=[]
    for path in sorted(folder.glob('capture-*.pt')):
        d=torch.load(path,map_location='cpu',weights_only=False)
        for row in range(7):
            q=d['q'][row];p=d['live_p'][row];ids=d['candidate_ids'][row]
            history=set((d['prefix']+d['input_ids'][:row+1].tolist())[prompt_len:])
            logits=torch.where(q>0,q.log(),torch.full_like(q,-torch.inf))
            baseline=float(torch.minimum(p,q).sum());values={}
            for tau in (.5,.7,.85,1.,1.15,1.3,1.5,2.):
                proposal=(logits/tau).softmax(-1);values[str(tau)]=float(torch.minimum(p,proposal).sum())
            penalty=torch.tensor([1.5 if int(token) in history else 0. for token in ids]) if case['dataset']=='mt-bench' else torch.zeros_like(q)
            values['match_target_presence']=float(torch.minimum(p,(logits-penalty).softmax(-1)).sum())
            changes.append(dict(step=d['step'],row=row,baseline=baseline,overlap=values))
        del d
    groups.append(dict(pair=case['pair'],dataset=case['dataset'],ordinal=case['ordinal'],n=len(changes),
        baseline=statistics.mean(r['baseline'] for r in changes),
        overlap={name:statistics.mean(r['overlap'][name] for r in changes) for name in changes[0]['overlap']}))
assert not torch.cuda.is_initialized()
result=dict(status='CPU_FIXED_PREFIX_PROPOSAL_ANALYSIS',groups=groups,gpu_work=0,
    limit='Only8 prompts and their realized draft prefixes; this is proposal/target overlap, not predicted AL, accuracy or TPS. A different proposal changes future prefixes.',
    probability_source='Saved actual processed live p and cached realized q; processed-p oracle already validated to1.2e-7')
(root/'proposal-reweight-analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
