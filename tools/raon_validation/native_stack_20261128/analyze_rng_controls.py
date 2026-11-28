"""Paired-prompt bootstrap of completed RNG/checkpoint counterfactuals."""
import hashlib
import json
from pathlib import Path
import random
import statistics
root=Path('/mnt/hammerspace/jackhan/raon-rng-al-control-v2-20261128')
summary=json.loads((root/'results/status.json').read_bytes());assert summary['status']=='COMPLETE'
cache={};sources={}
for case in summary['cases']:
    key=(case['pair'],case['rng']);folder=root/'results'/('-'.join(key))/'samples'
    groups={}
    for path in folder.glob('*.json'):
        raw=path.read_bytes();row=json.loads(raw)
        groups.setdefault(row['dataset'],{})[row['ordinal']]=row
        sources[str(path.relative_to(root))]=hashlib.sha256(raw).hexdigest()
    assert sum(map(len,groups.values()))==592
    cache[key]=groups

def paired(a,b,dataset):
    left=cache[a][dataset];right=cache[b][dataset];assert set(left)==set(right)
    changes=[]
    for ordinal in sorted(left):
        x,y=left[ordinal],right[ordinal]
        assert x['messages_sha256']==y['messages_sha256'] and x['seed']==y['seed']
        changes.append(y['mean_request_al']-x['mean_request_al'])
    generator=random.Random(90128);n=len(changes)
    bootstrap=sorted(statistics.mean(generator.choices(changes,k=n)) for _ in range(2000))
    return dict(a=list(a),b=list(b),dataset=dataset,n=n,delta_mean_request_al=statistics.mean(changes),
        paired_prompt_bootstrap95=[bootstrap[50],bootstrap[1949]],
        limit='Paired prompts/seeds; stochastic generated trajectories differ. Single evaluation per variant, not cross-run variance.')

contrasts=[]
for dataset in ('gsm8k','math500','humaneval','mbpp','mt-bench'):
    contrasts.append(paired(('final','current'),('preview','current'),dataset))
    for pair in ('final','preview'):
        contrasts.append(paired((pair,'current'),(pair,'philox_independent'),dataset))
result=dict(status='COMPLETE',samples=4736,cases=summary['cases'],contrasts=contrasts,
    source_sha256=sources,production_rng_changed=False,
    conclusions=['Independent RNG family changes are a small AL effect in this grid; checkpoint pair accounts for part of the gap.',
        'Coupled-noise variants failed separate distribution marginal gold and are explanatory controls only.',
        'No direct claim about the unpublished B300 runtime commit or full NVIDIA output distribution.'])
(root/'rng-checkpoint-analysis.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(samples=4736,contrasts=contrasts)),flush=True)
