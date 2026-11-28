"""Paired prompt/seed contrasts; no timing or equivalent-trajectory claim."""
import hashlib
import json
from pathlib import Path
import random
import statistics
root=Path('/mnt/hammerspace/jackhan/raon-al-path-factors-20261128')
reference=Path('/mnt/hammerspace/jackhan/raon-rng-al-control-v2-20261128')
plan=json.loads((root/'rng-plan.json').read_bytes());rows=[];sources={}
plan['cases'] += [dict(pair=p,rng='selector64') for p in ('final','preview')]
for case in plan['cases']:
  pair,variant=case['pair'],case['rng']
  folder=root/'results'/f'{pair}-{variant}'
  status=json.loads((folder/'status.json').read_bytes()) if (folder/'status.json').exists() else {}
  actual={}
  for path in (folder/'samples').glob('*.json'):
    raw=path.read_bytes();r=json.loads(raw);actual[(r['dataset'],r['ordinal'])]=r
    sources[str(path.relative_to(root))]=hashlib.sha256(raw).hexdigest()
  for dataset in ('humaneval','gsm8k','mt-bench','mbpp','math500'):
    changes=[];values=[];base=[]
    for (name,ordinal),row in sorted(actual.items()):
      if name!=dataset:continue
      old=json.loads((reference/'results'/f'{pair}-current/samples/{dataset}-{ordinal:03d}.json').read_bytes())
      assert row['seed']==old['seed'] and row['messages_sha256']==old['messages_sha256']
      values.append(row['mean_request_al']);base.append(old['mean_request_al']);changes.append(values[-1]-base[-1])
    if not changes:continue
    rng=random.Random(90144);n=len(changes)
    boot=sorted(statistics.mean(rng.choices(changes,k=n)) for _ in range(2000))
    rows.append(dict(pair=pair,variant=variant,dataset=dataset,n=n,
      variant_status=status.get('status'),mean_request_al=statistics.mean(values),reference_al=statistics.mean(base),
      paired_delta=statistics.mean(changes),paired_prompt_bootstrap95=[boot[50],boot[1949]],
      interpretation='Paired prompts/seeds but generated trajectories differ; bootstrap is across prompts, not repeated-run variance.'))
result=dict(status='COMPLETE' if len(sources)==4736 else 'PARTIAL',samples=len(sources),rows=rows,source_sha256=sources,
  baseline_changed=False,no_tps_measurement=True,
  termination='Six original supported variants completed592 each. BF16 KV failed startup before generation. K64 added later retained487 final/537 preview; parent container cleanup ended the last Math samples. No completed records discarded.',
  completed_panel='RDA fused/unfused and K32: all5 datasets. K64: full HumanEval/GSM8K/MT-Bench/MBPP; Math partial.')
(root/'path-factor-analysis.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)
