import hashlib
import json
import math
from pathlib import Path
import sys
root=Path(sys.argv[1] if len(sys.argv)>1 else '/mnt/hammerspace/jackhan/raon-composed-native-20261128')
repair=Path('/mnt/hammerspace/jackhan/raon-composed-port-repair-20261128')
composed=Path('/mnt/hammerspace/jackhan/raon-composed-native-20261128/analysis.json')
previous={r['case']:r for r in json.loads(composed.read_bytes())['rows']} if composed.exists() else {}
prior=Path('/mnt/hammerspace/jackhan/raon-public-target-20261128/dataset-effects.json')
old={r['case']:r for r in json.loads(prior.read_bytes())['rows']}
card={'gsm8k':[(296.3,646.3),(1110.4,1836.3),(1528.9,2619.3)],
    'math500':[(308.,666.6),(1496.1,2511.3),(3082.5,5166.9)],
    'humaneval':[(305.4,607.9),(1470.8,2269.6),(2793.1,4596.4)],
    'mbpp':[(307.,604.1),(1450.6,2197.2),(2956.2,5017.9)],
    'mt-bench':[(304.9,570.2),(1477.1,2038.2),(2521.3,3935.4)]}
rows=[]
for dataset in card:
  for c in (1,8,32):
    for sd in ('off','on'):
      name=f'{dataset}-{sd}-c{c}';folder=root/'results'/name;p=folder/'measurement/result.json'
      if root.name=='raon-composed-native-20261128' and not p.exists() and (repair/'results'/name/'measurement/result.json').exists():
        folder=repair/'results'/name;p=folder/'measurement/result.json'
      if not p.exists():
        state=json.loads((folder/'status.json').read_bytes()) if (folder/'status.json').exists() else {}
        rows.append(dict(case=name,status=state.get('status','NOT_STARTED'),error=state.get('traceback','')[-500:]));continue
      raw=p.read_bytes();r=json.loads(raw)
      valid=r['status']=='VALID_SINGLE_MEASUREMENT' and not r['errors'] and not r['server_log_check']['jit_warnings']
      assert r['requests']==(80 if dataset=='mt-bench' else 128)
      assert math.isclose(r['output_tps'],r['output_tokens']/r['exact_measured_wall_seconds'],rel_tol=1e-12)
      public=card[dataset][(1,8,32).index(c)][int(sd=='on')]
      baseline=old[name]
      rows.append(dict(case=name,status=r['status'],valid=valid,tps=r['output_tps'],baseline_tps=baseline['output_tps'],
          speed_ratio=r['output_tps']/baseline['output_tps'],public_tps=public,public_ratio=r['output_tps']/public,
          meets90pct=valid and r['output_tps']>=.9*public,output_tokens=r['output_tokens'],mean_request_al=r['upstream_mean_request_acceptance_length'],
          sha256=hashlib.sha256(raw).hexdigest(),result_path=str(p),
          previous_composed_tps=previous.get(name,{}).get('tps')))
traces=[]
for folder in sorted((root/'traces').glob('*')):
    p=folder/'status.json'
    if p.exists():
        data=json.loads(p.read_bytes());data['traceback']=data.get('traceback','')[-500:];traces.append(data)
result=dict(rows=rows,valid=sum(r.get('valid',False) for r in rows),meets90pct=sum(r.get('meets90pct',False) for r in rows),
    planned=30,traces=traces,baseline_modified=False,
    comparison_limit='Current and candidate targets are final Raon-mini; public B300 reference uses preview. Threshold attainment is a descriptive reference comparison, not an identical-checkpoint hardware ranking.')
(root/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
