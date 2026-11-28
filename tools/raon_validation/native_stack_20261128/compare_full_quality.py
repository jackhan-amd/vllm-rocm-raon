"""Read immutable completed answers; compare paired correctness, never rerun."""
import collections
import json
import math
from pathlib import Path
old=Path('/mnt/hammerspace/jackhan/raon-quality-bounds-20261128')
new=Path('/mnt/hammerspace/jackhan/raon-rda-quality-20261128')
def read(root):
    summary=json.loads((root/'mmlu-final-summary.json').read_bytes());assert summary['status']=='FULL_SET_VALIDATED'
    data={}
    for path in (root/'state'/summary['state_fingerprint']).glob('shard-*/sd_on/mmlu/completed/*.json'):
        row=json.loads(path.read_bytes());assert row['index'] not in data;data[row['index']]=row
    assert len(data)==12032
    return summary,data
a,left=read(old);b,right=read(new);records=[];counts=collections.Counter()
for index in sorted(left):
    x,y=left[index],right[index]
    for key in ('question_id','category','answer','prompt_tokens_sha256','prompt_token_count'):assert x[key]==y[key],(index,key)
    difference=int(y['correct'])-int(x['correct']);resumed=len(x['segments'])>1 or len(y['segments'])>1
    counts[(bool(x['correct']),bool(y['correct']))]+=1
    records.append(dict(index=index,category=x['category'],old_correct=x['correct'],new_correct=y['correct'],delta=difference,resumed=resumed))
def contrast(rows):
    n=len(rows);d=sum(r['delta'] for r in rows)/n;variance=sum(r['delta']**2 for r in rows)/n-d*d;se=math.sqrt(variance/n)
    gained=sum(r['delta']==1 for r in rows);lost=sum(r['delta']==-1 for r in rows);discordant=gained+lost
    limit=min(gained,lost)
    if discordant:
        terms=[math.lgamma(discordant+1)-math.lgamma(k+1)-math.lgamma(discordant-k+1)-discordant*math.log(2) for k in range(limit+1)]
        m=max(terms);p=min(1.,2*math.exp(m)*sum(math.exp(t-m) for t in terms))
    else:p=1.
    return dict(n=n,gained=gained,lost=lost,delta_accuracy_pp=100*d,paired_normal95_pp=[100*(d-1.96*se),100*(d+1.96*se)],
        mcnemar_exact_two_sided_p=p)
result=dict(status='PAIRED_FULL_SET_ANALYSIS',old_correct=a['correct'],new_correct=b['correct'],
    old_accuracy_percent=a['accuracy_percent'],new_accuracy_percent=b['accuracy_percent'],
    all=contrast(records),uninterrupted_questions=contrast([r for r in records if not r['resumed']]),
    original_frozen_baseline=dict(correct=10021,total=12032,accuracy_percent=100*10021/12032),
    delta_vs_original_frozen_pp=100*(b['correct']-10021)/12032,
    categories={name:contrast([r for r in records if r['category']==name]) for name in sorted({r['category'] for r in records})},
    limitations=['Same questions and prompt hashes; stochastic generated trajectories and batch composition differ.',
        'Paired confidence interval is across questions, not repeated-run variance or a proof of numerical equivalence.',
        'No predeclared quality noninferiority margin; no automatic accuracy acceptance.',
        'New raw audit field delta_vs_frozen_baseline_pp names the prior optimized10029 answer reference, not the original10021 frozen baseline. Raw artifacts remain unchanged.'])
(new/'paired-quality-analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
