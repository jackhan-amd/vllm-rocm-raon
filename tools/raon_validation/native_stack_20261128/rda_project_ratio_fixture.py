"""Real-checkpoint stateless RDA projection/ratio gold, graph replay and timing."""
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32=False
import vllm.models.raon1_mini as package
package.__path__.insert(0,'/work/rda-side')
from vllm.models.raon1_mini.rocm_rda_project_ratio import forward
from vllm.models.raon1_mini.rocm_dense import _gemm,configuration
from safetensors import safe_open
rank=int(os.environ.get('RAON_REPLICA','0'))
root=Path('/work/rda-side-results')/f'gpu-{rank}';root.mkdir(parents=True,exist_ok=True)
model=Path('/release/target/3f85cd63ea420100a84378f72ae162af86a441fd')
index=json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map']
keys=sorted(k for k in index if k.endswith('rda_w_proj.weight'));assert len(keys)>=8,keys
def weight_for(key):
    with safe_open(model/index[key],framework='pt',device='cpu') as file:w=file.get_tensor(key)
    assert w.shape==(40,2560) and w.dtype==torch.bfloat16
    return w.cuda().contiguous()

def error(actual,gold,rms=1e-2,peak=2e-2,atol=1e-6):
    a,b=actual.double(),gold.double();finite=bool(torch.isfinite(a).all() & torch.isfinite(b).all())
    if not finite:return dict(pass_=False,finite=False)
    diff=a-b;r=float(diff.square().mean().sqrt());br=float(b.square().mean().sqrt());p=float(diff.abs().max());bp=float(b.abs().max())
    return dict(pass_=r<=atol+rms*br and p<=atol+peak*bp,rms=r,relative_rms=r/max(br,1e-30),peak=p,relative_peak=p/max(bp,1e-30),finite=finite)

def ratio(beta,delta):return torch.exp(F.logsigmoid(beta.float()+delta.float())-F.logsigmoid(beta.float()))

if '--profile-only' in sys.argv:
    # Initialize profiler before any GPU allocations in this new process.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]):
        torch.empty(1,device='cuda');torch.cuda.synchronize()
    w=weight_for(keys[0]);x=torch.randn((8,2560),device='cuda',dtype=torch.bfloat16)
    beta=torch.randn((1,8,40),device='cuda',dtype=torch.bfloat16)
    for _ in range(3):forward(x,w,beta)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for _ in range(20):forward(x,w,beta)
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(root/'kernel-trace.json'))
    names=sorted({e.name for e in profiler.events() if e.device_type.name=='CUDA'})
    receipt=dict(status='PASS' if any('raon_rda_projection_ratio' in n for n in names) else 'MISSING_GPU_KERNEL_WITNESS',kernels=names)
    (root/'kernel-witness.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));raise SystemExit(0 if receipt['status']=='PASS' else 2)

result=dict(status='RUNNING',cases=[],timings=[],cache_updates=0,precision='BF16 inputs/weights and delta seam; FP32 ratio',
    scope='Real write-projection weights, synthetic activations; not model TPS or quality')
try:
  for ordinal,rows in enumerate((1,2,4,7,8,16,31,32,33,63,64)):
    if ordinal%8!=rank:continue
    torch.manual_seed(10900+rows);w=weight_for(keys[rows%len(keys)])
    for layout in ('compact','projected_stride20656'):
      for scale in (.125,1.,8.,'wide_beta'):
        x=(torch.randn((rows,2560),device='cuda')*(1. if scale=='wide_beta' else scale)).bfloat16()
        storage=torch.full((rows,40 if layout=='compact' else 20656),-111.,device='cuda',dtype=torch.bfloat16)
        beta=storage[:,-40:].unsqueeze(0)
        values=torch.linspace(-80,80,rows*40,device='cuda').view(1,rows,40) if scale=='wide_beta' else torch.randn_like(beta.float())*3
        beta.copy_(values)
        config=configuration(rows,40,2560)
        def baseline():return ratio(beta,(_gemm()(x,w,**config) if config is not None else F.linear(x,w))[None])
        expected=baseline();actual=forward(x,w,beta)
        gold=ratio(beta,F.linear(x.float(),w.float()).bfloat16()[None])
        native_check=error(actual,expected,5e-5,5e-4) if config is not None else error(actual,expected)
        torch_check=error(actual,gold)
        case=dict(rows=rows,layout=layout,scale=scale,checkpoint_weight=keys[rows%len(keys)],native=native_check,torch_fp32=torch_check,
            baseline_projection='original native skinny' if config is not None else 'original wider-row torch BF16 linear',chunks=(rows+31)//32)
        result['cases'].append(case)
        assert native_check['pass_'] and torch_check['pass_'],case
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=forward(x,w,beta)
        for repeat in range(3):
            x.add_(.005);beta.add_(.01)
            expected=forward(x,w,beta).clone();graph.replay()
            assert torch.equal(captured,expected),'Changed-input replay mismatch'
        case['changed_graph_replays']=3
        if layout=='projected_stride20656':assert bool(torch.all(storage[:,:-40]==-111.))
    # One event window per implementation, after separate compile and warmup.
    baseline_graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(baseline_graph):baseline_output=baseline()
    fused_graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(fused_graph):fused_output=forward(x,w,beta)
    values={}
    for name,g in (('baseline',baseline_graph),('fused',fused_graph)):
        for _ in range(20):g.replay()
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(500):g.replay()
        end.record();end.synchronize();values[name+'_us']=start.elapsed_time(end)*1000/500
    result['timings'].append(dict(rows=rows,**values,speedup=values['baseline_us']/values['fused_us'],operator_only=True))
    (root/'gold.json').write_text(json.dumps(result,indent=2))
  result['status']='PASS'
except BaseException:
    import traceback
    result.update(status='FAILED_RETAINED',traceback=traceback.format_exc())
finally:(root/'gold.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(status=result['status'],cases=len(result['cases']),timings=result['timings'],error=result.get('traceback'))),flush=True)
if result['status']!='PASS':raise SystemExit(2)
