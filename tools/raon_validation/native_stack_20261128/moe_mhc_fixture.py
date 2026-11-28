"""Native original-chain and independent Torch gold before any timing."""
import json
import os
from pathlib import Path
import time

import torch
from vllm.models.raon1_mini import mhc,rocm_mhc_gate
from vllm.models.raon1_mini.rocm_flydsl_combine import combine_unweighted
from vllm.models.raon1_mini.rocm_moe_mhc import DeferredMoE,finish
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

rank=int(os.environ.get('RAON_REPLICA','0'))
out=Path('/work/fixtures')/f'gpu-{rank}.json';out.parent.mkdir(exist_ok=True)
shapes=(1,2,7,8,16,31,32,64,128)
rows_list=[r for i,r in enumerate(shapes) if i%8==rank]
records=[]
for rows in rows_list:
    for mix_one in (False,True):
      for steer in (False,True):
        torch.manual_seed(3100+rows)
        mhc._BATCH_INVARIANT=True;mhc._FIXED_MIX_ONE=mix_one
        v=torch.randn((rows,6,2560),device='cuda',dtype=torch.bfloat16)*.2
        ids=torch.randint(-1,128,(rows,6),device='cuda',dtype=torch.int32)
        shared=torch.randn((rows,2560),device='cuda',dtype=torch.bfloat16)*.2
        carrier=torch.randn((rows,4,2560),device='cuda',dtype=torch.bfloat16)*.2
        coeff=torch.randn((rows,20),device='cuda',dtype=torch.float32)*.2
        alpha=torch.tensor([0.,.05,.1,.5],device='cuda').repeat((rows+3)//4)[:rows].contiguous()
        vector=torch.randn((2560,),device='cuda',dtype=torch.float32)*.1
        value=DeferredMoE(v,ids,shared,alpha if steer else None,vector if steer else None)
        launch_mix=rocm_mhc_gate._create(False,mix_one,False,False)[1]
        def original():
            y=combine_unweighted(v,ids)+shared
            if steer:y=torch.where(alpha[:,None]==0,y,(y.float()+alpha[:,None]*vector).to(y.dtype))
            output=torch.empty_like(carrier)
            _run_compiled(launch_mix,carrier.view(-1),y.view(-1),coeff.view(-1),output.view(-1),rows,torch.cuda.current_stream().cuda_stream)
            return output
        def gold():
            order=torch.argsort(ids,dim=1,stable=True)
            selected=v.gather(1,order[:,:,None].expand_as(v))
            y=torch.zeros_like(shared,dtype=torch.float32)
            for j in range(6):y=y+selected[:,j].float()
            y=(y.to(v.dtype)+shared)
            if steer:y=torch.where(alpha[:,None]==0,y,(y.float()+alpha[:,None]*vector).to(v.dtype))
            # Independent FP64 mHC reference. Exact native schedule is checked
            # separately above; this reference has no fused-rounding mimic.
            matrix=coeff[:,4:].reshape(rows,4,4).double()
            return (torch.einsum('bij,bjh->bih',matrix,carrier.double())+
                coeff[:,:4,None].double()*y[:,None,:].double()).to(v.dtype)
        expected=original();actual=finish(value,carrier,coeff)
        assert torch.equal(actual,expected), (rows,mix_one,steer,'native bits', (actual.float()-expected.float()).abs().max().item())
        torch.testing.assert_close(actual,gold(),atol=.0078125,rtol=.015625)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):replayed=finish(value,carrier,coeff)
        for seed in (11,12,13):
            torch.manual_seed(seed)
            v.copy_(torch.randn_like(v)*.2);shared.copy_(torch.randn_like(shared)*.2)
            carrier.copy_(torch.randn_like(carrier)*.2);coeff.copy_(torch.randn_like(coeff)*.2)
            alpha.copy_(torch.rand_like(alpha)*.1);alpha[0]=0
            ids.copy_(torch.randint_like(ids,-1,128));vector.copy_(torch.randn_like(vector)*.1)
            graph.replay();assert torch.equal(replayed,original()),(rows,mix_one,steer,'changed replay')
        # Independent CUDA graphs remove Python scheduling from operator timing.
        ref_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(ref_graph):ref_output=original()
        for g in (ref_graph,graph):
            for _ in range(5):g.replay()
        torch.cuda.synchronize()
        def micros(g):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(200):g.replay()
            end.record();end.synchronize();return start.elapsed_time(end)*1000/200
        ref_us=micros(ref_graph);fused_us=micros(graph)
        records.append(dict(rows=rows,mix_one=mix_one,steering=steer,status='PASS',
            original_chain_us=ref_us,fused_us=fused_us,operator_speedup=ref_us/fused_us,
            exact_native=True,changed_graph_replays=3,independent_torch_gold=True))
        out.write_text(json.dumps(dict(status='RUNNING',records=records),indent=2))
out.write_text(json.dumps(dict(status='PASS',records=records,model_tps_measured=False),indent=2))
print(out.read_text(),flush=True)
