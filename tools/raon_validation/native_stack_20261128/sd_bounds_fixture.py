"""Reproduce original SD capacity mismatch, then native/Torch/cache/graph gold."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import torch
import torch.nn.functional as F
from tools.raon_validation.common import error
from tools.raon_validation.reference import convolution,rda
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

def load(name,path):
    spec=importlib.util.spec_from_file_location('vllm.models.raon1_mini.'+name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

original=load('_bounds_original','/work/sd-bounds/original_adapter.py')
candidate=load('_bounds_candidate','/work/sd-bounds/candidate_adapter.py')
rank=int(os.environ.get('RAON_REPLICA','0'))
H,D=40,128;P=H*D;STATE=H*D*D;SENTINEL=12345.
out=Path('/work/sd-bounds-results')/f'gpu-{rank}.json';out.parent.mkdir(exist_ok=True)
cases=[(backend,batch,short) for backend in ('hip_rda_auto_sd','hip_rda_fused_sd','hip_rda_vsplit_sd','hip_rda')
       for batch in (1,8,16,32) for short in (1,2,7)]
records=[]
for case_index,(backend,batch,short) in enumerate(cases):
    if case_index%8!=rank:continue
    torch.manual_seed(89100+case_index)
    padded=batch*8;short_index=6 if batch==16 else batch//2
    lengths=[8]*batch;lengths[short_index]=short;live=sum(lengths);slots=padded+5
    def rand(*shape):return (torch.randn(shape,device='cuda')*.2).bfloat16()
    raw=rand(padded,3*P);g=rand(1,padded,H,D).contiguous()
    beta,delta=rand(1,padded,H),rand(1,padded,H)
    ratio=torch.empty((1,padded,H),device='cuda',dtype=torch.float32)
    out_gate=rand(padded,H,D)
    weight=rand(3*P,4)
    a_log=torch.randn(H,device='cuda')*.1;bias=torch.randn(P,device='cuda')*.1
    norm=torch.ones(D,device='cuda',dtype=torch.bfloat16)
    initial=torch.randn((slots,H,D,D),device='cuda')*.01
    initial_conv=rand(slots,3*P,10)
    actual_backing=torch.full((slots,STATE+64),SENTINEL,device='cuda')
    reference_backing=actual_backing.clone()
    state=actual_backing[:,:STATE].view(slots,H,D,D)
    reference_state=reference_backing[:,:STATE].view_as(state)
    conv=initial_conv.clone();reference_conv=initial_conv.clone()
    ids=torch.arange(1,padded+1,device='cuda',dtype=torch.int32).view(batch,8)
    accepted=torch.ones(batch,device='cuda',dtype=torch.int32)
    cu=torch.tensor([0]+list(__import__('itertools').accumulate(lengths)),device='cuda',dtype=torch.int32)
    metadata=NS(num_actual_tokens=live,num_spec_decodes=batch,num_decodes=0,num_prefills=0,
        spec_state_indices_tensor=ids,num_accepted_tokens=accepted,spec_query_start_loc=cu)
    layer=NS(local_num_heads=H,head_dim=D,conv1d=NS(weight=weight,bias=None),
        o_norm=NS(weight=norm,eps=1e-6),A_log=a_log,dt_bias=bias,gate_lower_bound=-5.,
        rocm_decode_backend=backend)
    actual=torch.full((1,padded,H,D),float('nan'),device='cuda',dtype=torch.bfloat16)
    expected=torch.empty((1,live,H,D),device='cuda',dtype=torch.bfloat16)
    inputs=(raw,g,out_gate,beta,ratio)
    def update_ratio():ratio.copy_(torch.exp(F.logsigmoid(beta.float()+delta.float())-F.logsigmoid(beta.float())))
    def reset():
        state.copy_(initial);reference_state.copy_(initial)
        conv.copy_(initial_conv);reference_conv.copy_(initial_conv)
    def cache(history,recurrent):
        return (history if is_conv_state_dim_first() else history.transpose(-1,-2),recurrent)
    def call(mod,history,recurrent,output,trim=False):
        layer.kv_cache=cache(history,recurrent)
        supplied=(raw[:live],g[:,:live],out_gate[:live],beta[:,:live],ratio[:,:live]) if trim else inputs
        assert mod.try_forward(layer,metadata,supplied,output)
    def gold():
        output=torch.zeros((live,H,D),device='cuda',dtype=torch.bfloat16)
        recurrent=initial.clone();history=initial_conv.clone()
        for n,length in enumerate(lengths):
            offset=int(accepted[n])-1;first=int(ids[n,0]);read=int(ids[n,offset])
            lo,hi=int(cu[n]),int(cu[n+1])
            old=initial_conv[first:first+1,:,offset:offset+3]
            projected,_=convolution(raw[None,lo:hi],old,weight)
            q,k,v=[part.reshape(1,length,H,D) for part in projected.chunk(3,-1)]
            values,states=rda(q,k,v,g[:,lo:hi],beta[:,lo:hi],delta[:,lo:hi],a_log,bias,initial[read:read+1])
            rounded=values.bfloat16().float()
            output[lo:hi]=(rounded*torch.rsqrt(rounded.square().mean(-1,keepdim=True)+1e-6)*out_gate[lo:hi].float().sigmoid()).bfloat16()[0]
            for t in range(length):recurrent[ids[n,t]]=states[0,t]
            history[first,:,:2]=old[0,:,1:]
            history[first,:,2:2+length]=raw[lo:hi].T
        return output,recurrent,history
    def check(kind):
        target,gold_state,gold_conv=gold()
        checks=dict(native_output_exact=bool(torch.equal(actual[:,:live],expected)),
            native_state_exact=bool(torch.equal(state,reference_state)),
            native_conv_exact=bool(torch.equal(conv,reference_conv)),
            padded_output_zero=bool(torch.all(actual[:,live:]==0)),
            state_gap_untouched=bool(torch.all(actual_backing[:,STATE:]==SENTINEL)),
            torch_output=error(actual[0,:live],target),torch_state=error(state,gold_state),
            torch_conv_exact=bool(torch.equal(conv,gold_conv)))
        record=dict(backend=backend,batch=batch,short=short,live=live,padded=padded,
            short_index=short_index,kind=kind,checks=checks)
        records.append(record)
        out.write_text(json.dumps(dict(status='RUNNING',records=records),indent=2))
        assert all(v.get('pass') if isinstance(v,dict) else v for v in checks.values()),record
    update_ratio();reset()
    try:call(original,conv,state,actual)
    except RuntimeError as error_value:
        assert 'Invalid SD output' in str(error_value),str(error_value)
    else:raise AssertionError('Original capacity mismatch did not reproduce')
    assert torch.equal(state,initial) and torch.equal(conv,initial_conv),'Original guard mutated state'
    call(original,reference_conv,reference_state,expected,trim=True)
    call(candidate,conv,state,actual);check('direct_reproduced_and_repaired')
    # Every warm/capture uses disposable fixture state, never a live model cache.
    reset();call(candidate,conv,state,actual);torch.cuda.synchronize();reset()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):call(candidate,conv,state,actual)
    for replay in range(3):
        raw.add_(.005);g.add_(.01);beta.add_(.005);delta.sub_(.005);update_ratio()
        accepted.fill_(1+(replay+case_index)%8)
        # Move the short request while retaining the same packed live row count.
        lengths=[8]*batch;lengths[(short_index+replay)%batch]=short
        cu.copy_(torch.tensor([0]+list(__import__('itertools').accumulate(lengths)),device='cuda',dtype=torch.int32))
        reset();actual.fill_(float('nan'));graph.replay()
        call(original,reference_conv,reference_state,expected,trim=True)
        check(f'changed_graph_replay_{replay}')
out.write_text(json.dumps(dict(status='PASS',records=records,model_forward_count=0,
    original_guard_reproduced=True),indent=2))
print(json.dumps(dict(status='PASS',rank=rank,records=len(records))),flush=True)
