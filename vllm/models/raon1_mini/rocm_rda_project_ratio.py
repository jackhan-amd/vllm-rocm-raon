# SPDX-License-Identifier: Apache-2.0
"""Opt-in native BF16 RDA write projection and stable gate ratio fusion."""
import os
import torch
_announced=set()

def forward(x,weight,beta):
    from .rocm_dense import _validate_device,configuration
    if not (x.is_cuda and x.dtype==weight.dtype==beta.dtype==torch.bfloat16
        and x.ndim==2 and 1<=x.shape[0]<=64 and x.shape[1]==2560
        and weight.shape==(40,2560) and beta.shape==(1,x.shape[0],40)
        and x.is_contiguous() and weight.is_contiguous() and beta.stride(-1)==1
        and beta.stride(-2)>=40 and x.device==weight.device==beta.device):
        raise ValueError('RDA projection/ratio requires BF16 TP1 [T,2560],[40,2560],[1,T,40], T<=64')
    _validate_device(x.device)
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled,ptr_arg
    from flydsl.expr import Stream
    from .rocm_rda_project_ratio_kernel import _compile
    output=torch.empty((1,x.shape[0],40),device=x.device,dtype=torch.float32)
    for start in range(0,x.shape[0],32):
        end=min(x.shape[0],start+32);rows=end-start
        config=configuration(rows,40,2560)
        if config is None:raise RuntimeError('No validated native RDA projection layout')
        launch=_compile(40,2560,config['block_m'],config['block_n'],config['block_k'],config['stages'],config['k_waves'],x.device.index)
        _run_compiled(launch,ptr_arg(output[:,start:end]),ptr_arg(x[start:end]),ptr_arg(weight),ptr_arg(beta[:,start:end]),
            beta.stride(-2),rows,Stream(torch.cuda.current_stream(x.device)))
    key=(x.shape[0],beta.stride(-2))
    if key not in _announced:
        _announced.add(key)
        print(f'RAON_RDA_PROJECT_RATIO LAUNCHED rows={x.shape[0]} beta_stride={beta.stride(-2)} native=aiter_flydsl delta_round=bf16 output=fp32 fallback=0',flush=True)
    return output

def try_forward(projection,x,beta):
    backend=os.environ.get('RAON_ROCM_RDA_PROJECT_RATIO','legacy')
    if backend not in ('legacy','flydsl'):raise ValueError('Unknown RDA projection/ratio backend: '+backend)
    if backend=='legacy':return None
    weight=getattr(projection,'weight',None)
    if (not isinstance(weight,torch.Tensor) or weight.dtype!=torch.bfloat16
        or weight.shape!=(40,2560) or x.shape[0]>64 or getattr(projection,'bias',None) is not None):
        return None
    return forward(x,weight,beta)
