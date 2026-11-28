# SPDX-License-Identifier: Apache-2.0
"""Experimental native MoE combine/shared/steering/mHC launch boundary.

Expert GEMMs and routing retain their existing native implementation. This
boundary removes their materialized combined/steered output, preserving each
BF16 rounding seam and the selected Raon mHC arithmetic schedule.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
import os

import torch


@dataclass(frozen=True)
class DeferredMoE:
    values: torch.Tensor
    ids: torch.Tensor
    shared: torch.Tensor
    alpha: torch.Tensor | None = None
    steering: torch.Tensor | None = None


def enabled(rows):
    backend=os.environ.get('RAON_ROCM_MOE_MHC_FINISH','legacy')
    if backend not in ('legacy','flydsl'):
        raise ValueError(f'Unknown MoE/mHC finish backend: {backend}')
    return backend=='flydsl' and 1<=rows<=128


def attach_steering(value,vector):
    if vector is None:return value
    from . import effort_steering
    if effort_steering.G is None:raise RuntimeError('Steering token coefficients uninitialized')
    return replace(value,alpha=effort_steering.G[:value.values.shape[0]],steering=vector)


@lru_cache(maxsize=32)
def _create(topk,hidden,mix_one,steering,ids_dtype,device):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import range_constexpr

    @flyc.kernel
    def raon_moe_mhc_finish(values:fx.Tensor,ids:fx.Tensor,shared:fx.Tensor,carrier:fx.Tensor,
               coeff:fx.Tensor,alpha:fx.Tensor,vector:fx.Tensor,out:fx.Tensor):
        row=fx.Int64(fx.block_idx.x)
        tid=fx.Int64(fx.thread_idx.x)
        keys=[fx.Int64(ids[row*topk+j]) for j in range_constexpr(topk)]
        slots=[fx.Int64(j) for j in range_constexpr(topk)]
        for end in range_constexpr(topk-1,0,-1):
            for j in range_constexpr(end):
                swap=keys[j]>keys[j+1]
                a,b=keys[j],keys[j+1];i,k=slots[j],slots[j+1]
                keys[j]=swap.select(b,a);keys[j+1]=swap.select(a,b)
                slots[j]=swap.select(k,i);slots[j+1]=swap.select(i,k)
        for offset in range_constexpr(2):
            col=fx.Int64(fx.block_idx.y)*256+tid+offset*128
            acc=fx.Float32(0.0)
            for j in range_constexpr(topk):
                acc=acc+values[(row*topk+slots[j])*hidden+col].to(fx.Float32)
            routed=acc.to(fx.BFloat16).to(fx.Float32)
            shared_value=shared[row*hidden+col].to(fx.Float32)
            branch=(routed+shared_value).to(fx.BFloat16).to(fx.Float32)
            if steering:
                a=alpha[row]
                c=vector[col]
                steered=(branch+a*c).to(fx.BFloat16).to(fx.Float32)
                branch=(a==fx.Float32(0.0)).select(branch,steered)
            s0=carrier[row*4*hidden+col].to(fx.Float32)
            s1=carrier[row*4*hidden+hidden+col].to(fx.Float32)
            s2=carrier[row*4*hidden+2*hidden+col].to(fx.Float32)
            s3=carrier[row*4*hidden+3*hidden+col].to(fx.Float32)
            for i in range_constexpr(4):
                start=row*20+4+i*4
                h0,h1,h2,h3=coeff[start],coeff[start+1],coeff[start+2],coeff[start+3]
                value=fx.fma(h3,s3,fx.fma(h2,s2,fx.fma(h1,s1,h0*s0)))
                if mix_one:value=((h0*s0+h1*s1)+h2*s2)+h3*s3
                value=value+coeff[row*20+i]*branch
                out[row*4*hidden+i*hidden+col]=value.to(fx.BFloat16)

    @flyc.jit
    def launch(values:fx.Tensor,ids:fx.Tensor,shared:fx.Tensor,carrier:fx.Tensor,
               coeff:fx.Tensor,alpha:fx.Tensor,vector:fx.Tensor,out:fx.Tensor,
               rows:fx.Int32,stream:fx.Stream):
        raon_moe_mhc_finish(values,ids,shared,carrier,coeff,alpha,vector,out).launch(
            grid=(rows,hidden//256),block=(128,),stream=stream)
    return launch


def finish(value,carrier,coeff):
    from . import mhc
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
    v,ids,shared=value.values,value.ids,value.shared
    rows,topk,hidden=v.shape
    if not (1<=rows<=128 and topk==6 and hidden==2560 and
            v.dtype==shared.dtype==carrier.dtype==torch.bfloat16 and
            v.is_cuda and all(t.device==v.device and t.is_contiguous() for t in (v,ids,shared,carrier,coeff)) and
            ids.dtype in (torch.int32,torch.int64) and ids.shape==(rows,topk) and
            shared.shape==(rows,hidden) and carrier.shape==(rows,4,hidden) and
            coeff.shape==(rows,20) and coeff.dtype==torch.float32):
        raise ValueError('MoE/mHC finish requires local BF16 top6 slots, shared and carrier with FP32 coefficients')
    steer=value.steering is not None
    if steer and not (value.alpha is not None and value.alpha.dtype==value.steering.dtype==torch.float32 and
        value.alpha.numel()==rows and value.steering.shape==(hidden,) and
        all(t.device==v.device and t.is_contiguous() for t in (value.alpha,value.steering))):
        raise ValueError('MoE/mHC steering requires contiguous FP32 row coefficients and vector')
    mix_one=mhc._FIXED_MIX_ONE if mhc._BATCH_INVARIANT else rows==1
    out=torch.empty_like(carrier)
    _run_compiled(_create(topk,hidden,mix_one,steer,ids.dtype,v.device.index),
        v.view(-1),ids.view(-1),shared.view(-1),carrier.view(-1),coeff.view(-1),
        value.alpha if steer else coeff,value.steering if steer else coeff,
        out.view(-1),rows,torch.cuda.current_stream(v.device).cuda_stream)
    return out
