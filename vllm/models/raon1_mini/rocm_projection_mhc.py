# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in output-projection/mHC fusion shared by Raon RDA and MLA.

Only BF16, bias-free, unquantized TP1 projections use this experiment. Other
layouts and wider rows retain the original projection and mHC implementations.
Selection uses actual padded token rows, never the client's concurrency label.
"""

import os
from dataclasses import dataclass

import torch

_ANNOUNCED = set()


@dataclass(frozen=True)
class DeferredProjection:
    """Explicit attention-to-mHC handoff; contains no recurrent state."""

    inputs: torch.Tensor
    projection: torch.nn.Module


def enabled():
    """Return the opt-in verdict, rejecting misspelled backends."""
    backend = os.environ.get("RAON_ROCM_ATTN_PROJECT_MHC", "legacy")
    if backend not in ("legacy", "flydsl"):
        raise ValueError(f"Unknown attention projection/mHC backend: {backend}")
    return backend == "flydsl"


def project_mix(x, weight, carrier, coeff):
    """Return a native fused carrier, or None for an unsupported GEMM layout."""
    from . import mhc
    from .rocm_dense import _validate_device, configuration

    if (
        x.ndim != 2
        or weight.ndim != 2
        or x.dtype != torch.bfloat16
        or weight.dtype != x.dtype
        or not x.is_cuda
        or weight.device != x.device
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or weight.shape[0] != 2560
        or weight.shape[1] != x.shape[1]
    ):
        return None
    rows = x.shape[0]
    if (
        carrier.shape != (rows, 4, 2560)
        or carrier.dtype != x.dtype
        or carrier.device != x.device
        or not carrier.is_contiguous()
        or coeff.shape != (rows, 20)
        or coeff.dtype != torch.float32
        or coeff.device != x.device
        or not coeff.is_contiguous()
    ):
        raise ValueError("Projection/mHC requires BF16 [T,4,2560], FP32 [T,20]")
    config = configuration(rows, weight.shape[0], weight.shape[1])
    if config is None:
        return None
    if weight.shape[1] == 5120:
        # All 25 observed target attention output projections use this shape.
        # One N tile serves every supported row count. At N=2560, tile64 has
        # only 40 workgroups; tile16 exposes 160 independent workgroups and
        # lowers per-workgroup LDS use. Keep K/M arithmetic and BF16 seams.
        config = dict(config, block_n=16)
    _validate_device(x.device)
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg
    from flydsl.expr import Stream

    from .rocm_projection_mhc_kernel import _compile

    mix_one = mhc._FIXED_MIX_ONE if mhc._BATCH_INVARIANT else rows == 1
    launch = _compile(
        2560,
        weight.shape[1],
        config["block_m"],
        config["block_n"],
        config["block_k"],
        config["stages"],
        config["k_waves"],
        mix_one,
        x.device.index,
    )
    out = torch.empty_like(carrier)
    _run_compiled(
        launch,
        ptr_arg(out),
        ptr_arg(x),
        ptr_arg(weight),
        ptr_arg(carrier),
        ptr_arg(coeff),
        rows,
        Stream(torch.cuda.current_stream(x.device)),
    )
    key = (rows, weight.shape[1], mix_one)
    if key not in _ANNOUNCED:
        _ANNOUNCED.add(key)
        print(
            f"RAON_ATTN_PROJECT_MHC LAUNCHED M={rows} N=2560 "
            f"K={weight.shape[1]} backend=aiter_derived_flydsl "
            f"mix_one={int(mix_one)} gemm_bf16_seam=preserved launches=1",
            flush=True,
        )
    return out


def try_finish(deferred, carrier, coeff):
    """Fuse only a local unquantized projection; preserve quantized dispatch."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    projection = deferred.projection
    if (
        not enabled()
        or getattr(projection, "tp_size", None) != 1
        or not isinstance(
            getattr(projection, "quant_method", None), UnquantizedLinearMethod
        )
        or getattr(projection, "bias", None) is not None
    ):
        return None
    return project_mix(deferred.inputs, projection.weight, carrier, coeff)
