# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in gfx950 MXFP8 expert rows using AITER's native FlyDSL GEMMs.

This is a candidate backend, not a validated performance default. Routing and
both GEMMs run FlyDSL; MXFP8 activation quantization uses AITER's HIP launcher.
No Triton, CUTLASS, or automatic backend fallback is selected here. The caller
owns the existing Raon combine and BF16 shared expert.
"""

import json
import os
from functools import cache

import torch


@cache
def tile_config():
    """Explicit opt-in geometry overrides; checkpoint packing stays fixed."""
    raw = os.environ.get("RAON_MXFP8_FLYDSL_TILE_CONFIG", "{}")
    config = json.loads(raw)
    allowed = {
        "block_m": (16, 32),
        "fused_route_prepare": (0, 1),
        "stage1_tile_n": (32, 64, 128, 256),
        "stage1_k_wave": (1, 2, 4),
        "stage1_waves_per_eu": (1, 2, 3, 4),
        "stage2_tile_n": (64, 128, 256),
    }
    if not isinstance(config, dict) or set(config) - allowed.keys():
        raise ValueError("Unknown native MXFP8 MoE tile configuration")
    for key, value in config.items():
        if type(value) is not int or value not in allowed[key]:
            raise ValueError(f"Unsupported native MXFP8 MoE geometry: {key}={value}")
    return config


def fused_route_prepare_enabled(tokens):
    return tokens <= 32 and tile_config().get("fused_route_prepare", 0) == 1


def enabled():
    backend = os.environ.get("RAON_MOE_BACKEND", "legacy")
    if backend not in ("legacy", "flydsl"):
        raise ValueError(f"Unknown Raon MoE backend: {backend}")
    return backend == "flydsl"


def padded_intermediate(intermediate):
    # The FlyDSL stage1 N tile is 256 per gate/up half. Raon's I=1088 is
    # not divisible by it; padding the combined 2*I axis would mix the halves.
    return ((intermediate + 255) // 256) * 256


@cache
def _ops():
    from aiter import fused_dynamic_mxfp8_quant_moe_sort
    from aiter.fused_moe import _flydsl_moe_sorting
    from aiter.ops.flydsl.moe_kernels import (
        flydsl_moe_stage1,
        flydsl_moe_stage2,
    )
    from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight

    return (
        _flydsl_moe_sorting,
        fused_dynamic_mxfp8_quant_moe_sort,
        flydsl_moe_stage1,
        flydsl_moe_stage2,
        moe_shuffle_weight,
        moe_shuffle_scale,
    )


def _source_weights(w13, w2):
    # Reuse the exact checkpoint bytes. Never dequantize/requantize experts.
    # The compact loader has already registered these four tensors as buffers.
    from .mxfp8_w8a8_triton import _WQ_CACHE

    entry = _WQ_CACHE.get(id(w13))
    if entry is None or entry[4] is not w13 or entry[5] is not w2:
        raise RuntimeError("FlyDSL requires preloaded compact MXFP8 checkpoint weights")
    return entry[:4]


def _identity(tensor):
    try:
        version = tensor._version
    except RuntimeError:
        # Inference tensors have no version counter. The compact loader rejects
        # refits, and model._invalidate_refit_derived drops the packed cache.
        version = None
    return (id(tensor), tensor.data_ptr(), version, tuple(tensor.shape))


def weight_stamp(w13, w2):
    return tuple(_identity(t) for t in _source_weights(w13, w2))


def packed_weights(w13, w2):
    source = _source_weights(w13, w2)
    stamp = tuple(_identity(t) for t in source)
    stored = getattr(w13, "_raon_flydsl_weights", None)
    if stored is not None and stored[0] == stamp:
        return stored[1]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up FlyDSL weight packing before graph capture")
    q13, s13, q2, s2 = source
    experts, twice_i, hidden = q13.shape
    intermediate = twice_i // 2
    padded = padded_intermediate(intermediate)
    if (
        twice_i % 2
        or hidden % 256
        or intermediate % 32
        or tuple(q2.shape) != (experts, hidden, intermediate)
        or tuple(s13.shape) != (experts, twice_i, hidden // 32)
        or tuple(s2.shape) != (experts, hidden, intermediate // 32)
        or q13.dtype != torch.float8_e4m3fn
        or q2.dtype != torch.float8_e4m3fn
        or s13.dtype != torch.uint8
        or s2.dtype != torch.uint8
    ):
        raise ValueError("Unsupported FlyDSL MXFP8 checkpoint shape or scale layout")

    # Zero FP8 payload and finite E8M0 unit scales for every padding lane.
    p13 = torch.zeros(
        (experts, 2 * padded, hidden), dtype=torch.uint8, device=q13.device
    ).view(q13.dtype)
    ps13 = torch.full(
        (experts, 2 * padded, hidden // 32),
        127,
        dtype=torch.uint8,
        device=q13.device,
    )
    p2 = torch.zeros(
        (experts, hidden, padded), dtype=torch.uint8, device=q2.device
    ).view(q2.dtype)
    ps2 = torch.full(
        (experts, hidden, padded // 32),
        127,
        dtype=torch.uint8,
        device=q2.device,
    )
    for src_start, dst_start in ((0, 0), (intermediate, padded)):
        p13[:, dst_start : dst_start + intermediate].copy_(
            q13[:, src_start : src_start + intermediate]
        )
        ps13[:, dst_start : dst_start + intermediate].copy_(
            s13[:, src_start : src_start + intermediate]
        )
    p2[:, :, :intermediate].copy_(q2)
    ps2[:, :, : intermediate // 32].copy_(s2)
    *_, shuffle_weight, shuffle_scale = _ops()
    result = (
        shuffle_weight(p13, is_guinterleave=True, gate_up=True),
        shuffle_scale(ps13.flatten(0, 1), experts, True, True),
        shuffle_weight(p2, is_guinterleave=True, gate_up=False),
        shuffle_scale(ps2.flatten(0, 1), experts, True, False),
    )
    w13._raon_flydsl_weights = (stamp, result)
    return result


def expert_rows(x, weights, ids, w13, w2, beta, linear_beta, layer_idx=None):
    """Return BF16 [T, topk, H] with routing probabilities folded once."""
    if not torch.version.hip or not x.is_cuda:
        raise RuntimeError("The FlyDSL MXFP8 backend requires ROCm GPU tensors")
    arch = torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0]
    if arch != "gfx950":
        raise RuntimeError(f"FlyDSL Raon MXFP8 candidate supports gfx950, got {arch}")
    if x.dtype != torch.bfloat16 or x.ndim != 2 or ids.ndim != 2:
        raise ValueError("FlyDSL Raon expects BF16 [T,H] and routing [T,K]")
    if weights.shape != ids.shape or ids.shape[0] != x.shape[0]:
        raise ValueError("Routing weights/indices must match the input token count")
    if float(beta) <= 0 or linear_beta is None or float(linear_beta) <= 0:
        raise ValueError("FlyDSL SiTUv2 requires positive gate and linear beta")
    p13, s13, p2, s2 = packed_weights(w13, w2)
    sort, quant, stage1, stage2, *_ = _ops()
    config = tile_config()
    fused_a2 = os.environ.get("RAON_MXFP8_FLYDSL_FUSED_A2", "0") == "1"
    topk = ids.shape[1]
    # Sort, A1/A2 scale layout and both GEMMs share one M quantum. M16 is
    # experimental: the native builder has half-scale support, but its general
    # dispatch registry does not advertise this FP8/FP8 combination.
    block_m = config.get("block_m", 32)
    if fused_route_prepare_enabled(x.shape[0]):
        if not fused_a2:
            raise ValueError("Fused route preparation requires native fused A2")
        from .rocm_flydsl_route_prepare import prepare

        sorted_ids, sorted_weights, expert_ids, valid = prepare(
            ids, weights, p13.shape[0], block_m
        )
    else:
        route_weights = weights.bfloat16().float() if fused_a2 else weights.float()
        sorted_ids, sorted_weights, expert_ids, valid, _ = sort(
            ids.to(torch.int32),
            route_weights,
            p13.shape[0],
            x.shape[1],
            x.dtype,
            block_m,
            None,
            None,
            accumulate=False,
        )
    quant_args = dict(
        sorted_ids=sorted_ids,
        num_valid_ids=valid,
        token_num=x.shape[0],
        topk=topk,
        block_size=block_m,
        sorted_weights=sorted_weights,
        num_experts_upper_bound=p13.shape[0],
    )
    a1, as1 = quant(x, **quant_args)
    intermediate = stage1(
        a1,
        p13,
        sorted_ids,
        expert_ids,
        valid,
        topk=topk,
        tile_m=block_m,
        tile_n=config.get("stage1_tile_n", 256),
        tile_k=256,
        a_dtype="fp8",
        b_dtype="fp8",
        out_dtype="fp8" if fused_a2 else "bf16",
        act="situv2",
        situ_beta=float(beta),
        situ_linear_beta=float(linear_beta),
        w1_scale=s13,
        a1_scale=as1,
        sorted_weights=sorted_weights,
        gate_mode="interleave",
        use_async_copy=True,
        k_batch=1,
        k_wave=config.get("stage1_k_wave", 1),
        waves_per_eu=config.get("stage1_waves_per_eu", 3),
    )
    # AITER materializes BF16 here; the legacy Triton chain quantizes directly
    # from FP32. This is a numerical boundary to validate, not bit-neutral glue.
    # Keep the probability fold before A2 quant, unlike fused_moe's usual stage2.
    if fused_a2:
        a2, as2 = intermediate
    else:
        a2, as2 = quant(intermediate.reshape(-1, p2.shape[-1]), **quant_args)
    rows = stage2(
        a2.view(x.shape[0], topk, -1),
        p2,
        sorted_ids,
        expert_ids,
        valid,
        topk=topk,
        tile_m=block_m,
        tile_n=config.get("stage2_tile_n", 128),
        tile_k=256,
        a_dtype="fp8",
        b_dtype="fp8",
        out_dtype="bf16",
        mode="reduce",
        w2_scale=s2,
        a2_scale=as2,
        sorted_weights=None,
        sort_block_m=block_m,
        return_per_slot=True,
        use_async_copy=True,
    )
    if not getattr(w13, "_raon_flydsl_live", False):
        w13._raon_flydsl_live = True
        print(
            f"RAON_MXFP8_FLYDSL LIVE layer={layer_idx} shape={tuple(x.shape)} "
            f"intermediate={w2.shape[-1]} padded={p2.shape[-1]} "
            "sort=flydsl gemm1=flydsl gemm2=flydsl quant=aiter_hip "
            f"fused_a2={int(fused_a2)} prob_fold=stage1 "
            f"tiles={json.dumps(config, sort_keys=True, separators=(',', ':'))} "
            "combine=existing backend_fallback=0",
            flush=True,
        )
    return rows
