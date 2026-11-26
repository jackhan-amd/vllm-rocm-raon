# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-batch native route packing with Raon's BF16 probability boundary."""

from functools import lru_cache

import torch

_ANNOUNCED = False


@lru_cache(maxsize=32)
def _create(tokens, topk, experts, unit, blocks, ids_dtype, device_index):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels.moe_sorting_kernel import (
        _allwave_inclusive_prefix_sum,
    )
    from flydsl.expr import range_constexpr

    routes = tokens * topk

    @fx.struct
    class Scratch:
        counts: fx.Array[fx.Int32, 128, 16]
        offsets: fx.Array[fx.Int32, 128, 16]
        positions: fx.Array[fx.Int32, 256, 16]
        scan: fx.Array[fx.Int32, 4, 16]

    @flyc.kernel
    def raon_mxfp8_route_prepare(
        ids: fx.Tensor,
        weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_experts: fx.Tensor,
        valid_count: fx.Tensor,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        shared = fx.SharedAllocator().allocate(Scratch).peek()
        if tid < 128:
            shared.counts[tid] = fx.Int32(0)
        if tid < blocks:
            sorted_experts[tid] = fx.Int32(0)
        # Native GEMM launch grids use a host-side capacity bound. Keep every
        # inactive metadata entry safe for their address/prefetch preparation.
        for j in range_constexpr((blocks * unit + 255) // 256):
            row = tid + j * 256
            if row < blocks * unit:
                sorted_ids[row] = fx.Int32(-16777216 + tokens)
                sorted_weights[row] = fx.Float32(0.0)
        fx.gpu.barrier()
        safe_route = (tid < routes).select(tid, 0)
        raw_expert = fx.Int64(ids[safe_route])
        live = (tid < routes) & (raw_expert >= 0) & (raw_expert < experts)
        expert = fx.Int32(raw_expert)
        if live:
            position = fx.atomic_add(
                shared.counts.ptr + fx.Int64(expert),
                fx.Int32(1),
                syncscope=fx.rocdl.SyncScope.Workgroup,
            )
            shared.positions[tid] = position
        fx.gpu.barrier()
        count = shared.counts[tid % 128]
        padded = (tid < experts).select(((count + unit - 1) // unit) * unit, 0)
        _, inclusive = _allwave_inclusive_prefix_sum(
            padded, tid % 64, tid // 64, shared.scan.ptr, 4, 64
        )
        if tid < experts:
            shared.offsets[tid] = inclusive - padded
        if tid == experts - 1:
            valid_count[0] = inclusive
            valid_count[1] = fx.Int32(tokens)
        fx.gpu.barrier()
        if tid < experts:
            begin = shared.offsets[tid]
            for j in range_constexpr((routes + unit - 1) // unit):
                if j * unit < padded:
                    sorted_experts[begin // unit + j] = tid
        if live:
            row = shared.offsets[expert] + shared.positions[tid]
            sorted_ids[row] = (tid // topk) | ((tid % topk) << 24)
            sorted_weights[row] = (
                fx.Float32(weights[tid]).to(fx.BFloat16).to(fx.Float32)
            )

    @flyc.jit
    def launch(
        ids: fx.Tensor,
        weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_experts: fx.Tensor,
        valid_count: fx.Tensor,
        stream: fx.Stream,
    ):
        raon_mxfp8_route_prepare(
            ids, weights, sorted_ids, sorted_weights, sorted_experts, valid_count
        ).launch(grid=(1,), block=(256,), stream=stream)

    return launch


def prepare(ids, weights, experts, unit):
    """Pack valid router selections; order within one expert is unspecified."""
    global _ANNOUNCED
    tokens, topk = ids.shape
    if (
        not ids.is_cuda
        or ids.dtype not in (torch.int32, torch.int64)
        or not ids.is_contiguous()
        or weights.shape != ids.shape
        or weights.dtype != torch.float32
        or weights.device != ids.device
        or not weights.is_contiguous()
        or not 1 <= tokens <= 32
        or not 1 <= experts <= 128
        or not 1 <= topk <= 8
        or unit not in (16, 32)
    ):
        raise ValueError("Unsupported native small-batch route preparation")
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    blocks = (tokens * topk + experts * unit - topk + unit - 1) // unit
    sorted_ids = torch.empty(blocks * unit, device=ids.device, dtype=torch.int32)
    sorted_weights = torch.empty_like(sorted_ids, dtype=torch.float32)
    sorted_experts = torch.empty(blocks, device=ids.device, dtype=torch.int32)
    valid = torch.empty(2, device=ids.device, dtype=torch.int32)
    _run_compiled(
        # AITER's fast launcher keeps one compiled signature per factory result.
        _create(tokens, topk, experts, unit, blocks, ids.dtype, ids.device.index),
        ids.view(-1),
        weights.view(-1),
        sorted_ids,
        sorted_weights,
        sorted_experts,
        valid,
        torch.cuda.current_stream(ids.device).cuda_stream,
    )
    if not _ANNOUNCED:
        _ANNOUNCED = True
        print(
            "RAON_FLYDSL_ROUTE_PREPARE LAUNCHED sort=flydsl "
            f"probability_round=bf16 input_ids={ids.dtype} packing=token24_slot8",
            flush=True,
        )
    return sorted_ids, sorted_weights, sorted_experts, valid
