# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raon's stable expert-ID-ordered, unweighted BF16 slot reduction."""

from functools import lru_cache

import torch

_ANNOUNCED = False


@lru_cache(maxsize=32)
def _create(topk, hidden, ids_dtype, device_index):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels.gdn_decode_verify import _load_vec, _store_vec
    from flydsl.expr import range_constexpr

    @flyc.kernel
    def combine(values: fx.Tensor, ids: fx.Tensor, out: fx.Tensor):
        row = fx.Int64(fx.block_idx.x)
        col = fx.Int64(fx.block_idx.y) * 512 + fx.Int64(fx.thread_idx.x) * 4
        keys = [fx.Int64(ids[row * topk + j]) for j in range_constexpr(topk)]
        slots = [fx.Int64(j) for j in range_constexpr(topk)]
        # Stable sorting: equal IDs never swap, and negative IDs remain values.
        for end in range_constexpr(topk - 1, 0, -1):
            for j in range_constexpr(end):
                swap = keys[j] > keys[j + 1]
                left_key, right_key = keys[j], keys[j + 1]
                left_slot, right_slot = slots[j], slots[j + 1]
                keys[j] = swap.select(right_key, left_key)
                keys[j + 1] = swap.select(left_key, right_key)
                slots[j] = swap.select(right_slot, left_slot)
                slots[j + 1] = swap.select(left_slot, right_slot)
        vb = fx.rocdl.make_buffer_tensor(values)
        ob = fx.rocdl.make_buffer_tensor(out)
        if col < hidden:
            acc = fx.full(4, 0.0, fx.Float32)
            for j in range_constexpr(topk):
                offset = (row * topk + slots[j]) * hidden + col
                acc = acc + _load_vec(vb, offset, 4, fx.BFloat16).to(fx.Float32)
            _store_vec(ob, row * hidden + col, acc.to(fx.BFloat16), fx.BFloat16)

    @flyc.jit
    def launch(
        values: fx.Tensor,
        ids: fx.Tensor,
        out: fx.Tensor,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        combine(values, ids, out).launch(
            grid=(rows, (hidden + 511) // 512), block=(128,), stream=stream
        )

    return launch


def combine_unweighted(values, ids):
    """Preserve BF16 slot boundaries and sequential FP32 addition, without weights."""
    global _ANNOUNCED
    if (
        values.ndim != 3
        or values.dtype != torch.bfloat16
        or not values.is_cuda
        or not values.is_contiguous()
        or ids.shape != values.shape[:2]
        or ids.dtype not in (torch.int32, torch.int64)
        or ids.device != values.device
        or not ids.is_contiguous()
        or values.shape[1] < 1
        or values.shape[1] > 16
        or values.shape[2] % 4
    ):
        raise ValueError(
            "FlyDSL combine requires contiguous BF16 slots and integer IDs"
        )
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    rows, topk, hidden = values.shape
    out = torch.empty((rows, hidden), device=values.device, dtype=values.dtype)
    if rows:
        _run_compiled(
            # AITER's fast launcher caches one compiled tensor signature.
            _create(topk, hidden, ids.dtype, ids.device.index),
            values,
            ids.view(-1),
            out,
            rows,
            torch.cuda.current_stream(values.device).cuda_stream,
        )
    if not _ANNOUNCED:
        _ANNOUNCED = True
        print(
            "RAON_FLYDSL_COMBINE LAUNCHED order=stable_expert_id acc=fp32 "
            "input=bf16 output=bf16 weighted=0 triton_compute=0",
            flush=True,
        )
    return out
