# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse the native mHC split reduction, BF16 score cast and mean square."""

from functools import cache

import torch


@cache
def _create():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import range_constexpr

    @flyc.kernel
    def finish(
        partial: fx.Tensor,
        squares: fx.Tensor,
        scores: fx.Tensor,
        mean: fx.Tensor,
        rows: fx.Int32,
    ):
        row = fx.Int64(fx.block_idx.x)
        col = fx.Int64(fx.thread_idx.x)
        if col < 17:
            acc = fx.Float32(0.0)
            for split in range_constexpr(16):
                acc = acc + partial[(split * fx.Int64(rows) + row) * 32 + col]
            scores[row * 17 + col] = acc.to(fx.BFloat16)
        if col == 0:
            total = fx.Float32(0.0)
            for split in range_constexpr(16):
                total = total + squares[split * fx.Int64(rows) + row]
            mean[row] = total / fx.Float32(10240.0)

    @flyc.jit
    def launch(
        partial: fx.Tensor,
        squares: fx.Tensor,
        scores: fx.Tensor,
        mean: fx.Tensor,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        finish(partial, squares, scores, mean, rows).launch(
            grid=(rows,), block=(64,), stream=stream
        )

    return launch


def reduce(partial, squares):
    rows = partial.shape[1]
    if (
        partial.shape != (16, rows, 32)
        or squares.shape != (16, rows)
        or partial.dtype != torch.float32
        or squares.dtype != torch.float32
        or not partial.is_contiguous()
        or not squares.is_contiguous()
        or partial.device != squares.device
        or not partial.is_cuda
    ):
        raise ValueError("Invalid native mHC split buffers")
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    scores = torch.empty((rows, 17), device=partial.device, dtype=torch.bfloat16)
    mean = torch.empty(rows, device=partial.device, dtype=torch.float32)
    if rows:
        _run_compiled(
            _create(),
            partial.view(-1),
            squares.view(-1),
            scores.view(-1),
            mean,
            rows,
            torch.cuda.current_stream(partial.device).cuda_stream,
        )
    return scores, mean
