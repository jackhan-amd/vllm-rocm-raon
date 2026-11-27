# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in native Raon chart/aggregate/mix; not AITER's Sinkhorn mHC gate."""

import os
from functools import cache

import torch

_BACKEND = os.environ.get("RAON_ROCM_MHC_GATE", "legacy")
_ANNOUNCED = False


def enabled():
    if _BACKEND not in ("legacy", "flydsl"):
        raise ValueError(f"Unknown Raon mHC gate: {_BACKEND}")
    return _BACKEND == "flydsl"


@cache
def _create(agg_tree: bool, mix_one: bool, chart_fma: bool, fuse_finish: bool):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import range_constexpr

    @flyc.kernel
    def head(
        s: fx.Tensor,
        scores: fx.Tensor,
        mean: fx.Tensor,
        a: fx.Tensor,
        b: fx.Tensor,
        q: fx.Tensor,
        h0: fx.Tensor,
        neg_h0: fx.Tensor,
        eye: fx.Tensor,
        gate: fx.Tensor,
        coeff: fx.Tensor,
        aggregate: fx.Tensor,
        rows: fx.Int32,
    ):
        row = fx.Int64(fx.block_idx.x)
        tile = fx.Int64(fx.block_idx.y)
        tid = fx.Int64(fx.thread_idx.x)
        lane = tid % 64
        r = (lane < 17).select(lane, fx.Int64(0))
        score = fx.Float32(0.0)
        square_mean = fx.Float32(0.0)
        if fuse_finish:
            total = fx.Float32(0.0)
            for split in range_constexpr(16):
                score = score + scores[(split * fx.Int64(rows) + row) * 32 + r]
                total = total + mean[split * fx.Int64(rows) + row]
            # Preserve the separately launched finisher's BF16 score seam.
            score = score.to(fx.BFloat16).to(fx.Float32)
            square_mean = total / fx.Float32(10240.0)
        else:
            score = scores[row * 17 + r].to(fx.Float32)
            square_mean = mean[row]
        inverse = fx.rsqrt(square_mean + fx.Float32(1e-5))
        pre = a[r] * (score * inverse) + b[r]
        sig = fx.Float32(1.0) / (fx.Float32(1.0) + fx.exp(-pre))
        # All lanes participate in the shuffles, including lanes >= 17.
        p0 = fx.gpu.shuffle_idx(sig, 0, 64)
        p1 = fx.gpu.shuffle_idx(sig, 1, 64)
        p2 = fx.gpu.shuffle_idx(sig, 2, 64)
        p3 = fx.gpu.shuffle_idx(sig, 3, 64)
        post = fx.gpu.shuffle_idx(sig, lane % 4 + 4, 64) * fx.Float32(2.0)
        i = (lane // 4) % 4
        j = lane % 4
        q0, q1, q2 = q[i * 3], q[i * 3 + 1], q[i * 3 + 2]
        z0 = fx.gpu.shuffle_idx(pre, 8, 64)
        z1 = fx.gpu.shuffle_idx(pre, 9, 64)
        z2 = fx.gpu.shuffle_idx(pre, 10, 64)
        z3 = fx.gpu.shuffle_idx(pre, 11, 64)
        z4 = fx.gpu.shuffle_idx(pre, 12, 64)
        z5 = fx.gpu.shuffle_idx(pre, 13, 64)
        z6 = fx.gpu.shuffle_idx(pre, 14, 64)
        z7 = fx.gpu.shuffle_idx(pre, 15, 64)
        z8 = fx.gpu.shuffle_idx(pre, 16, 64)
        t0 = (q0 * z0 + q1 * z3) + q2 * z6
        t1 = (q0 * z1 + q1 * z4) + q2 * z7
        t2 = (q0 * z2 + q1 * z5) + q2 * z8
        e = (t0 * q[j * 3] + t1 * q[j * 3 + 1]) + t2 * q[j * 3 + 2]
        if chart_fma:
            e = fx.fma(t2, q[j * 3 + 2], fx.fma(t1, q[j * 3 + 1], t0 * q[j * 3]))
        index = i * 4 + j
        rho = fx.maximumf(e / neg_h0[index], fx.Float32(0.0))
        for offset in range_constexpr(4):
            rho = fx.maximumf(rho, fx.gpu.shuffle_xor(rho, 1 << offset, 64))
        residual = eye[index] + gate[0] * (h0[index] + e / (fx.Float32(1.0) + rho))
        if tile == 0:
            if tid < 16:
                coeff[row * 20 + 4 + tid] = residual
            if tid < 4:
                coeff[row * 20 + tid] = post
        for offset in range_constexpr(2):
            c = tile * 256 + tid + offset * 128
            if c < 2560:
                s0 = s[row * 10240 + c].to(fx.Float32)
                s1 = s[row * 10240 + 2560 + c].to(fx.Float32)
                s2 = s[row * 10240 + 5120 + c].to(fx.Float32)
                s3 = s[row * 10240 + 7680 + c].to(fx.Float32)
                value = fx.fma(p3, s3, fx.fma(p2, s2, fx.fma(p1, s1, p0 * s0)))
                if agg_tree:
                    value = fx.fma(p1, s1, p0 * s0) + fx.fma(p3, s3, p2 * s2)
                aggregate[row * 2560 + c] = value.to(fx.BFloat16)

    @flyc.kernel
    def mix(s: fx.Tensor, y: fx.Tensor, coeff: fx.Tensor, out: fx.Tensor):
        row = fx.Int64(fx.block_idx.x)
        tile = fx.Int64(fx.block_idx.y)
        tid = fx.Int64(fx.thread_idx.x)
        for offset in range_constexpr(2):
            c = tile * 256 + tid + offset * 128
            if c < 2560:
                s0 = s[row * 10240 + c].to(fx.Float32)
                s1 = s[row * 10240 + 2560 + c].to(fx.Float32)
                s2 = s[row * 10240 + 5120 + c].to(fx.Float32)
                s3 = s[row * 10240 + 7680 + c].to(fx.Float32)
                branch = y[row * 2560 + c].to(fx.Float32)
                for i in range_constexpr(4):
                    base = row * 20 + 4 + i * 4
                    h0, h1 = coeff[base], coeff[base + 1]
                    h2, h3 = coeff[base + 2], coeff[base + 3]
                    value = fx.fma(h3, s3, fx.fma(h2, s2, fx.fma(h1, s1, h0 * s0)))
                    if mix_one:
                        value = ((h0 * s0 + h1 * s1) + h2 * s2) + h3 * s3
                    value = value + coeff[row * 20 + i] * branch
                    out[row * 10240 + i * 2560 + c] = value.to(fx.BFloat16)

    @flyc.jit
    def launch_head(
        s: fx.Tensor,
        scores: fx.Tensor,
        mean: fx.Tensor,
        a: fx.Tensor,
        b: fx.Tensor,
        q: fx.Tensor,
        h0: fx.Tensor,
        neg_h0: fx.Tensor,
        eye: fx.Tensor,
        gate: fx.Tensor,
        coeff: fx.Tensor,
        aggregate: fx.Tensor,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        head(
            s, scores, mean, a, b, q, h0, neg_h0, eye, gate, coeff, aggregate, rows
        ).launch(grid=(rows, 10), block=(128,), stream=stream)

    @flyc.jit
    def launch_mix(
        s: fx.Tensor,
        y: fx.Tensor,
        coeff: fx.Tensor,
        out: fx.Tensor,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        mix(s, y, coeff, out).launch(grid=(rows, 10), block=(128,), stream=stream)

    return launch_head, launch_mix


def prepare_branch(site, s):
    """Return BF16 branch input and FP32 [post(4), residual(16)] coefficients."""
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    from . import mhc, rocm_mhc

    if not enabled() or rocm_mhc._BACKEND != "hip":
        raise RuntimeError("FlyDSL mHC gate requires the native HIP projection")
    if os.environ.get("RAON_ROCM_MHC_FINISHER", "torch") != "flydsl":
        raise RuntimeError("FlyDSL mHC gate requires the native FlyDSL finisher")
    if s.ndim != 3 or s.shape[1:] != (4, 2560) or s.dtype != torch.bfloat16:
        raise ValueError("FlyDSL Raon mHC requires BF16 [T,4,2560]")
    if not s.is_cuda or not s.is_contiguous() or s.device != site.phi.device:
        raise ValueError("FlyDSL Raon mHC requires contiguous inputs on phi's GPU")
    rows = s.shape[0]
    if rows < 1:
        raise ValueError("FlyDSL Raon mHC requires at least one row")
    agg_tree = mhc._FIXED_AGG_TREE if mhc._BATCH_INVARIANT else rows <= 8
    mix_one = mhc._FIXED_MIX_ONE if mhc._BATCH_INVARIANT else rows == 1
    # Keep the chart's rounding schedule independent of speculative batch size.
    chart_fma = False if mhc._BATCH_INVARIANT else rows >= 5
    project_head = os.environ.get("RAON_ROCM_MHC_PROJECT_HEAD", "legacy")
    if project_head not in ("legacy", "hip_single_cta"):
        raise ValueError(f"Unknown mHC project/head backend: {project_head}")
    if project_head == "hip_single_cta":
        if not mhc._BATCH_INVARIANT or agg_tree or chart_fma:
            raise ValueError(
                "HIP project/head requires the fixed Raon arithmetic schedule"
            )
        from .rocm_mhc_project_head import forward as project_head_forward

        aggregate, coeff = project_head_forward(site, s)
        return aggregate, coeff, _create(agg_tree, mix_one, chart_fma, False)[1]
    fuse_finish = os.environ.get("RAON_ROCM_MHC_FUSE_FINISH_HEAD", "0") == "1"
    launch_head, launch_mix = _create(agg_tree, mix_one, chart_fma, fuse_finish)
    scores, mean = rocm_mhc.project(site, s.view(rows, -1), defer_finish=fuse_finish)
    coeff = torch.empty((rows, 20), dtype=torch.float32, device=s.device)
    aggregate = torch.empty((rows, 2560), dtype=s.dtype, device=s.device)
    stream = torch.cuda.current_stream(s.device).cuda_stream
    _run_compiled(
        launch_head,
        s.view(-1),
        scores.view(-1),
        mean.view(-1),
        site.A_row.view(-1),
        site.B_row.view(-1),
        site.Q.view(-1),
        site.H0.view(-1),
        site.negH0.view(-1),
        site.g_eye.view(-1),
        site.g_gate.view(-1),
        coeff.view(-1),
        aggregate.view(-1),
        rows,
        stream,
    )
    return aggregate, coeff, launch_mix


def forward(site, s, branch):
    global _ANNOUNCED
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    aggregate, coeff, launch_mix = prepare_branch(site, s)
    y = branch(aggregate)
    if y.shape != aggregate.shape or y.dtype != s.dtype or y.device != s.device:
        raise ValueError("FlyDSL mHC branch must return BF16 [T,2560] on its GPU")
    y = y.contiguous()
    out = torch.empty_like(s)
    _run_compiled(
        launch_mix,
        s.view(-1),
        y.view(-1),
        coeff.view(-1),
        out.view(-1),
        s.shape[0],
        torch.cuda.current_stream(s.device).cuda_stream,
    )
    if not _ANNOUNCED and not torch.cuda.is_current_stream_capturing():
        fused = os.environ.get("RAON_ROCM_MHC_FUSE_FINISH_HEAD", "0")
        project_head = os.environ.get("RAON_ROCM_MHC_PROJECT_HEAD", "legacy")
        realization = (
            "gate=hip_single_cta projection=hip_single_cta "
            "finisher=hip_single_cta mix=flydsl"
            if project_head == "hip_single_cta"
            else "gate=flydsl projection=hip finisher=flydsl mix=flydsl"
        )
        print(
            f"RAON_ROCM_MHC_GATE LAUNCHED {realization} dtype=bf16 chart=raon "
            f"finish_head_fused={fused}",
            flush=True,
        )
        _ANNOUNCED = True
    return out
