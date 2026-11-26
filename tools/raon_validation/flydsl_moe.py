# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 FlyDSL rows vs independent Torch gold and repeated HIP graph replay.

Synthetic quantized expert weights, no model download or performance benchmark.
Both legacy FP32-to-MXFP8 and native BF16-to-MXFP8 boundaries are reported.
"""

import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .common import error


def quant_dequant(x):
    groups = x.float().reshape(*x.shape[:-1], -1, 32)
    amax = groups.abs().amax(-1, keepdim=True)
    scale = torch.exp2(torch.ceil(torch.log2((amax / 448).clamp_min(2.0**-126))))
    payload = (groups / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (payload.float() * scale).reshape(x.shape)


def gold(x, probs, ids, q13, s13, q2, s2, native):
    a1 = quant_dequant(x)
    output = torch.empty(
        (*ids.shape, x.shape[-1]), dtype=torch.bfloat16, device=x.device
    )
    # Bound temporary memory by one expert; no [T,K,N,H] weight gather.
    for expert in range(q13.shape[0]):
        token, slot = (ids == expert).nonzero(as_tuple=True)
        if token.numel() == 0:
            continue
        w13 = q13[expert].float() * torch.exp2(
            s13[expert].float() - 127
        ).repeat_interleave(32, -1)
        w2 = q2[expert].float() * torch.exp2(
            s2[expert].float() - 127
        ).repeat_interleave(32, -1)
        gate, up = F.linear(a1[token], w13).chunk(2, -1)
        activation = (
            100
            * torch.tanh(gate * 0.25)
            * (((1 + torch.tanh(gate * 0.5)) * 0.5) * torch.tanh(up * 0.04))
        )
        prob = probs[token, slot]
        if not native:
            prob = prob.bfloat16().float()
        activation = activation * prob[:, None]
        if native:
            activation = activation.bfloat16().float()
        output[token, slot] = F.linear(quant_dequant(activation), w2).bfloat16()
    return output


def run(args, result):
    from vllm.models.raon1_mini import mxfp8_flydsl as candidate
    from vllm.models.raon1_mini.mxfp8_w8a8_triton import (
        _WQ_CACHE,
        expert_chain_mxw8a8,
    )

    free, total = torch.accelerator.get_memory_info()
    if free / total < 0.95:
        raise RuntimeError(
            "Select an idle GPU; this diagnostic must not share an evaluation GPU"
        )
    if args.moe_full_shape:
        experts, hidden, intermediate, topk = 128, 2560, 1088, 6
    else:
        experts, hidden, intermediate, topk = 8, 256, 256, 6
    torch.manual_seed(args.seed)
    q13 = torch.randn(
        (experts, 2 * intermediate, hidden), device="cuda", dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    q2 = torch.randn(
        (experts, hidden, intermediate), device="cuda", dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    s13 = torch.full(
        (experts, 2 * intermediate, hidden // 32), 122, device="cuda", dtype=torch.uint8
    )
    s2 = torch.full(
        (experts, hidden, intermediate // 32), 122, device="cuda", dtype=torch.uint8
    )
    if args.moe_varied_scales:
        s13.random_(120, 125)
        s2.random_(120, 125)
    w13, w2 = SimpleNamespace(), SimpleNamespace()
    w13.shape, w2.shape = q13.shape, q2.shape
    _WQ_CACHE[id(w13)] = (q13, s13, q2, s2, w13, w2)
    result["scope"] = __doc__
    fused_a2 = os.environ.get("RAON_MXFP8_FLYDSL_FUSED_A2", "0") == "1"
    result["candidate_intermediate"] = "fp32_to_mxfp8" if fused_a2 else "bf16_to_mxfp8"
    result["fixture"] = dict(
        experts=experts, hidden=hidden, intermediate=intermediate, topk=topk
    )
    result["tile_config"] = candidate.tile_config()
    result["varied_weight_scales"] = args.moe_varied_scales
    result["timing_scope"] = (
        "Native expert_rows only, including sorting/quantization and both GEMMs; "
        "synthetic quantized full shapes, warmed graph GPU time, not serving TPS"
    )
    result["tolerances"] = {
        "relative_rms": 0.01,
        "relative_peak": 0.02,
        "absolute": 1e-5,
    }
    try:
        if not args.moe_token_counts or min(args.moe_token_counts) < 1:
            raise ValueError("Use positive MoE token counts")
        for tokens in args.moe_token_counts:
            x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
            probs = torch.rand(tokens, topk, device="cuda")
            probs /= probs.sum(-1, keepdim=True)
            ids = (
                torch.rand(tokens, experts, device="cuda").topk(topk, -1).indices.int()
            )

            def fn(x=x, probs=probs, ids=ids):
                return candidate.expert_rows(x, probs, ids, w13, w2, 4, 25)

            actual = fn()
            legacy = expert_chain_mxw8a8(x, probs, ids, w13, w2, 4, 25)
            native_gold = gold(x, probs, ids, q13, s13, q2, s2, not fused_a2)
            legacy_gold = gold(x, probs, ids, q13, s13, q2, s2, False)
            case = dict(
                tokens=tokens,
                flydsl_vs_native_gold=error(actual, native_gold),
                triton_vs_legacy_gold=error(legacy, legacy_gold),
                flydsl_vs_legacy_gold=error(actual, legacy_gold),
                flydsl_vs_triton=error(actual, legacy),
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = fn()
            # Changed routes, inputs and probabilities must be read on replay.
            # This catches accidental specialization to first-call routing.
            for repeat in range(3):
                x.add_(0.125)
                ids.copy_((ids + 1) % experts)
                probs.copy_(probs.roll(1, -1))
                expected = fn().clone()
                graph.replay()
                check = error(captured, expected, rms_limit=0, peak_limit=0, atol=0)
                case[f"graph_replay_{repeat}"] = check
            changed_gold = gold(x, probs, ids, q13, s13, q2, s2, not fused_a2)
            case["changed_routes_vs_native_gold"] = error(captured, changed_gold)
            replays = args.moe_timing_replays
            if replays < 0:
                raise ValueError("Use a nonnegative replay count")
            if replays and all(
                v["pass"] for v in case.values() if isinstance(v, dict) and "pass" in v
            ):
                for _ in range(10):
                    graph.replay()
                torch.accelerator.synchronize()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(replays):
                    graph.replay()
                end.record()
                end.synchronize()
                case["timing_replays"] = replays
                case["graph_us"] = start.elapsed_time(end) * 1000 / replays
            if args.moe_kernel_witness:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profiler:
                    fn()
                    torch.accelerator.synchronize()
                kernels = sorted(
                    {
                        event.name
                        for event in profiler.events()
                        if event.device_type.name == "CUDA"
                    }
                )
                block_m = candidate.tile_config().get("block_m", 32)
                witnesses = {
                    stage: any(
                        f"mfma_moe{stage}_" in name and f"_t{block_m}x" in name
                        for name in kernels
                    )
                    for stage in (1, 2)
                }
                if candidate.fused_route_prepare_enabled(tokens):
                    witnesses["route_prepare"] = any(
                        "raon_mxfp8_route_prepare" in name for name in kernels
                    )
                case["kernel_witness"] = dict(
                    expected_m=block_m, stages=witnesses, kernels=kernels
                )
                if not all(witnesses.values()):
                    raise RuntimeError("Missing native MoE M-tile kernel witness")
            result["cases"].append(case)
    finally:
        _WQ_CACHE.pop(id(w13), None)
