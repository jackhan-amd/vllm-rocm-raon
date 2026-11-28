# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the opt-in projection/mHC boundary and warmed graph latency.

Run on one gfx950 GPU with the patched runtime. This is an operator diagnostic,
not end-to-end serving TPS or model accuracy. The exact oracle is the same AITER
GEMM followed by the accepted mHC mix. A separate FP32 Torch matmul/mix checks
the mathematical result with an explicitly reported tolerance.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm.models.raon1_mini import mhc, rocm_mhc_gate
from vllm.models.raon1_mini.rocm_dense import configuration
from vllm.models.raon1_mini.rocm_projection_mhc import project_mix


def torch_mix(carrier, branch, coeff, mix_one):
    """Torch oracle preserving the accepted row-dependent FP32 schedule."""
    streams = carrier.float().unbind(1)
    result = []
    for i in range(4):
        h = coeff[:, 4 + i * 4 : 8 + i * 4].unbind(-1)
        value = h[0][:, None] * streams[0]
        for j in range(1, 4):
            if mix_one:
                value = value + h[j][:, None] * streams[j]
            else:
                value = torch.addcmul(value, h[j][:, None], streams[j])
        result.append((value + coeff[:, i, None] * branch.float()).bfloat16())
    return torch.stack(result, dim=1)


def graph_latency(graph):
    """Measure warm graph replay after separate compilation/capture/warmup."""
    for _ in range(16):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(100):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 10)
    return statistics.median(samples)


def case(rows, k):
    from aiter.ops.flydsl.kernels.gemm_a16w16_skinny_gfx950 import (
        gemm_a16w16_skinny,
    )
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    config = configuration(rows, 2560, k)
    x = torch.randn((rows, k), device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn((2560, k), device="cuda", dtype=torch.bfloat16) * 0.1
    carrier = torch.randn((rows, 4, 2560), device="cuda", dtype=torch.bfloat16)
    coeff = torch.randn((rows, 20), device="cuda", dtype=torch.float32) * 0.2
    if config is None:
        assert project_mix(x, weight, carrier, coeff) is None
        return {"rows": rows, "k": k, "status": "ORIGINAL_PATH_RETAINED"}
    skinny = {
        key: config[key]
        for key in ("block_m", "block_n", "block_k", "stages", "k_waves")
    }
    mix_one = mhc._FIXED_MIX_ONE if mhc._BATCH_INVARIANT else rows == 1
    mix = rocm_mhc_gate._create(False, mix_one, False, False)[1]

    def separate(matrix=weight):
        projected = gemm_a16w16_skinny(x, matrix, **skinny)
        out = torch.empty_like(carrier)
        _run_compiled(
            mix,
            carrier.view(-1),
            projected.view(-1),
            coeff.view(-1),
            out.view(-1),
            rows,
            torch.cuda.current_stream().cuda_stream,
        )
        return projected, out

    projected, expected = separate()
    actual = project_mix(x, weight, carrier, coeff)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    mix_gold = torch_mix(carrier, projected, coeff, mix_one)
    torch.testing.assert_close(actual, mix_gold, rtol=0, atol=0)
    # Independent math oracle: BF16 GEMM seam followed by the same Torch mix.
    math_gold = torch_mix(
        carrier, F.linear(x.float(), weight.float()).bfloat16(), coeff, mix_one
    )
    torch.testing.assert_close(actual, math_gold, atol=0.03125, rtol=0.015625)
    torch.cuda.synchronize()
    baseline_graph, fused_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(baseline_graph):
        _, replay_expected = separate()
    with torch.cuda.graph(fused_graph):
        replay_actual = project_mix(x, weight, carrier, coeff)
    checks = []
    for seed in (101, 202, 303):
        torch.manual_seed(seed)
        # Refresh all live input addresses without recapture.
        x.normal_(std=0.1)
        carrier.normal_()
        coeff.normal_(std=0.2)
        baseline_graph.replay()
        fused_graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(replay_actual, replay_expected, rtol=0, atol=0)
        checks.append({"seed": seed, "exact": True})
    baseline_us = graph_latency(baseline_graph)
    fused_us = graph_latency(fused_graph)
    rotation = None
    if rows in (1, 8, 32):
        # Distinct addresses exceed L2; a single repeated projection can give
        # a misleading hot-cache win that disappears across model layers.
        matrices = [weight.clone() for _ in range(24)]
        separate_chain, fused_chain = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(separate_chain):
            for matrix in matrices:
                _, last_expected = separate(matrix)
        with torch.cuda.graph(fused_chain):
            for matrix in matrices:
                last_actual = project_mix(x, matrix, carrier, coeff)
        separate_chain.replay()
        fused_chain.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(last_actual, last_expected, rtol=0, atol=0)
        cold_baseline_us = graph_latency(separate_chain)
        cold_fused_us = graph_latency(fused_chain)
        rotation = {
            "distinct_weight_addresses": 24,
            "weight_bytes": 24 * weight.numel() * weight.element_size(),
            "baseline_us_per_projection": cold_baseline_us / 24,
            "fused_us_per_projection": cold_fused_us / 24,
            "operator_speedup": cold_baseline_us / cold_fused_us,
            "not_full_layer_latency": True,
        }
    return {
        "rows": rows,
        "k": k,
        "status": "PASS",
        "oracle_config": skinny,
        "candidate_config": dict(skinny, block_n=16) if k == 5120 else skinny,
        "baseline_exact": True,
        "torch_mix_exact": True,
        "torch_math_atol": 0.03125,
        "torch_math_rtol": 0.015625,
        "torch_math_max_abs": (actual.float() - math_gold.float()).abs().max().item(),
        "graph_input_refresh": checks,
        "baseline_graph_us": baseline_us,
        "fused_graph_us": fused_us,
        "operator_speedup": baseline_us / fused_us,
        "layer_address_rotation": rotation,
        "scope": "operator-only; five timer batches, not public TPS repetitions",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=[*range(1, 33), 64])
    parser.add_argument("--k", nargs="+", type=int, default=[5120, 10240])
    args = parser.parse_args()
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.split(":")[0] == "gfx950"
    report = {
        "status": "RUNNING",
        "gpu": props.name,
        "cases": [],
        "batch_invariant": mhc._BATCH_INVARIANT,
        "profile": "actual-row-dependent mHC schedule",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for rows in args.rows:
            for k in args.k:
                torch.manual_seed(42)
                report["cases"].append(case(rows, k))
                args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["status"] = "PASS"
    except Exception as error:
        report.update(status="FAIL", error=repr(error), failed_rows=rows, failed_k=k)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
