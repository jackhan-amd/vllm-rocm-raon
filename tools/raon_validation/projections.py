# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native AITER/FlyDSL projection screen against FP32 Torch gold on Raon shapes."""

import argparse
import hashlib
import inspect
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import environment, error, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 8, 32, 256])
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument(
        "--shape", nargs=2, type=int, action="append", metavar=("N", "K")
    )
    args = parser.parse_args()
    import tvm_ffi._optional_torch_c_dlpack as dlpack

    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl
    from vllm.models.raon1_mini import rocm_dense

    result = {
        "status": "RUNNING",
        "scope": __doc__,
        "environment": environment(),
        "allocator_library_loaded": dlpack._LIB is not None,
        "cases": [],
    }
    assert dlpack._LIB is not None, "TVM FFI must load its prebuilt ROCm addon"
    result["native_source_sha256"] = hashlib.sha256(
        Path(inspect.getsourcefile(rocm_dense._gemm())).read_bytes()
    ).hexdigest()
    torch.manual_seed(423)

    def timed(fn):
        for _ in range(2):
            fn()
        torch.accelerator.synchronize()
        begin, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        begin.record()
        for _ in range(args.repetitions):
            fn()
        end.record()
        end.synchronize()
        return begin.elapsed_time(end) * 1000 / args.repetitions

    try:
        with torch.inference_mode():
            for n, k in args.shape or (
                (20656, 2560),
                (5120, 128),
                (10240, 512),
                (2560, 1088),
                (40, 2560),
            ):
                weight = (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
                for rows in args.rows:
                    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
                    gold = F.linear(x.float(), weight.float())
                    os.environ["RAON_ROCM_DENSE_LINEAR_BACKEND"] = "legacy"
                    baseline = rocm_unquantized_gemm_impl(x, weight)
                    baseline_us = timed(
                        lambda x=x, weight=weight: rocm_unquantized_gemm_impl(x, weight)
                    )
                    baseline_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(baseline_graph):
                        baseline_captured = rocm_unquantized_gemm_impl(x, weight)
                    baseline_graph_us = timed(baseline_graph.replay)
                    os.environ["RAON_ROCM_DENSE_LINEAR_BACKEND"] = "flydsl"
                    start = time.monotonic()
                    native = rocm_unquantized_gemm_impl(x, weight)
                    compile_seconds = time.monotonic() - start
                    native_us = timed(
                        lambda x=x, weight=weight: rocm_unquantized_gemm_impl(x, weight)
                    )
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = rocm_unquantized_gemm_impl(x, weight)
                    native_graph_us = timed(graph.replay)
                    x.add_(0.125)
                    baseline_graph.replay()
                    graph.replay()
                    replay_gold = F.linear(x.float(), weight.float())
                    case = dict(
                        rows=rows,
                        n=n,
                        k=k,
                        native_kernel_selected=(
                            rocm_dense.configuration(rows, n, k) is not None
                        ),
                        compile_seconds_excluded=compile_seconds,
                        baseline_vs_gold=error(baseline, gold),
                        native_vs_gold=error(native, gold),
                        graph_vs_gold=error(captured, replay_gold),
                        baseline_graph_vs_gold=error(baseline_captured, replay_gold),
                        baseline_api_us=baseline_us,
                        native_api_us=native_us,
                        baseline_graph_us=baseline_graph_us,
                        native_graph_us=native_graph_us,
                        speedup=baseline_us / native_us,
                    )
                    result["cases"].append(case)
                    write_json(args.output, result)
            weight = torch.randn(40, 128, device="cuda", dtype=torch.bfloat16)
            x = torch.randn(2, 4, 256, device="cuda", dtype=torch.bfloat16)[..., ::2]
            bias = torch.randn(40, device="cuda", dtype=torch.bfloat16)
            result["strided_3d_native"] = error(
                rocm_unquantized_gemm_impl(x, weight),
                F.linear(x.float(), weight.float()),
            )
            actual = rocm_unquantized_gemm_impl(x, weight, bias)
            result["strided_3d_with_bias"] = error(
                actual, F.linear(x.float(), weight.float(), bias.float())
            )
            result["fp32_preserved"] = (
                rocm_dense.try_gemm(x.float(), weight.float()) is None
            )
        result["status"] = (
            "PASS"
            if result["fp32_preserved"]
            and result["strided_3d_native"]["pass"]
            and result["strided_3d_with_bias"]["pass"]
            and all(
                c[key]["pass"]
                for c in result["cases"]
                for key in (
                    "baseline_vs_gold",
                    "native_vs_gold",
                    "graph_vs_gold",
                    "baseline_graph_vs_gold",
                )
            )
            else "FAIL"
        )
    except BaseException:
        import traceback

        result.update(status="ERROR", traceback=traceback.format_exc())
    write_json(args.output, result)
    print(result["status"], len(result["cases"]), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
