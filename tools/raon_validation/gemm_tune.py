# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen the existing small-M Raon GEMM setting against native and Torch gold.

No model weights, precision, reduction size or kernel source are changed.
Timings are GPU event intervals around repeated API calls, including launch gaps.
They are not end-to-end serving measurements.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 16, 64, 65, 128])
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    if min(args.rows) < 1 or args.repetitions < 1:
        parser.error("Rows and repetitions must be positive")

    import torch
    from triton import knobs

    from vllm.models.raon1_mini import batch_invariant as bi

    from .common import environment

    result = {
        "status": "RUNNING",
        "scope": __doc__,
        "environment": environment(),
        "flag": "RAON_KDA_PROJ_GEMM_TUNE",
        "baseline": "0",
        "candidate": "1 (existing BLOCK_M=16, warps=4 for BF16 M<=64)",
        "kernel_sha256": hashlib.sha256(Path(bi.__file__).read_bytes()).hexdigest(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": args.rows,
        "cases": [],
    }
    free, total = torch.cuda.mem_get_info()
    if free / total <= 0.99:
        raise RuntimeError("Selected tuning GPU must be idle before the screen")
    if not result["environment"]["architecture"].startswith("gfx950"):
        raise RuntimeError("This screen was requested for gfx950")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def no_compile(*_args, **_kwargs):
        raise RuntimeError("Triton cache miss blocked during GEMM timing")

    def errors(actual, gold):
        delta = actual.double() - gold
        return {
            "max_abs": delta.abs().max().item(),
            "normalized_rmse": (
                delta.square().mean().sqrt()
                / gold.square().mean().sqrt().clamp_min(1e-30)
            ).item(),
        }

    def timed(a, b, dtype):
        bi.matmul_persistent(a, b, out_dtype=dtype)
        bi.matmul_persistent(a, b, out_dtype=dtype)
        torch.accelerator.synchronize()
        start, end = torch.Event(enable_timing=True), torch.Event(enable_timing=True)
        saved = knobs.runtime.jit_cache_hook
        knobs.runtime.jit_cache_hook = no_compile
        try:
            start.record()
            for _ in range(args.repetitions):
                bi.matmul_persistent(a, b, out_dtype=dtype)
            end.record()
            torch.accelerator.synchronize()
        finally:
            knobs.runtime.jit_cache_hook = saved
        return start.elapsed_time(end) * 1000 / args.repetitions

    flag = "RAON_KDA_PROJ_GEMM_TUNE"
    previous = os.environ.get(flag)
    torch.manual_seed(20261123)
    shapes = [
        (2560, 20656),
        (2560, 40),
        (5120, 2560),
        (128, 5120),
        (2560, 24064),
        (12032, 2560),
        (2560, 128),
    ]
    try:
        with torch.inference_mode():
            for k, n in shapes:
                # Same noncontiguous B layout as a contiguous linear weight transposed.
                weight = (torch.randn((n, k), device="cuda") * 0.02).bfloat16()
                b = weight.t()
                population = torch.randn((max(args.rows), k), device="cuda").bfloat16()
                gold = population.double() @ b.double()
                for dtype in [torch.bfloat16, torch.float32]:
                    references = {}
                    for value in ["0", "1"]:
                        os.environ[flag] = value
                        references[value] = bi.matmul_persistent(
                            population, b, out_dtype=dtype
                        )
                    for m in args.rows:
                        a = population[:m]
                        outputs, times = {}, {}
                        for value in ["0", "1"]:
                            os.environ[flag] = value
                            outputs[value] = bi.matmul_persistent(a, b, out_dtype=dtype)
                            times[value] = timed(a, b, dtype)
                        equal = torch.equal(outputs["0"], outputs["1"])
                        invariance = {
                            value: torch.equal(outputs[value], references[value][:m])
                            for value in ["0", "1"]
                        }
                        finite = all(
                            torch.isfinite(t).all().item() for t in outputs.values()
                        )
                        result["cases"].append(
                            {
                                "m": m,
                                "n": n,
                                "k": k,
                                "out_dtype": str(dtype),
                                "candidate_bitwise_equal": equal,
                                "batch_invariant": invariance,
                                "finite": finite,
                                "torch_fp64_gold": {
                                    value: errors(t, gold[:m])
                                    for value, t in outputs.items()
                                },
                                "gpu_event_api_us": times,
                                "pass": equal and finite and all(invariance.values()),
                            }
                        )
                        save()
            result["status"] = (
                "PASS" if all(c["pass"] for c in result["cases"]) else "FAIL"
            )
    except BaseException:
        import traceback

        result.update(status="ERROR", traceback=traceback.format_exc())
        raise
    finally:
        if previous is None:
            os.environ.pop(flag, None)
        else:
            os.environ[flag] = previous
        save()
    print(result["status"], len(result["cases"]), str(args.output), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
