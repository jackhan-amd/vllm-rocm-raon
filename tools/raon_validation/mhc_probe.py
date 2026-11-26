# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen native mHC projection/statistics with real checkpoint weights.

This measures isolated kernels, not model throughput or task accuracy. The HIP
candidate calls only GEMM/square-sum: Raon's gate/chart is not replaced by the
different Sinkhorn gate in AITER's complete mHC operator.
"""

import argparse
import json
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from tools.raon_validation.common import environment, error, write_json


def load_phi(model, layer):
    index = next(model.glob("*.safetensors.index.json"))
    mapping = json.loads(index.read_text())["weight_map"]
    prefix = f"model.layers.{layer}.self_attn_raon_mhc."
    tensors = []
    for suffix in ("phi_pre", "phi_post", "res.phi_z"):
        name = prefix + suffix
        with safe_open(str(model / mapping[name]), framework="pt", device="cpu") as f:
            tensors.append(f.get_tensor(name))
    original = [str(t.dtype) for t in tensors]
    return torch.cat(tensors).to(device="cuda", dtype=torch.bfloat16), original


def graph_time(fn):
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(8):
            captured = fn()
    graph.replay()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(40):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / 320, graph, captured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--candidate", choices=["baseline", "flydsl", "hip"], required=True
    )
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-n", type=int, default=8)
    parser.add_argument("--k-waves", type=int, default=4)
    parser.add_argument("--splits", type=int, default=16)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 32, 64, 128, 256])
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 12, 24])
    args = parser.parse_args()
    result = dict(
        status="RUNNING", scope=__doc__, arguments=vars(args).copy(), cases=[]
    )
    result["arguments"]["model"] = str(args.model)
    try:
        result["environment"] = environment()
        if not result["environment"]["architecture"].startswith("gfx950"):
            raise RuntimeError("This native mHC screen requires gfx950")
        from vllm.models.raon1_mini import mhc
        from vllm.models.raon1_mini.batch_invariant import matmul_persistent

        def baseline(x, phi):
            rows, width = x.shape
            mm = matmul_persistent(x, phi.t())
            square = torch.empty((rows, width), dtype=torch.float32, device=x.device)
            mhc._mhc_square[((x.numel() + 1023) // 1024,)](
                x, square, x.numel(), B=1024, enable_fp_fusion=False
            )
            return mm, square.mean(-1)

        with torch.inference_mode():
            for layer in args.layers:
                phi, source_dtypes = load_phi(args.model, layer)
                n, k = phi.shape
                if (n, k) != (17, 10240):
                    raise ValueError(f"Unexpected Raon mHC shape: {phi.shape}")
                padded = torch.zeros((24, k), dtype=phi.dtype, device=phi.device)
                padded[:n].copy_(phi)
                hip_weight = padded.float()
                if args.candidate == "hip":
                    from aiter.ops.mhc import mhc_pre_gemm_sqrsum, mhc_shuffle_fn

                    if args.packed:
                        hip_weight = mhc_shuffle_fn(hip_weight)
                elif args.candidate == "flydsl":
                    from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm
                    from aiter.ops.flydsl.kernels.gemm_a16w16_skinny_gfx950 import (
                        skinny_config_error,
                    )

                    reason = skinny_config_error(
                        24, k, args.block_m, args.block_n, 128, 2, args.k_waves
                    )
                    if reason:
                        raise ValueError(reason)
                torch.manual_seed(2701 + layer)
                inputs = torch.randn(
                    (max(args.rows), k), device="cuda", dtype=phi.dtype
                )
                for rows in args.rows:
                    x = inputs[:rows].clone()

                    def candidate(
                        value=x,
                        phi=phi,
                        padded=padded,
                        hip_weight=hip_weight,
                        n=n,
                        k=k,
                    ):
                        if args.candidate == "baseline":
                            return baseline(value, phi)
                        m = value.shape[0]
                        if args.candidate == "flydsl":
                            mm = torch.empty(
                                (m, 24), dtype=phi.dtype, device=value.device
                            )
                            for first in range(0, m, args.block_m):
                                last = min(m, first + args.block_m)
                                flydsl_hgemm(
                                    value[first:last],
                                    padded,
                                    out=mm[first:last],
                                    policy="sk",
                                    block_m=args.block_m,
                                    block_n=args.block_n,
                                    block_k=128,
                                    stages=2,
                                    split_k=1,
                                    m_waves=1,
                                    n_waves=1,
                                    k_waves=args.k_waves,
                                )
                            square = torch.empty(
                                (m, k), dtype=torch.float32, device=value.device
                            )
                            mhc._mhc_square[((value.numel() + 1023) // 1024,)](
                                value,
                                square,
                                value.numel(),
                                B=1024,
                                enable_fp_fusion=False,
                            )
                            return mm[:, :n], square.mean(-1)
                        partial = torch.empty(
                            (args.splits, m, 32),
                            dtype=torch.float32,
                            device=value.device,
                        )
                        squares = torch.empty(
                            (args.splits, m), dtype=torch.float32, device=value.device
                        )
                        mhc_pre_gemm_sqrsum(
                            partial[:, :, :24],
                            squares,
                            value,
                            hip_weight,
                            128,
                            w_preshuffle_bf16=int(args.packed),
                        )
                        # Include partial reduction and the original BF16 GEMM
                        # output boundary in timing; do not time just the producer.
                        return partial.sum(0)[:, :n].to(phi.dtype), squares.sum(0) / k

                    start = time.monotonic()
                    actual = candidate()
                    torch.accelerator.synchronize()
                    cold = time.monotonic() - start
                    gold = F.linear(x.float(), phi.float())
                    mean_gold = x.float().square().mean(-1)
                    reference = baseline(x, phi)
                    one = candidate(x[:1])
                    repeated = candidate()
                    case = dict(
                        layer=layer,
                        rows=rows,
                        n=n,
                        k=k,
                        checkpoint_dtypes=source_dtypes,
                        input_dtype=str(x.dtype),
                        cold_seconds_excluded=cold,
                        mm_vs_fp32=error(actual[0], gold),
                        mean_vs_fp32=error(
                            actual[1], mean_gold, rms_limit=1e-5, peak_limit=1e-5
                        ),
                        mm_vs_persistent=error(actual[0], reference[0]),
                        phi_batch_invariant=torch.equal(actual[0][:1], one[0]),
                        exact_repeat=all(
                            torch.equal(a, b) for a, b in zip(actual, repeated)
                        ),
                    )
                    us, graph, captured = graph_time(candidate)
                    case["graph_microseconds_including_reductions"] = us
                    x.add_(0.125)
                    graph.replay()
                    torch.accelerator.synchronize()
                    case["changed_input_replay_mm"] = error(
                        captured[0], F.linear(x.float(), phi.float())
                    )
                    case["changed_input_replay_mean"] = error(
                        captured[1],
                        x.float().square().mean(-1),
                        rms_limit=1e-5,
                        peak_limit=1e-5,
                    )
                    result["cases"].append(case)
                    write_json(args.output, result)
                    print(
                        json.dumps(
                            {
                                k: case[k]
                                for k in [
                                    "layer",
                                    "rows",
                                    "graph_microseconds_including_reductions",
                                    "phi_batch_invariant",
                                    "exact_repeat",
                                ]
                            }
                        ),
                        flush=True,
                    )
        checks = [
            v["pass"]
            for c in result["cases"]
            for v in c.values()
            if isinstance(v, dict) and "pass" in v
        ]
        result["status"] = (
            "PASS"
            if checks
            and all(checks)
            and all(
                c["phi_batch_invariant"] and c["exact_repeat"] for c in result["cases"]
            )
            else "FAIL"
        )
    except Exception:
        result.update(status="ERROR", traceback=traceback.format_exc())
        raise
    finally:
        write_json(args.output, result)
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
