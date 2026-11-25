# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native FlyDSL dense linears versus independent FP32 Torch GEMM gold.

Includes real Raon dimensions, padding, strided inputs and changing-input HIP
graph replay. Quantization error against BF16 is not a kernel error. No TPS
measurement is performed by this diagnostic.
"""

import argparse

import torch
import torch.nn.functional as F

from .common import environment, error, write_json
from .quantized_math import even_exponents


def fp4_reference(w):
    blocks = w.float().reshape(w.shape[0], -1, 32)
    exponent = even_exponents(blocks.abs().amax(-1))
    scale = torch.exp2(exponent.float())
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=w.device)
    mid = (table[1:] + table[:-1]) * 0.5
    value = blocks / scale[..., None]
    idx = torch.bucketize(value.abs().contiguous(), mid)
    tie = (idx < 7) & (value.abs() == mid[idx.clamp_max(6)])
    idx += (tie & ((idx % 2) == 1)).int()
    signed = (idx + (value < 0).int() * 8).byte().reshape(w.shape)
    packed = signed[:, ::2] | (signed[:, 1::2] << 4)
    gold = (table[idx] * value.sign() * scale[..., None]).reshape(w.shape)
    return packed, (exponent + 127).byte(), gold


def fp8_reference(w):
    blocks = w.float().reshape(w.shape[0], -1, 32)
    fraction, exponent = torch.frexp(blocks.abs().amax(-1))
    exponent = (exponent - 9 + (fraction >= 0.96875).int()).clamp(-127, 127)
    scale = torch.exp2(exponent.float())
    payload = (blocks / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (
        payload.reshape(w.shape),
        (exponent + 127).byte(),
        (payload.float() * scale[..., None]).reshape(w.shape),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from vllm.model_executor.kernels.linear.mxfp4.base import MxFp4LinearLayerConfig
    from vllm.model_executor.kernels.linear.mxfp4.flydsl_a16 import (
        FlydslMxfp4A16LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.flydsl import FlydslMxfp8LinearKernel
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )

    result = {
        "status": "RUNNING",
        "scope": __doc__,
        "environment": environment(),
        "cases": [],
    }
    torch.manual_seed(42)
    shapes = [
        (40, 2560),
        (5120, 128),
        (20656, 2560),
        (24064, 2560),
        (2560, 12032),
        (2176, 2560),
        (2560, 1088),
        (1152, 2560),
        (7680, 576),
        (259840, 2560),
    ]
    with torch.inference_mode():
        for n, k in shapes:
            precision = "mxfp8" if n == 259840 else "mxfp4_a16"
            w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.02
            packed, scales, gold_weight = (
                fp8_reference(w) if precision == "mxfp8" else fp4_reference(w)
            )
            layer = torch.nn.Module()
            layer.register_parameter(
                "weight", torch.nn.Parameter(packed, requires_grad=False)
            )
            layer.register_parameter(
                "weight_scale", torch.nn.Parameter(scales, requires_grad=False)
            )
            kernel = (
                FlydslMxfp8LinearKernel(Mxfp8LinearLayerConfig((n, k)))
                if precision == "mxfp8"
                else FlydslMxfp4A16LinearKernel(MxFp4LinearLayerConfig())
            )
            kernel.process_weights_after_loading(layer)
            for m in (1, 7, 128):
                x = torch.randn((m, 2 * k), device="cuda", dtype=torch.bfloat16)[:, ::2]

                def gold(a, precision=precision, m=m, gold_weight=gold_weight):
                    if precision == "mxfp8":
                        # Existing native vLLM MXFP8 activation boundary: fit
                        # amax within finite E4M3, independent Torch operations.
                        blocks = a.float().reshape(m, -1, 32)
                        s = torch.exp2(
                            torch.ceil(
                                torch.log2(
                                    (blocks.abs().amax(-1) / 448).clamp_min(2.0**-126)
                                )
                            )
                        )
                        a = (
                            (blocks / s[..., None]).to(torch.float8_e4m3fn).float()
                            * s[..., None]
                        ).reshape(a.shape)
                    return F.linear(a.float(), gold_weight).bfloat16()

                direct = kernel.apply_weights(layer, x)
                assert direct.is_contiguous(), (
                    "Dense consumers require packed output rows"
                )
                case = {
                    "precision": precision,
                    "m": m,
                    "n": n,
                    "k": k,
                    "direct": error(direct, gold(x), 0.01, 0.02),
                }
                if n == 2176:
                    gate, up = direct.float().chunk(2, -1)
                    expected_act = (
                        4
                        * torch.tanh(gate / 4)
                        * torch.sigmoid(gate)
                        * 25
                        * torch.tanh(up / 25)
                    ).bfloat16()
                    activated = torch.empty((m, n // 2), device=x.device, dtype=x.dtype)
                    torch.ops._C.situ_and_mul(activated, direct, 4.0, 25.0)
                    case["situ_composition"] = error(
                        activated, expected_act, 0.01, 0.02
                    )
                    if m > 1:
                        bad_view = F.pad(direct, (0, 128), value=17.0)[:, :n]
                        torch.ops._C.situ_and_mul(activated, bad_view, 4.0, 25.0)
                        case["negative_control_strided_situ"] = error(
                            activated, expected_act, 0.01, 0.02
                        )
                sx = x.contiguous()
                kernel.apply_weights(layer, sx)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = kernel.apply_weights(layer, sx)
                assert captured.is_contiguous()
                case["changing_input_graph"] = []
                for _ in range(3):
                    changed = torch.randn_like(sx)
                    sx.copy_(changed)
                    graph.replay()
                    case["changing_input_graph"].append(
                        error(captured, gold(changed), 0.01, 0.02)
                    )
                result["cases"].append(case)
                write_json(args.output, result)
                print(precision, m, n, k, case["direct"], flush=True)
            del w, packed, scales, gold_weight, layer, kernel
    passed = all(
        c["direct"]["pass"]
        and all(v["pass"] for v in c["changing_input_graph"])
        and c.get("situ_composition", {"pass": True})["pass"]
        and not c.get("negative_control_strided_situ", {"pass": False})["pass"]
        for c in result["cases"]
    )
    result["status"] = "PASS" if passed else "FAIL_TORCH_GOLD"
    write_json(args.output, result)
    assert passed, "Native dense kernel differs from independent Torch gold"


if __name__ == "__main__":
    main()
