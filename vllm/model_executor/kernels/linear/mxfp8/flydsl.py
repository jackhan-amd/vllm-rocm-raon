# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in gfx950 MXFP8 linear: AITER HIP quantization and FlyDSL GEMM."""

import torch
from torch.nn import Parameter

from vllm.model_executor.kernels.linear.mxfp4.flydsl_a16 import enabled_on_gfx950
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
    Mxfp8LinearKernel,
)


class FlydslMxfp8LinearKernel(Mxfp8LinearKernel):
    @classmethod
    def is_supported(cls, compute_capability=None):
        return enabled_on_gfx950()

    @classmethod
    def can_implement(cls, config):
        n, k = config.weight_shape
        if config.bmm_batch_size is not None or n % 16 or k % 256:
            return False, "FlyDSL MXFP8 requires GEMM with N%16=0 and K%256=0"
        return True, None

    def process_weights_after_loading(self, layer):
        from aiter.ops.shuffle import shuffle_weight

        w = layer.weight.detach()
        s = layer.weight_scale.detach().view(torch.uint8)
        n, k = w.shape
        if w.dtype != torch.float8_e4m3fn or tuple(s.shape) != (n, k // 32):
            raise ValueError("FlyDSL MXFP8 requires E4M3 + row-wise E8M0 scales")
        if bool((s == 255).any()):
            raise ValueError("Nonfinite MXFP8 scale before FlyDSL packing")
        layer.weight = Parameter(
            shuffle_weight(w, layout=(16, 16)), requires_grad=False
        )
        layer.weight_scale = Parameter(s.contiguous(), requires_grad=False)
        self.announced = False

    def apply_weights(self, layer, x, bias=None):
        from aiter import dtypes
        from aiter.ops.flydsl.gemm_a8w8_mxfp8 import run_gemm_a8w8_mxfp8
        from aiter.ops.quant import per_1x32_mx_quant_hip

        if x.dtype != torch.bfloat16:
            raise ValueError(
                "FlyDSL MXFP8 expects BF16 input before dynamic quantization"
            )
        k = layer.weight.shape[-1]
        a = x.reshape(-1, k).contiguous()
        aq, scales = per_1x32_mx_quant_hip(
            a, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=False
        )
        y = torch.empty(
            (a.shape[0], layer.weight.shape[0]), device=x.device, dtype=x.dtype
        )
        run_gemm_a8w8_mxfp8(aq, layer.weight, scales, layer.weight_scale, y)
        y = y.reshape(*x.shape[:-1], layer.weight.shape[0])
        if bias is not None:
            y = y + bias
        if not self.announced:
            self.announced = True
            print(
                f"RAON_DENSE_FLYDSL LIVE precision=a8w8 n={layer.weight.shape[0]} "
                f"k={k} quant=aiter_hip gemm=aiter_flydsl",
                flush=True,
            )
        return y
