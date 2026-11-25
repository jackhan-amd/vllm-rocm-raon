# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in gfx950 A16W4 linear using AITER's native FlyDSL implementation."""

import os

import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.linear.mxfp4.base import (
    MxFp4LinearKernel,
    MxFp4LinearLayerConfig,
)
from vllm.platforms import current_platform


def enabled_on_gfx950():
    if (
        not current_platform.is_rocm()
        or os.environ.get("RAON_ROCM_DENSE_LINEAR_BACKEND", "legacy") != "flydsl"
    ):
        return False, "native FlyDSL dense candidate is not selected"
    arch = torch.cuda.get_device_properties(
        torch.accelerator.current_device_index()
    ).gcnArchName
    if arch.split(":")[0] != "gfx950":
        raise RuntimeError("FlyDSL Raon dense candidate requires gfx950")
    return True, None


class FlydslMxfp4A16LinearKernel(MxFp4LinearKernel):
    @classmethod
    def is_supported(cls, compute_capability=None):
        return enabled_on_gfx950()

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig):
        return (
            (True, None)
            if config.activation_quant_key is None
            else (False, "A16W4 requires unquantized activations")
        )

    def process_weights_after_loading(self, layer):
        from aiter.ops.flydsl.gemm_a16wfp4 import prepare_gemm_a16wfp4_weight

        w = layer.weight.detach().view(torch.uint8)
        s = layer.weight_scale.detach().view(torch.uint8)
        self.n, self.k = w.shape[0], w.shape[1] * 2
        if self.k % 32 or tuple(s.shape) != (self.n, self.k // 32):
            raise ValueError("FlyDSL A16W4 requires row-wise E8M0 block-32 scales")
        if bool((s == 255).any()):
            raise ValueError("Nonfinite E8M0 weight scale before FlyDSL packing")
        self.padded_n = (self.n + 255) // 256 * 256
        self.padded_k = (self.k + 255) // 256 * 256
        w = F.pad(w, (0, (self.padded_k - self.k) // 2, 0, self.padded_n - self.n))
        s = F.pad(
            s,
            (0, (self.padded_k - self.k) // 32, 0, self.padded_n - self.n),
            value=127,
        )
        self.prepared = prepare_gemm_a16wfp4_weight(w, s)
        self.announced = False

    def apply_weights(self, layer, x, bias=None):
        from aiter.ops.flydsl.gemm_a16wfp4 import flydsl_gemm_a16wfp4

        if x.dtype != torch.bfloat16 or x.shape[-1] != self.k:
            raise ValueError("FlyDSL A16W4 expects BF16 input with the loaded K")
        original_shape = x.shape[:-1]
        a = x.reshape(-1, self.k).contiguous()
        if self.padded_k != self.k:
            a = F.pad(a, (0, self.padded_k - self.k))
        # Unpadding is a strided view when N was rounded up. SiTU and other
        # downstream native consumers address rows as row * logical_N, so
        # preserve the ordinary dense Linear output contract here.
        y = flydsl_gemm_a16wfp4(a, self.prepared)[:, : self.n].contiguous()
        y = y.reshape(*original_shape, self.n)
        if bias is not None:
            y = y + bias
        if not self.announced:
            self.announced = True
            print(
                f"RAON_DENSE_FLYDSL LIVE precision=a16w4 n={self.n} k={self.k} "
                f"padded_n={self.padded_n} padded_k={self.padded_k} "
                "backend=aiter_flydsl activation_quant=none",
                flush=True,
            )
        return y
