# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in AITER/FlyDSL A16W16 projections without changing checkpoint precision."""

from functools import cache

import torch

_ANNOUNCED = set()


@cache
def _gemm():
    from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm

    return flydsl_hgemm


@cache
def configuration(rows, n, k):
    """Use skinny decode tiles; retain AITER dispatch for other shapes."""
    if not 1 <= rows <= 32 or (rows > 8 and (n, k) == (20656, 2560)):
        return None
    from aiter.ops.flydsl.kernels.gemm_a16w16_skinny_gfx950 import skinny_config_error

    block_m = 8 if rows <= 8 else 32
    block_k = 128 if k % 128 == 0 else 64
    k_waves = 4 if k % (4 * block_k) == 0 else 1
    tuned_n = {
        (5120, 128): 32 if rows <= 8 else 16,
        (10240, 512): 64 if rows == 1 or rows > 8 else 16,
        (2560, 1088): 8 if 1 < rows <= 8 else 16,
    }.get((n, k))
    for block_n in dict.fromkeys([tuned_n, 64, 32, 16, 8]):
        if block_n is None:
            continue
        if skinny_config_error(n, k, block_m, block_n, block_k, 2, k_waves) is None:
            return dict(
                policy="sk",
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                stages=2,
                m_waves=1,
                n_waves=1,
                k_waves=k_waves,
                split_k=1,
            )
    return None


@cache
def _validate_device(device):
    if torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] != "gfx950":
        raise RuntimeError("Native Raon A16W16 projections require gfx950")


def try_gemm(x, weight, bias=None):
    if x.dtype != torch.bfloat16 or weight.dtype != x.dtype or bias is not None:
        # FP32 routing/steering operations retain their original precision.
        return None
    if not x.is_cuda or weight.device != x.device:
        return None
    _validate_device(x.device)
    if weight.ndim != 2 or x.shape[-1] != weight.shape[-1]:
        raise ValueError("Invalid native Raon projection dimensions")
    k, n = weight.shape[-1], weight.shape[0]
    rows = x.numel() // k
    config = configuration(rows, n, k)
    if config is None:
        return None
    a = x.reshape(-1, k).contiguous()
    out = _gemm()(a, weight, **config)
    key = (rows, n, k, x.dtype)
    if key not in _ANNOUNCED:
        _ANNOUNCED.add(key)
        print(
            f"RAON_ROCM_DENSE_FLYDSL LAUNCHED M={rows} N={n} K={k} "
            f"dtype={x.dtype} policy=sk tile={config['block_m']}x{config['block_n']} "
            "backend=aiter_flydsl fallback=0",
            flush=True,
        )
    return out.reshape(*x.shape[:-1], n)
