# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in native mHC GEMM/statistics; preserves Raon's gate and BF16 boundary."""

import os
from functools import cache

import torch

_BACKEND = os.environ.get("RAON_ROCM_MHC_BACKEND", "legacy")
_ANNOUNCED = set()


def enabled():
    if _BACKEND not in ("legacy", "hip", "flydsl"):
        raise ValueError(f"Unknown Raon mHC backend: {_BACKEND}")
    return _BACKEND != "legacy"


@cache
def _device(device):
    prop = torch.cuda.get_device_properties(device)
    if not getattr(prop, "gcnArchName", "").startswith("gfx950"):
        raise RuntimeError("Native Raon mHC requires ROCm gfx950")


def prepare(site):
    if not enabled():
        return
    phi = site.phi
    _device(phi.device)
    if phi.shape != (17, 10240) or phi.dtype != torch.bfloat16:
        raise ValueError("Native mHC requires the BF16 Raon 17x10240 phi")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare native mHC weights before graph capture")
    padded = torch.zeros((24, 10240), device=phi.device, dtype=phi.dtype)
    padded[:17].copy_(phi.detach())
    # Refit uses the existing buffers so captured pointers remain valid.
    site._refit_set_buf("_rocm_mhc_phi", padded)
    if _BACKEND == "hip":
        from aiter.ops.mhc import mhc_shuffle_fn

        # Widen BF16 values losslessly; the native hi/lo pack has zero low parts.
        site._refit_set_buf("_rocm_mhc_packed", mhc_shuffle_fn(padded.float()))


def project(site, x):
    """Return compact BF16 scores and optional FP32 mean-square statistics."""
    if not enabled():
        raise RuntimeError("Native mHC project called with legacy backend")
    if (
        x.ndim != 2
        or x.shape[0] < 1
        or x.shape[1] != 10240
        or x.dtype != torch.bfloat16
    ):
        raise ValueError("Invalid native mHC activation shape or dtype")
    if not x.is_contiguous() or x.device != site.phi.device:
        raise ValueError("Native mHC needs contiguous activations on phi's device")
    if not hasattr(site, "_rocm_mhc_phi"):
        raise RuntimeError("Finalize native mHC weights before inference")
    rows = x.shape[0]
    if _BACKEND == "hip":
        from aiter.ops.mhc import mhc_pre_gemm_sqrsum

        # Fixed reduction schedule across batches, verified at M=1..256.
        # This calls only GEMM/statistics, never AITER's different Sinkhorn gate.
        partial = torch.empty((16, rows, 32), device=x.device, dtype=torch.float32)
        squares = torch.empty((16, rows), device=x.device, dtype=torch.float32)
        mhc_pre_gemm_sqrsum(
            partial[:, :, :24],
            squares,
            x,
            site._rocm_mhc_packed,
            128,
            w_preshuffle_bf16=1,
        )
        finisher = os.environ.get("RAON_ROCM_MHC_FINISHER", "torch")
        if finisher == "flydsl":
            from .rocm_mhc_finish import reduce

            scores, mean = reduce(partial, squares)
        elif finisher == "torch":
            scores = partial.sum(0)[:, :17].to(x.dtype)
            mean = squares.sum(0) / 10240
        else:
            raise ValueError(f"Unknown native mHC finisher: {finisher}")
    else:
        from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm

        # One fixed tile preserves per-row arithmetic across decode batches.
        scores = torch.empty((rows, 24), device=x.device, dtype=x.dtype)
        for start in range(0, rows, 32):
            end = min(rows, start + 32)
            flydsl_hgemm(
                x[start:end],
                site._rocm_mhc_phi,
                out=scores[start:end],
                policy="sk",
                block_m=32,
                block_n=8,
                block_k=128,
                stages=2,
                split_k=1,
                m_waves=1,
                n_waves=1,
                k_waves=4,
            )
        scores = scores[:, :17].contiguous()
        mean = None
    if _BACKEND not in _ANNOUNCED:
        _ANNOUNCED.add(_BACKEND)
        print(
            f"RAON_ROCM_MHC LAUNCHED backend={_BACKEND} "
            "dtype=bf16 gate=raon gate_unchanged=1 "
            f"native_statistics={mean is not None}",
            flush=True,
        )
    return scores, mean
