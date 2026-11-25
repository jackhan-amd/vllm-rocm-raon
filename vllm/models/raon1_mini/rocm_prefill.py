# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Use the same ROCm fused-KDA support gate as Kimi-K3, without silent fallback."""

import os


def select_backend(backend, head_dim, dtype, lower_bound):
    from vllm.models.kimi_k3.amd.ops.kda_chunk import can_use_fused_kda_chunk
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

    selected = os.environ.get("RAON_KDA_PREFILL") or backend
    if selected not in ("auto", "fused", "triton"):
        raise ValueError(f"Unsupported ROCm KDA prefill selection: {selected}")
    supported = lower_bound is not None and can_use_fused_kda_chunk(
        head_dim, head_dim, dtype, FLA_CHUNK_SIZE
    )
    if selected == "fused" and not supported:
        raise RuntimeError(
            "Explicit fused KDA prefill requires the gfx950 native HIP kernels, "
            "BF16 inputs, head_dim=128, chunk_size=64 and a bounded gate"
        )
    actual = "rocm_fused" if supported and selected != "triton" else "rocm_triton"
    print(
        f"RAON_ROCM_KDA_PREFILL SELECTED requested={selected} actual={actual}",
        flush=True,
    )
    return actual
