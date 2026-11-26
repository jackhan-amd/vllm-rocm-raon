# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in stateless native MXFP8 MoE inside the enclosing breakable graph."""

import os

import torch

_ANNOUNCED: set[bool] = set()


def forward(module, x):
    from . import model, mxfp8_flydsl

    if (
        not torch.version.hip
        or not mxfp8_flydsl.enabled()
        or not model._MOE_EAGER_BREAK
        or not model.moe_trainer_realized()
        or not model.moe_expert_mxfp8_realized()
        or model._MOE_ROUTE_CAPTURE
        or model.raon_phase.PREFILL_ALIGNED
        or model.moe_combine_ep_size()
        or model._MXFP8_MIN_M
        or os.environ.get("RAON_MOE_COMBINE_UNW_FUSED") != "flydsl"
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or x.shape[1] != 2560
        or getattr(module.experts.router, "top_k", None) != 6
        or not getattr(module, "_raon_local_graph_world", False)
    ):
        raise RuntimeError(
            "Direct MoE graph requires native FlyDSL MXFP8 TP1/DP1, "
            "ordered FlyDSL combine and breakable graphs, "
            "without phase/route diagnostics"
        )
    # No local graph, borrowed output, wrapper copy or eager-break callback.
    # The raw native chain already has GPU-only routing and bounded workspaces.
    # AITER packing/JIT must be warmed before capture; their own guards enforce it.
    output = model._raon_fp32_combine_moe_impl_eager(module, x)
    capture = torch.cuda.is_current_stream_capturing()
    if capture not in _ANNOUNCED:
        _ANNOUNCED.add(capture)
        print(
            f"RAON_ROCM_MOE_DIRECT_GRAPH EXECUTED capturing={int(capture)} "
            "backend=flydsl local_graph=0 eager_callback=0 shared_owner=existing",
            flush=True,
        )
    return output
