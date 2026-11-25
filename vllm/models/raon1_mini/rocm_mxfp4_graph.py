# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in graph wrapper around the existing Quark/AITER FlyDSL A4W4 path.

Keep Quark's packing, native activation quantization and native shared-expert
scheduling. Never send these packed FP4 weights through the MXFP8 adapter.
"""

import os

import torch


def stage_is_flydsl(stage):
    while hasattr(stage, "func"):
        keywords = getattr(stage, "keywords", {}) or {}
        if any(
            str(keywords.get(key, "")).startswith("flydsl_")
            for key in ("kernelName", "kernelName1", "kernelName2")
        ):
            return True
        stage = stage.func
    return getattr(stage, "__module__", "").startswith("aiter.ops.flydsl.")


def verify_native_backend(module, x):
    routed = module.experts.routed_experts
    method = routed.quant_method
    backend = getattr(getattr(method, "mxfp4_backend", None), "value", None)
    if (
        type(method).__name__ != "QuarkOCP_MX_MoEMethod"
        or backend not in ("AITER_MXFP4_BF16", "AITER_MXFP4_MXFP4")
        or getattr(method, "ocp_mx_scheme", None) not in ("w_mxfp4", "w_mxfp4_a_mxfp4")
    ):
        raise RuntimeError(
            "FlyDSL MXFP4 graph requires the native Quark/AITER MXFP4 backend"
        )
    from vllm._aiter_ops import _resolve_situv2_activation, rocm_aiter_ops

    if _resolve_situv2_activation() != "a4w4":
        raise RuntimeError(
            "FlyDSL MXFP4 graph requires VLLM_ROCM_USE_AITER_MOE_SITUV2=a4w4"
        )
    checked = getattr(module, "_raon_mxfp4_flydsl_checked", None)
    if checked is None:
        checked = module._raon_mxfp4_flydsl_checked = set()
    key = (tuple(x.shape), x.dtype, x.device)
    if key in checked:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Verify native MXFP4 FlyDSL dispatch before graph capture")
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import get_2stage_cfgs

    cfg = routed.moe_config
    intermediate = routed.w13_weight.shape[1] // 2
    hidden_pad = x.shape[1] - cfg.hidden_dim_unpadded
    intermediate_pad = (
        cfg.intermediate_size_per_partition
        - cfg.intermediate_size_per_partition_unpadded
        if cfg.intermediate_pad is None
        else cfg.intermediate_pad
    )
    gate_mode = (
        "interleave"
        if rocm_aiter_ops.is_fused_moe_situv2_gate_up_interleaved()
        else "separated"
    )
    metadata = get_2stage_cfgs(
        x.shape[0],
        x.shape[1],
        intermediate,
        routed.w13_weight.shape[0],
        module.experts.router.top_k,
        x.dtype,
        dtypes.fp4x2,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Situv2,
        False,
        hidden_pad,
        intermediate_pad,
        is_shuffled=True,
        gate_mode=gate_mode,
        gate_up_interleave=(gate_mode == "interleave"),
        situ_beta=cfg.activation_situ_beta,
        situ_linear_beta=cfg.activation_situ_linear_beta,
    )
    if metadata.run_1stage or not all(
        stage_is_flydsl(stage) for stage in (metadata.stage1, metadata.stage2)
    ):
        raise RuntimeError(
            "AITER selected a non-FlyDSL MXFP4 stage; refusing graph candidate"
        )
    checked.add(key)
    print(
        f"RAON_MXFP4_FLYDSL SELECTED shape={tuple(x.shape)} "
        "precision=a4w4 stage1=flydsl stage2=flydsl packing=quark-native",
        flush=True,
    )


def try_forward(module, x):
    if os.environ.get("RAON_MOE_BACKEND", "legacy") != "flydsl":
        return None
    if not torch.version.hip or os.environ.get("RAON_MOE_OUTER_GRAPH", "0") != "1":
        return None
    if not getattr(module, "_raon_local_graph_world", False):
        raise RuntimeError("Raon native MXFP4 outer graphs require TP1/DP1 without EP")
    verify_native_backend(module, x)
    module._raon_graph_precision = "mxfp4"
    from .rocm_moe_outer_graph import try_forward as graph_forward

    return graph_forward(module, x, lambda owner, value: owner._native_forward(value))
