# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in pure AR/SD Raon decode with no Triton compute in the selected chain."""

import os

import torch

_ANNOUNCED = set()


def resolve_backend():
    selected = os.environ.get("RAON_ROCM_KDA_DECODE_BACKEND")
    if selected is None:
        selected = (
            "aiter_triton"
            if os.environ.get("RAON_ROCM_KDA_FUSED_DECODE", "0") == "1"
            else "legacy"
        )
    if selected not in ("legacy", "aiter_triton", "flydsl"):
        raise ValueError(f"Unknown ROCm Raon decode backend: {selected}")
    return selected


def conv_weight(layer, weight, dtype):
    """Recover the checkpoint's BF16 values from the widened loader parameter."""
    if weight.dtype == dtype:
        return weight
    if weight.dtype != torch.float32 or dtype != torch.bfloat16:
        raise ValueError("Unsupported FlyDSL convolution weight/state dtype pair")
    from .mxfp8_flydsl import _identity

    key = _identity(layer.conv1d.weight)
    cached = getattr(layer, "_raon_flydsl_conv_weight", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Warm up the lossless FlyDSL conv-weight view before capture"
        )
    packed = weight.to(dtype).contiguous()
    if not torch.equal(packed.float(), weight):
        raise ValueError("FlyDSL conv requires losslessly BF16-representable weights")
    layer._raon_flydsl_conv_weight = (key, packed)
    return packed


def forward(layer, m, inputs, output, conv, state, weight, indices, accepted, cu):
    """Execute FlyDSL conv, vector-decay RDA and gated RMSNorm exactly once."""
    if os.environ.get("COMPILE_ONLY", "0") == "1":
        raise RuntimeError("FlyDSL serving decode cannot run in COMPILE_ONLY mode")
    from aiter.ops.flydsl.causal_conv1d_update_kernels import (
        causal_conv1d_update_flydsl,
    )
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    from .rocm_flydsl_rda import create_gated_norm, create_recurrence

    raw, gate, out_gate, beta, ratio = inputs
    weight = conv_weight(layer, weight, conv.dtype)
    spec = m.num_spec_decodes > 0
    batch = m.num_spec_decodes if spec else m.num_decodes
    tokens, heads = m.num_actual_tokens, layer.local_num_heads
    if (
        layer.A_log.dtype != torch.float32
        or layer.dt_bias.dtype != torch.float32
        or beta.dtype != torch.bfloat16
        or ratio.dtype != torch.float32
        or out_gate.dtype != torch.bfloat16
        or out_gate.stride(-1) != 1
        or layer.o_norm.weight.dtype not in (torch.float32, torch.bfloat16)
    ):
        raise ValueError("Unsupported FlyDSL Raon gate/norm dtype or layout")
    steps = indices.shape[1] if spec else 1
    # The FlyDSL conv wrapper's sentinel is a concrete cache row. Map PAD to
    # NULL for SD; AR keeps physical row zero valid and maps PAD to -1.
    conv_indices = indices[:, 0] if spec else indices
    conv_indices = torch.where(
        conv_indices >= (1 if spec else 0), conv_indices, 0 if spec else -1
    ).contiguous()
    packed = torch.empty((tokens, 3 * heads * 128), device=raw.device, dtype=raw.dtype)
    causal_conv1d_update_flydsl(
        raw[:tokens],
        conv,
        weight,
        None,
        activation="silu",
        conv_state_indices=conv_indices,
        num_accepted_tokens=accepted if spec else None,
        query_start_loc=cu,
        max_query_len=steps,
        null_block_id=0 if spec else -1,
        validate_data=True,
        out=packed,
    )
    intermediate = torch.zeros_like(output)
    recurrence = create_recurrence(
        heads,
        steps,
        spec,
        state.stride(0),
        indices.stride(0),
        indices.stride(1) if spec else 1,
        beta.stride(1),
        ratio.stride(1),
        float(layer.gate_lower_bound),
    )
    stream = torch.cuda.current_stream(raw.device).cuda_stream
    _run_compiled(
        recurrence,
        packed,
        gate,
        beta,
        ratio,
        layer.A_log,
        layer.dt_bias,
        state,
        indices,
        accepted,
        cu,
        intermediate,
        batch,
        stream,
    )
    norm = create_gated_norm(
        heads, layer.o_norm.eps, out_gate.stride(0), str(layer.o_norm.weight.dtype)
    )
    _run_compiled(
        norm, intermediate, out_gate, layer.o_norm.weight, output, tokens, stream
    )
    if spec not in _ANNOUNCED:
        _ANNOUNCED.add(spec)
        print(
            f"RAON_FLYDSL_RDA_DECODE LAUNCHED kind={'sd' if spec else 'ar'} "
            "conv=aiter_flydsl recurrence=flydsl gated_norm=flydsl "
            "decay=per_channel rda_write_gate=1 state=fp32 triton_compute=0",
            flush=True,
        )
