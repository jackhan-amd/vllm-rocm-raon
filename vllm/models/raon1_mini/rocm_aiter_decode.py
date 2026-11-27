# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind explicit ROCm RDA candidates to Raon's erase/write gate contract.

The legacy AITER-derived fused kernel uses Triton. FlyDSL selects the native
three-kernel chain, while hip_rda_ar selects the experimental Kimi HIP AR
derivative. MoE backend selection is independent.
"""

import torch

_ANNOUNCED = set()


def try_forward(layer, m, inputs, out):
    x, gate, out_gate, beta, ratio = inputs
    if m.num_prefills or (m.num_decodes and m.num_spec_decodes):
        if "mixed" not in _ANNOUNCED:
            _ANNOUNCED.add("mixed")
            print(
                "RAON_AITER_RDA_DECODE INERT prefill/mixed batch "
                "uses the existing AMD chain",
                flush=True,
            )
        return False
    spec = m.num_spec_decodes > 0
    batch = m.num_spec_decodes if spec else m.num_decodes
    tokens = m.num_actual_tokens
    if batch < 1 or tokens < 1:
        return False
    if not torch.version.hip or not x.is_cuda:
        raise RuntimeError("AITER-derived RDA decode requires ROCm")
    arch = torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0]
    if arch != "gfx950" or layer.head_dim != 128:
        raise RuntimeError(
            "AITER-derived RDA decode currently requires gfx950 and head_dim=128"
        )
    if layer.conv1d.bias is not None or layer.gate_lower_bound is None:
        raise ValueError("Fused RDA decode requires bias-free conv and a bounded gate")
    h, d = layer.local_num_heads, layer.head_dim
    if x.dtype != torch.bfloat16 or tuple(ratio.shape[-1:]) != (h,):
        raise ValueError(
            "Fused RDA decode expects BF16 projections and per-head RDA ratio"
        )
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    conv, state = layer.kv_cache
    if not is_conv_state_dim_first():
        conv = conv.transpose(-1, -2)
    if (
        state.dtype != torch.float32
        or tuple(state.shape[1:]) != (h, d, d)
        or tuple(state.stride()[1:]) != (d * d, d, 1)
    ):
        raise ValueError("Fused RDA decode requires FP32 V-major recurrent cache")
    weight = layer.conv1d.weight.view(3 * h * d, -1)
    width = weight.shape[-1]
    if width != 4:
        raise ValueError("This AITER-derived RDA candidate requires conv width 4")
    if (
        tuple(x.shape[1:]) != (3 * h * d,)
        or x.stride(-1) != 1
        or tuple(gate.shape) != (1, x.shape[0], h, d)
        or tuple(beta.shape) != (1, x.shape[0], h)
        or tuple(ratio.shape) != (1, x.shape[0], h)
        or beta.stride(-1) != 1
        or not gate.is_contiguous()
        or ratio.stride(-1) != 1
    ):
        raise ValueError(
            "Unsupported RDA fused projection shape/stride; gate must be contiguous"
        )
    if spec:
        indices = m.spec_state_indices_tensor[:batch]
        accepted = m.num_accepted_tokens[:batch]
        conv_indices = indices[:, 0]
        cu = m.spec_query_start_loc[: batch + 1]
        spec_len = indices.shape[1]
        if conv.shape[-1] < spec_len + width - 2:
            raise ValueError(
                "Speculative conv history is too short for the state slots"
            )
        if batch > 1 and accepted.stride(0) != 1:
            raise ValueError("AITER fused accepted-token counts must be contiguous")
    else:
        indices = m.non_spec_state_indices_tensor[:batch]
        conv_indices = accepted = indices  # unused pointer arguments
        cu = m.non_spec_query_start_loc
        if cu is None:
            cu = torch.arange(batch + 1, dtype=torch.int32, device=x.device)
        else:
            cu = cu[: batch + 1]
        spec_len = 1
        if tokens != batch:
            raise ValueError("Normal fused decode needs one token per request")
    output = out.view(-1, h * d)[:tokens]
    if getattr(layer, "rocm_decode_backend", "") == "hip_rda_ar":
        if spec:
            raise RuntimeError("Experimental HIP RDA AR does not support SD")
        from .rocm_hip_rda import forward as hip_forward

        hip_forward(layer, inputs, output, conv, state, weight, indices)
        return True
    if not output.is_contiguous():
        raise ValueError("Fused RDA output must be contiguous")
    # Match AITER's initialized spec output for NULL/PAD rows. The fused
    # kernel owns every live output and the single conv/recurrent state update.
    output.zero_()
    if getattr(layer, "rocm_decode_backend", "aiter_triton") == "flydsl":
        from .rocm_flydsl_decode import forward

        forward(layer, m, inputs, output, conv, state, weight, indices, accepted, cu)
        return True
    from .rocm_aiter_rda_kernel import fused_rda_conv_recurrent_norm_kernel

    fused_rda_conv_recurrent_norm_kernel[(batch, h)](
        x,
        weight,
        conv,
        gate,
        beta,
        ratio,
        layer.A_log,
        layer.dt_bias,
        state,
        indices,
        accepted,
        conv_indices,
        cu,
        layer.o_norm.weight,
        out_gate,
        output,
        layer.gate_lower_bound,
        layer.o_norm.eps,
        d**-0.5,
        H=h,
        K=d,
        V=d,
        W=width,
        STATE_LEN=conv.shape[-1],
        SHIFT_LEN_P2=1 << (max(conv.shape[-1] - 1, 1) - 1).bit_length(),
        IS_SPEC_DECODING=spec,
        SPEC_LEN=spec_len,
        stride_x_tok=x.stride(0),
        stride_cw_group=h * d * weight.stride(0),
        stride_cw_width=weight.stride(1),
        stride_cw_ch=weight.stride(0),
        stride_cs_slot=conv.stride(0),
        stride_cs_dim=conv.stride(1),
        stride_cs_pos=conv.stride(2),
        stride_beta_tok=beta.stride(1),
        stride_ratio_tok=ratio.stride(1),
        stride_og_tok=out_gate.stride(0),
        stride_ssm_slot=state.stride(0),
        stride_indices_seq=max(indices.stride(0), 1),
        stride_indices_tok=indices.stride(1) if spec else 1,
        stride_cidx=max(conv_indices.stride(0), 1),
        num_warps=4,
    )
    if spec not in _ANNOUNCED:
        _ANNOUNCED.add(spec)
        print(
            f"RAON_AITER_RDA_DECODE LAUNCHED kind={'sd' if spec else 'ar'} "
            "origin=aiter backend=triton rda_write_gate=1 conv_recurrent_norm=fused",
            flush=True,
        )
    return True
