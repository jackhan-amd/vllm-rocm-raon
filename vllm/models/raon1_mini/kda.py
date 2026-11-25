# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# ROCm adapter derived from KRAFTON Raon kda.py
# at 335a45e3f9d8a7f5f76c884734b5797ec9623fb3.
import os
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.models.kimi_k3.amd.ops.kda_prefill import chunk_kda_prefill
from vllm.models.kimi_k3.amd.ops.third_party.kda import (
    fused_recurrent_kda,
    fused_recurrent_kda_packed_decode,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata as RaonRDAMetadata

from . import kda_vsplit as _kda_vsplit
from .rda_base import RaonRDABase, _flashkda_prefill

_FLASHKDA_DEDICATED: dict = {}


def _flashkda_dedicated_buffers(specs, device):
    key = (str(device), tuple((tuple(shape), dtype) for shape, dtype in specs))
    bufs = _FLASHKDA_DEDICATED.get(key)
    if bufs is None:
        bufs = [
            torch.empty(shape, dtype=dtype, device=device) for shape, dtype in specs
        ]
        _FLASHKDA_DEDICATED[key] = bufs
    return bufs


def compute_rda_value_ratio(beta_logits, write_delta_logits):
    beta = beta_logits.float()
    return torch.exp(
        F.logsigmoid(beta + write_delta_logits.float()) - F.logsigmoid(beta)
    )


class RaonKDAAttention(RaonRDABase):
    def __init__(self, config, vllm_config, prefix=""):
        super().__init__(config, vllm_config, prefix)
        self.kda_decode_backend = "rocm_recurrent"
        from .rocm_flydsl_decode import resolve_backend

        self.rocm_decode_backend = resolve_backend()
        self.rocm_aiter_fused_decode = self.rocm_decode_backend != "legacy"
        parallel = vllm_config.parallel_config
        self._raon_decode_graph_local = (
            parallel.tensor_parallel_size == 1
            and parallel.data_parallel_size == 1
            and not parallel.enable_expert_parallel
        )
        self.rda_w_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.rda_w_proj",
        )
        self.kda_chunk_replay = False
        self.kda_cpr = False
        self.kda_fla_seams = False
        self.kda_inkernel_fla = False
        self.kda_prefill_front_fused = False
        self.kda_proj_persistent = False
        self.kda_sgl_fused = False
        self.kda_sgl_fused_spec = False
        self.rda_fused = False

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        num_tokens = hidden_states.size(0)
        projected_qkvgfab = self.in_proj_qkvgfab(hidden_states)[0]
        split_sizes = [
            3 * self.local_projection_size,
            self.local_projection_size,
            self.head_dim,
            self.local_num_heads,
        ]
        if self.in_proj_padding:
            split_sizes.append(self.in_proj_padding)
        projected = projected_qkvgfab.split(split_sizes, dim=-1)
        mixed_qkv, g_proj_states, f_a, beta = projected[:4]
        g1 = self.f_b_proj(f_a)[0]
        beta = beta.unsqueeze(0)
        delta = self.rda_w_proj(hidden_states)[0].unsqueeze(0)
        ratio = compute_rda_value_ratio(beta, delta)
        g1 = rearrange(g1, "n (h d) -> 1 n h d", d=self.head_dim)
        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)
        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._forward(
            mixed_qkv=mixed_qkv,
            g1=g1,
            g2=g2,
            beta=beta,
            ratio=ratio,
            core_attn_out=core_attn_out,
        )
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        return self.o_proj(core_attn_out)[0]

    def _scale_packed_v(
        self, mixed_qkv: torch.Tensor, ratio: torch.Tensor, beta: torch.Tensor
    ) -> None:
        """Apply the RDA ratio to the value block of a packed conv output.

        ``mixed_qkv`` is ``[T, 3 * local_projection_size]`` — q | k | v, each
        ``local_num_heads * head_dim`` wide — so only the trailing block is
        scaled.  The slice is row-strided (stride ``3P``) but its last axis is
        contiguous, which is exactly the case ``view`` can split, giving a
        writable ``[T, H, D]`` view into the packed buffer.  ``ratio`` is
        ``[1, T, H]``; the fp32 product is rounded once, matching Megatron's
        ``(value.float() * ratio.unsqueeze(-1)).to(value.dtype)``.

        In-place is safe: both call sites hand us a freshly allocated conv
        output buffer, never an aliased projection.
        """
        v_blk = mixed_qkv[:, 2 * self.local_projection_size :].view(
            -1, self.local_num_heads, self.head_dim
        )
        v_blk.copy_((v_blk.float() * ratio[0].unsqueeze(-1)).to(v_blk.dtype))

    @eager_break_during_capture
    def _forward(
        self,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        ratio: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if attn_metadata_raw is None:
            return
        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata_narrowed, RaonRDAMetadata)
        m = attn_metadata_narrowed
        _record_route(self.prefix, m)
        if os.environ.get("RAON_KDA_DECODE_GRAPH", "0") == "1":
            from .rocm_kda_graph import try_forward

            if try_forward(self, m, (mixed_qkv, g1, g2, beta, ratio), core_attn_out):
                return
        self._forward_step(m, mixed_qkv, g1, g2, beta, ratio, core_attn_out)

    def _forward_step(self, m, mixed_qkv, g1, g2, beta, ratio, core_attn_out):
        if self.rocm_aiter_fused_decode:
            from .rocm_aiter_decode import try_forward

            if try_forward(self, m, (mixed_qkv, g1, g2, beta, ratio), core_attn_out):
                return
        has_initial_state = m.has_initial_state
        non_spec_query_start_loc = m.non_spec_query_start_loc
        non_spec_state_indices_tensor = m.non_spec_state_indices_tensor
        spec_token_indx = m.spec_token_indx
        non_spec_token_indx = m.non_spec_token_indx
        spec_state_indices_tensor = m.spec_state_indices_tensor
        spec_query_start_loc = m.spec_query_start_loc
        num_accepted_tokens = m.num_accepted_tokens
        num_actual_tokens = m.num_actual_tokens
        has_spec_decode = m.num_spec_decodes > 0
        if self.kda_cpr and has_spec_decode:
            raise RuntimeError(
                "RAON_KDA_CPR does not support speculative decode (the "
                "spec path bypasses the 64-boundary entry-chain bookkeeping)"
            )
        mixed_qkv = mixed_qkv[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]
        ratio = ratio[:, :num_actual_tokens]
        if self.kda_sgl_fused_spec and has_spec_decode:
            from . import kda_sgl_fused_spec as _sgl_fused_spec

            if _sgl_fused_spec.try_spec(
                self, mixed_qkv, g1, g2, beta, ratio, core_attn_out, m
            ):
                return
        conv_state, recurrent_state = self.kv_cache
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        q_conv_weight, k_conv_weight, v_conv_weight = conv_weights.split(
            self.local_projection_size, dim=0
        )
        q_conv_state, k_conv_state, v_conv_state = conv_state.split(
            self.local_projection_size, dim=-2
        )
        if has_spec_decode:
            if m.num_prefills == 0 and m.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                g1_spec, beta_spec, ratio_spec = (g1, beta, ratio)
                mixed_qkv_ns = g1_ns = beta_ns = ratio_ns = g2_ns = None
            else:
                assert spec_token_indx is not None
                assert non_spec_token_indx is not None
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                g1_spec = g1.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                ratio_spec = ratio.index_select(1, spec_token_indx)
                mixed_qkv_ns = mixed_qkv.index_select(0, non_spec_token_indx)
                g1_ns = g1.index_select(1, non_spec_token_indx)
                beta_ns = beta.index_select(1, non_spec_token_indx)
                ratio_ns = ratio.index_select(1, non_spec_token_indx)
                g2_ns = g2.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = g1_spec = beta_spec = ratio_spec = None
            mixed_qkv_ns, g1_ns, beta_ns, ratio_ns = (mixed_qkv, g1, beta, ratio)
            g2_ns = g2
        core_attn_out_spec = None
        if has_spec_decode:
            assert spec_state_indices_tensor is not None
            assert spec_query_start_loc is not None
            spec_conv_indices = spec_state_indices_tensor[:, 0][: m.num_spec_decodes]
            spec_max_query_len = spec_state_indices_tensor.size(-1)
            spec_conv_out = torch.empty_like(mixed_qkv_spec)
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                activation="silu",
                conv_state_indices=spec_conv_indices,
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_max_query_len,
                validate_data=False,
                out=spec_conv_out,
            )
            self._scale_packed_v(mixed_qkv_spec, ratio_spec, beta_spec)
            q_spec, k_spec, v_spec = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
                for x in mixed_qkv_spec.split(self.local_projection_size, dim=-1)
            )
            spec_cu_seqlens = spec_query_start_loc[: m.num_spec_decodes + 1]
            spec_out = (
                core_attn_out[:, : q_spec.shape[1]]
                if m.num_prefills == 0 and m.num_decodes == 0
                else None
            )
            core_attn_out_spec, _ = fused_recurrent_kda(
                q=q_spec,
                k=k_spec,
                v=v_spec,
                raw_g=g1_spec,
                raw_beta=beta_spec,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                lower_bound=self.gate_lower_bound,
                initial_state=recurrent_state,
                cu_seqlens=spec_cu_seqlens,
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                out=spec_out,
            )
        core_attn_out_non_spec = None
        if mixed_qkv_ns is not None:
            assert g1_ns is not None and beta_ns is not None
            if m.num_prefills > 0:
                q_ns, k_ns, v_ns = mixed_qkv_ns.split(
                    self.local_projection_size, dim=-1
                )

                def _prefill_conv(
                    x: torch.Tensor, state: torch.Tensor, weight: torch.Tensor
                ) -> torch.Tensor:
                    return causal_conv1d_fn(
                        x.transpose(0, 1),
                        weight,
                        None,
                        activation="silu",
                        conv_states=state,
                        has_initial_state=has_initial_state,
                        cache_indices=non_spec_state_indices_tensor,
                        query_start_loc=non_spec_query_start_loc,
                        metadata=m,
                    ).transpose(0, 1)

                q_ns = _prefill_conv(q_ns, q_conv_state, q_conv_weight)
                k_ns = _prefill_conv(k_ns, k_conv_state, k_conv_weight)
                v_ns = _prefill_conv(v_ns, v_conv_state, v_conv_weight)
                q_ns, k_ns, v_ns = (
                    rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
                    for x in (q_ns, k_ns, v_ns)
                )
                assert ratio_ns is not None
                v_ns = (v_ns.float() * ratio_ns.unsqueeze(-1)).to(v_ns.dtype)
                assert non_spec_state_indices_tensor is not None
                assert has_initial_state is not None
                initial_state = gather_initial_states(
                    recurrent_state, non_spec_state_indices_tensor, has_initial_state
                )
                core_attn_out_non_spec, last_recurrent_state = chunk_kda_prefill(
                    q=q_ns,
                    k=k_ns,
                    v=v_ns,
                    raw_g=g1_ns,
                    raw_beta=beta_ns,
                    A_log=self.A_log,
                    g_bias=self.dt_bias,
                    lower_bound=self.gate_lower_bound,
                    initial_state=initial_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=non_spec_query_start_loc,
                    use_fused_chunk=self.kda_prefill_backend == "rocm_fused",
                )
                recurrent_state[non_spec_state_indices_tensor] = last_recurrent_state
            else:
                assert non_spec_state_indices_tensor is not None
                decode_conv_indices = non_spec_state_indices_tensor[
                    : mixed_qkv_ns.size(0)
                ]
                if self.kda_sgl_fused and (not has_spec_decode):
                    from . import kda_sgl_fused as _sgl_fused

                    if _sgl_fused.try_decode(
                        self,
                        mixed_qkv_ns,
                        g1_ns,
                        g2_ns,
                        beta_ns,
                        ratio_ns,
                        decode_conv_indices,
                        core_attn_out,
                        num_actual_tokens,
                    ):
                        return
                packed_conv_out = torch.empty_like(mixed_qkv_ns)
                mixed_qkv_ns = causal_conv1d_update(
                    mixed_qkv_ns,
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    activation="silu",
                    conv_state_indices=decode_conv_indices,
                    validate_data=True,
                    out=packed_conv_out,
                )
                self._scale_packed_v(mixed_qkv_ns, ratio_ns, beta_ns)
                if self.kda_decode_backend == "flashkda":
                    t_dec = mixed_qkv_ns.size(0)
                    q_d, k_d, v_d = (
                        rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
                        for x in mixed_qkv_ns.split(self.local_projection_size, dim=-1)
                    )
                    cu = torch.arange(
                        0, t_dec + 1, device=mixed_qkv_ns.device, dtype=torch.int32
                    )
                    init = gather_initial_states(
                        recurrent_state,
                        decode_conv_indices,
                        torch.ones(t_dec, dtype=torch.bool, device=mixed_qkv_ns.device),
                    )
                    assert self.gate_lower_bound is not None
                    assert self._flashkda_buffer_specs is not None
                    workspace_out, final_state, workspace = _flashkda_dedicated_buffers(
                        self._flashkda_buffer_specs, mixed_qkv.device
                    )
                    core_attn_out_non_spec, last_state = _kda_vsplit.prefill(
                        m,
                        stock=_flashkda_prefill,
                        q=q_d,
                        k=k_d,
                        v=v_d,
                        g=g1_ns,
                        beta=beta_ns,
                        A_log=self.A_log,
                        dt_bias=self.dt_bias,
                        lower_bound=self.gate_lower_bound,
                        initial_state=init,
                        cu_seqlens=cu,
                        out=workspace_out[:, :t_dec],
                        final_state=final_state[:t_dec],
                        workspace=workspace,
                    )
                    recurrent_state[decode_conv_indices] = last_state
                else:
                    core_attn_out_non_spec, _ = fused_recurrent_kda_packed_decode(
                        mixed_qkv=mixed_qkv_ns,
                        raw_g=g1_ns,
                        raw_beta=beta_ns,
                        A_log=self.A_log,
                        dt_bias=self.dt_bias,
                        lower_bound=self.gate_lower_bound,
                        initial_state=recurrent_state,
                        state_indices=decode_conv_indices,
                    )
        if core_attn_out_spec is not None and core_attn_out_non_spec is not None:
            core_attn_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            core_attn_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
        elif core_attn_out_non_spec is not None:
            if self.kda_prefill_backend != "flashkda" or m.num_prefills == 0:
                core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[
                    0, :num_actual_tokens
                ]
        else:
            assert core_attn_out_spec is not None
        core_attn_out.copy_(self.o_norm(core_attn_out, g2))


RUNTIME_WITNESS: dict[str, Any] = {
    "prefill_calls": 0,
    "decode_calls": 0,
    "spec_calls": 0,
    "layers": [],
}


def _record_route(prefix, metadata):
    w = RUNTIME_WITNESS
    if prefix not in w["layers"]:
        w["layers"].append(prefix)
    w["prefill_calls"] += int(metadata.num_prefills > 0)
    w["decode_calls"] += int(metadata.num_decodes > 0)
    w["spec_calls"] += int(metadata.num_spec_decodes > 0)
