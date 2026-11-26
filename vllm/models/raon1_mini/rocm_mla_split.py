# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep Raon's MLA projections in graphs; resolve live attention metadata only."""

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.model_executor.layers.attention.attention import get_attention_context

_ANNOUNCED: set[bool] = set()


@eager_break_during_capture
def _attention(wrapper, positions, q, kv, k_pe, out):
    # This is the same phase decision as KimiK3MultiHeadLatentAttentionWrapper.
    # Metadata and KV ownership remain live; no capture-time metadata is saved.
    fuse = wrapper._fused_qk_prep
    metadata = layer = cache = slots = None
    if fuse:
        metadata, layer, cache, slots = get_attention_context(
            wrapper.mla_attn.layer_name
        )
        fuse = (
            metadata is not None
            and slots is not None
            and cache.numel() > 0
            and metadata.num_actual_tokens == metadata.num_decode_tokens
        )
    if fuse:
        value = wrapper._fused_decode(
            q, kv, k_pe, positions, tuple(out.shape), metadata, layer, cache, slots
        )
    else:
        value = wrapper.mla_attn(
            q, kv, k_pe, output_shape=tuple(out.shape), q_dcp_replicated=None
        )
    out.copy_(value)


def forward(wrapper, positions, hidden):
    if (
        wrapper.q_lora_rank is None
        or wrapper.fused_qkv_a_proj is None
        or wrapper.q_b_proj is None
        or wrapper.rotary_emb is not None
        or wrapper.indexer is not None
        or wrapper.dcp_q_replicate
        or not isinstance(hidden, torch.Tensor)
        or hidden.dtype != torch.bfloat16
        or not hidden.is_cuda
    ):
        raise RuntimeError("Split MLA requires TP1 Raon BF16 activations and NoPE MLA")
    # Verbatim projection/normalization/gate order from the Kimi ROCm wrapper.
    # Existing linear adapters choose MXFP4/MXFP8/BF16 kernels and weight layout.
    qkv = wrapper.fused_qkv_a_proj(hidden)[0]
    q_c, kv_lora = qkv.split(
        [wrapper.q_lora_rank, wrapper.kv_lora_rank + wrapper.qk_rope_head_dim], -1
    )
    kv_c, k_pe = kv_lora.split([wrapper.kv_lora_rank, wrapper.qk_rope_head_dim], -1)
    q_input, kv = wrapper._normalize_q_kv(q_c, kv_c)
    q = wrapper.q_b_proj(q_input)[0].view(-1, wrapper.num_heads, wrapper.qk_head_dim)
    out = hidden.new_empty((hidden.shape[0], wrapper.num_heads * wrapper.v_head_dim))
    _attention(wrapper, positions, q, kv, k_pe.unsqueeze(1), out)
    if wrapper.g_proj is not None:
        out = out * wrapper.g_proj(hidden)[0].sigmoid()
    output = wrapper.o_proj(out)[0]
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing not in _ANNOUNCED:
        _ANNOUNCED.add(capturing)
        print(
            f"RAON_ROCM_MLA_SPLIT_GRAPH EXECUTED capturing={int(capturing)} "
            "projections=captured attention=live_kimi_aiter kv_owner=unchanged",
            flush=True,
        )
    return output
