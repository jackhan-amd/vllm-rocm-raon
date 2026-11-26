# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in native DFlash2 attention candidate for Raon's eight-query draft."""

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform
from vllm.v1.attention.backends.rocm_aiter_fa import (
    AiterFlashAttentionBackend,
    AiterFlashAttentionImpl,
)
from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadataBuilder

from .rocm_draft_attention import DraftAttentionWorkspace


class NativeDraftMetadataBuilder(RocmAttentionMetadataBuilder):
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        common = common_attn_metadata
        counts = common.query_start_loc_cpu.diff().tolist()
        # The target graph manager also prepares unused draft-group metadata
        # with target causality. Validate the real draft build and forward.
        if common.causal is False and any(x not in (0, 8) for x in counts):
            raise ValueError(
                "Native CK draft candidate requires noncausal eight-query requests; "
                f"causal={common.causal}, query_lengths={counts}"
            )
        return super().build(common_prefix_len, common, fast_build)


class NativeDraftBackend(AiterFlashAttentionBackend):
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "bfloat16", "fp8_e4m3"]

    @staticmethod
    def get_name():
        return "ROCM_RAON_DFLASH"

    @classmethod
    def get_supported_head_sizes(cls):
        return [128]

    @classmethod
    def supports_sink(cls):
        return False

    @staticmethod
    def get_impl_cls():
        return NativeDraftImpl

    @staticmethod
    def get_builder_cls():
        return NativeDraftMetadataBuilder


class NativeDraftImpl(AiterFlashAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            self.num_heads != 20
            or self.num_kv_heads != 5
            or self.head_size != 128
            or self.sliding_window[0] != 2047
            or self.alibi_slopes is not None
            or self.sinks is not None
            or self.logits_soft_cap != 0
            or self.kv_sharing_target_layer_name is not None
            or rocm_aiter_ops.is_shuffle_kv_cache_enabled()
        ):
            raise ValueError("Native draft candidate requires Raon DFlash2 NHD cache")
        self.workspaces = {}
        self.announced = False

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("Native draft output quantization is unsupported")
        if attn_metadata is None:
            return output.fill_(0)
        m = attn_metadata
        if (
            m.causal is not False
            or m.max_query_len != 8
            or key is None
            or value is None
        ):
            raise ValueError(
                "Native draft candidate requires noncausal query blocks of 8"
            )
        n = m.num_actual_tokens
        batch = m.seq_lens.numel()
        cache_key = (batch, n)
        if cache_key not in self.workspaces:
            self.workspaces[cache_key] = DraftAttentionWorkspace(
                batch, n, 8, query.device, provider="ck"
            )
        kc, vc = self._split_kv_cache(kv_cache)
        if self.kv_cache_dtype == "fp8_e4m3":
            kc, vc = [x.view(current_platform.fp8_dtype()) for x in (kc, vc)]
        self.workspaces[cache_key](
            query[:n],
            key[:n],
            value[:n],
            kc,
            vc,
            m.block_table,
            m.query_start_loc,
            m.seq_lens,
            layer._k_scale,
            layer._v_scale,
            output[:n],
            self.scale,
        )
        if not self.announced:
            self.announced = True
            print(
                "RAON_NATIVE_DRAFT LAUNCHED attention=ck_hip gather=flydsl "
                "cache_write=hip query=bf16 window=2048 causal=0",
                flush=True,
            )
        return output
