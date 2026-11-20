# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raon MLA through the same ROCm wrapper used by Kimi hybrid attention."""

from vllm.models.kimi_k3.amd.linear import KimiMLAAttention


class MultiHeadLatentAttention(KimiMLAAttention):
    def __init__(
        self,
        config,
        *args,
        use_output_gate=False,
        use_rope=False,
        non_causal_multi_token_decode=False,
        aux_stream=None,
        **kwargs,
    ):
        if use_rope or non_causal_multi_token_decode:
            raise NotImplementedError(
                "Initial ROCm target integration supports causal NoPE target MLA"
            )
        if bool(config.mla_use_output_gate) != bool(use_output_gate):
            raise ValueError("Preserve the checkpoint output-gate configuration")
        super().__init__(config, *args, use_nope=True, **kwargs)
        self.rms_norm_eps = 1e-6
        self.q_a_layernorm.variance_epsilon = 1e-6
        self.kv_a_layernorm.variance_epsilon = 1e-6
        self.rotary_emb = None

    def forward(self, positions, hidden_states):
        RUNTIME_WITNESS["calls"] += 1
        RUNTIME_WITNESS["max_tokens"] = max(
            RUNTIME_WITNESS["max_tokens"], hidden_states.shape[0]
        )
        return super().forward(positions, hidden_states)


RUNTIME_WITNESS = {"calls": 0, "max_tokens": 0}
