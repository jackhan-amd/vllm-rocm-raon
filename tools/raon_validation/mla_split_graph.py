# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Partition-only fixture vs upstream Kimi math; not native MLA kernel gold."""

import argparse
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from tools.raon_validation.common import environment, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = dict(status="RUNNING", scope=__doc__, environment=environment(), cases=[])
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.models.kimi_k3.amd import mla
    from vllm.models.raon1_mini import rocm_mla_split as split

    torch.manual_seed(1403)
    stream = torch.cuda.Stream()
    state = {}

    class Linear(torch.nn.Module):
        def __init__(self, n, k):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.01,
                requires_grad=False,
            )

        def forward(self, x):
            return F.linear(x, self.weight), None

    def context(_):
        return state["metadata"], None, state["cache"], state["slots"]

    def attention(q, kv, k_pe, output_shape=None, **kwargs):
        state["calls"] += 1
        state["cache"].index_add_(0, state["slots"], kv.float().sum(-1))
        value = q[..., :128].reshape(output_shape)
        return value + (kv.float().mean(-1, keepdim=True) * state["gain"]).bfloat16()

    def normalize(q, kv):
        def norm(x):
            return (
                x.float()
                * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
            ).bfloat16()

        return norm(q), norm(kv)

    def fused(q, kv, k_pe, positions, output_shape, metadata, layer, cache, slots):
        state["fused_calls"] += 1
        return attention(q, kv, k_pe, output_shape)

    original_context = mla.get_attention_context
    original_split_context = split.get_attention_context
    try:
        mla.get_attention_context = split.get_attention_context = context
        with torch.inference_mode(), torch.cuda.stream(stream):
            wrapper = SimpleNamespace(
                q_lora_rank=512,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                num_heads=40,
                qk_head_dim=192,
                v_head_dim=128,
                fused_qkv_a_proj=Linear(1088, 2560),
                q_a_layernorm=torch.nn.Identity(),
                q_b_proj=Linear(7680, 512),
                g_proj=Linear(5120, 2560),
                o_proj=Linear(2560, 5120),
                _normalize_q_kv=normalize,
                rotary_emb=None,
                indexer=None,
                dcp_q_replicate=False,
                _fused_qk_prep=True,
                _fused_decode=fused,
            )

            class Attention:
                layer_name = "fixture"

                def __call__(self, *a, **kw):
                    return attention(*a, **kw)

            wrapper.mla_attn = Attention()
            for rows in (1, 8, 32):
                hidden = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
                positions = torch.arange(rows, device="cuda")
                state.update(
                    metadata=SimpleNamespace(
                        num_actual_tokens=rows, num_decode_tokens=rows
                    ),
                    cache=torch.zeros(128, device="cuda"),
                    slots=torch.arange(rows, device="cuda"),
                    gain=0.5,
                    calls=0,
                    fused_calls=0,
                )
                split.forward(wrapper, positions, hidden)
                cap = BreakableCUDAGraphCapture()
                with cap:
                    actual = split.forward(wrapper, positions, hidden)
                assert cap.num_eager_breaks == 1, cap
                for phase in ("prefill", "decode", "mixed"):
                    hidden.add_(0.125)
                    state.update(
                        metadata=SimpleNamespace(
                            num_actual_tokens=rows,
                            num_decode_tokens=rows if phase == "decode" else 0,
                        ),
                        slots=torch.arange(rows, device="cuda") + 32,
                        gain=1.25,
                    )
                    before = state["cache"].clone()
                    expected = mla.KimiK3MultiHeadLatentAttentionWrapper.forward(
                        wrapper, positions, hidden
                    )
                    expected_cache = state["cache"].clone()
                    state["cache"].copy_(before)
                    state.update(calls=0, fused_calls=0)
                    cap.replay()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(
                        state["cache"], expected_cache, rtol=0, atol=0
                    )
                    assert state["calls"] == 1
                    assert state["fused_calls"] == int(phase == "decode")
                    result["cases"].append(
                        dict(
                            rows=rows,
                            phase=phase,
                            exact=True,
                            cache_once=True,
                            live_phase=True,
                        )
                    )
            result["status"] = "PASS"
    except Exception as error:
        result.update(status="ERROR", error=repr(error))
        raise
    finally:
        mla.get_attention_context = original_context
        split.get_attention_context = original_split_context
        write_json(args.output, result)
    print("MLA_SPLIT", result["status"], len(result["cases"]), flush=True)


if __name__ == "__main__":
    main()
