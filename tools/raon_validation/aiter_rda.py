# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selected ROCm RDA candidate vs the existing chain and independent Torch gold.

Checks AR and SD lengths 2/4/8, all accepted counts, reordered state slots,
entire conv/recurrent pools and stateful HIP graph capture/replay. No model
download, serving endpoint or throughput measurement is performed.
"""

from types import MethodType, SimpleNamespace

import torch

from .common import error
from .reference import convolution, rda


class TorchNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.eps = 1e-6
        self.weight = torch.nn.Parameter(torch.ones(dim, device="cuda"))

    def forward(self, x, gate):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (value * self.weight * gate.float().sigmoid()).to(x.dtype)


def run(args, result):
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.models.raon1_mini import rocm_aiter_decode, rocm_kda_graph
    from vllm.models.raon1_mini.kda import RaonKDAAttention, compute_rda_value_ratio

    free, total = torch.accelerator.get_memory_info()
    if free / total < 0.95 or args.dim != 128:
        raise RuntimeError("Use an idle gfx950 GPU and --dim 128")
    torch.manual_seed(args.seed)
    h, d, batch = args.heads, args.dim, args.decode_batch_size
    if batch < 1 or max(args.draft_counts) < 1:
        raise ValueError("Use a positive decode batch size and draft counts")
    result["scope"] = __doc__
    result["candidate_backend"] = args.decode_backend

    def random(*shape):
        return (torch.randn(*shape, device="cuda") * 0.2).bfloat16()

    for spec, length in [(False, 1), *[(True, k + 1) for k in args.draft_counts]]:
        tokens, channels = batch * length, 3 * h * d
        slots = tokens + 5
        history_len = length + 2 if spec else 3
        initial_conv = random(slots, channels, history_len)
        initial_state = torch.randn(slots, h, d, d, device="cuda") * 0.01
        weight = random(channels, 1, 4)
        if getattr(args, "conv_weight_fp32", False):
            weight = weight.float()
        a_log = torch.randn(h, device="cuda") * 0.1
        dt_bias = torch.randn(h * d, device="cuda") * 0.1
        norm = TorchNorm(d)
        raw, g, beta, delta = (
            random(tokens, channels),
            random(1, tokens, h, d),
            random(1, tokens, h),
            random(1, tokens, h),
        )
        out_gate = random(tokens, h, d)
        ratio = compute_rda_value_ratio(beta, delta)
        inputs = (raw, g, out_gate, beta, ratio)
        indices = torch.arange(1, tokens + 1, device="cuda").int().view(batch, length)
        indices = indices.roll(1, 0).contiguous()
        cu = torch.arange(0, tokens + 1, length, device="cuda", dtype=torch.int32)
        metadata = SimpleNamespace(
            num_prefills=0,
            num_decodes=0 if spec else batch,
            num_spec_decodes=batch if spec else 0,
            num_actual_tokens=tokens,
            has_initial_state=None,
            non_spec_query_start_loc=None if spec else cu,
            non_spec_state_indices_tensor=None if spec else indices[:, 0],
            spec_token_indx=None,
            non_spec_token_indx=None,
            spec_state_indices_tensor=indices if spec else None,
            spec_query_start_loc=cu if spec else None,
            num_accepted_tokens=torch.ones(batch, device="cuda", dtype=torch.int32)
            if spec
            else None,
        )

        def fixture(
            fused,
            initial_conv=initial_conv,
            initial_state=initial_state,
            weight=weight,
            a_log=a_log,
            dt_bias=dt_bias,
            norm=norm,
        ):
            conv = initial_conv.clone()
            state = initial_state.clone()
            layer = SimpleNamespace(
                head_dim=d,
                local_num_heads=h,
                local_projection_size=h * d,
                kda_decode_backend="rocm_recurrent",
                kda_prefill_backend="rocm_fused",
                kda_cpr=False,
                kda_sgl_fused=False,
                kda_sgl_fused_spec=False,
                rocm_aiter_fused_decode=fused,
                rocm_decode_backend=args.decode_backend if fused else "legacy",
                _raon_decode_graph_local=True,
                conv1d=SimpleNamespace(weight=weight, bias=None),
                A_log=a_log,
                dt_bias=dt_bias,
                gate_lower_bound=-5.0,
                o_norm=norm,
                kv_cache=(
                    conv if is_conv_state_dim_first() else conv.transpose(-1, -2),
                    state,
                ),
            )
            layer.parameters = lambda: iter(
                (layer.conv1d.weight, layer.A_log, layer.dt_bias, layer.o_norm.weight)
            )
            layer._scale_packed_v = MethodType(RaonKDAAttention._scale_packed_v, layer)
            layer._forward_step = MethodType(RaonKDAAttention._forward_step, layer)
            return layer, conv, state

        graph_layer, graph_conv, graph_state = fixture(True)
        for accepted in range(1, length + 1) if spec else (1,):
            if spec:
                metadata.num_accepted_tokens.fill_(accepted)
            indices.copy_(indices.roll(1, 0))
            raw.add_(0.01)
            old, old_conv, old_state = fixture(False)
            fused, fused_conv, fused_state = fixture(True)
            old_out = torch.zeros(1, tokens, h, d, device="cuda", dtype=torch.bfloat16)
            actual = torch.zeros_like(old_out)
            old._forward_step(metadata, *inputs, old_out)
            assert rocm_aiter_decode.try_forward(fused, metadata, inputs, actual)

            gold_out = torch.empty_like(actual)
            gold_state, gold_conv = initial_state.clone(), initial_conv.clone()
            for n in range(batch):
                offset = accepted - 1 if spec else 0
                conv_slot = int(indices[n, 0])
                state_slot = int(indices[n, offset])
                history = (
                    initial_conv[conv_slot : conv_slot + 1, :, offset : offset + 3]
                    if spec
                    else initial_conv[conv_slot : conv_slot + 1]
                )
                lo, hi = n * length, (n + 1) * length
                convolved, _ = convolution(raw[None, lo:hi], history, weight[:, 0])
                q, k, v = [
                    part.view(1, length, h, d) for part in convolved.chunk(3, -1)
                ]
                value, states = rda(
                    q,
                    k,
                    v,
                    g[:, lo:hi],
                    beta[:, lo:hi],
                    delta[:, lo:hi],
                    a_log,
                    dt_bias,
                    initial_state[state_slot : state_slot + 1],
                )
                gold_out[:, lo:hi] = norm(value.bfloat16(), out_gate[lo:hi])
                for token in range(length):
                    gold_state[indices[n, token]] = states[0, token]
                if spec:
                    gold_conv[conv_slot, :, :2] = history[0, :, 1:]
                    gold_conv[conv_slot, :, 2 : 2 + length] = raw[lo:hi].T
                else:
                    gold_conv[conv_slot] = torch.cat(
                        [history[0, :, 1:], raw[lo:hi].T], -1
                    )

            graph_conv.copy_(initial_conv)
            graph_state.copy_(initial_state)
            graph_out = torch.empty_like(actual)
            assert rocm_kda_graph.try_forward(graph_layer, metadata, inputs, graph_out)
            written = indices.flatten().long()
            untouched = torch.ones(slots, device="cuda", dtype=torch.bool)
            untouched[written] = False
            result["cases"].append(
                dict(
                    kind="sd" if spec else "ar",
                    length=length,
                    accepted=accepted,
                    output_vs_existing=error(actual, old_out),
                    state_vs_existing=error(fused_state[written], old_state[written]),
                    output_vs_torch_gold=error(actual, gold_out),
                    state_vs_torch_gold=error(
                        fused_state[written], gold_state[written]
                    ),
                    untouched_recurrent_exact={
                        "pass": bool(
                            torch.equal(
                                fused_state[untouched], initial_state[untouched]
                            )
                        )
                    },
                    conv_vs_existing={"pass": bool(torch.equal(fused_conv, old_conv))},
                    conv_vs_torch_gold={
                        "pass": bool(torch.equal(fused_conv, gold_conv))
                    },
                    graph_output_exact=error(graph_out, actual, 0, 0, 0),
                    graph_state_exact=error(graph_state, fused_state, 0, 0, 0),
                    graph_conv_exact={
                        "pass": bool(torch.equal(graph_conv, fused_conv))
                    },
                )
            )
