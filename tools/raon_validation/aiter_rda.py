# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selected ROCm RDA candidate vs the existing chain and independent Torch gold.

Checks AR and SD lengths 2/4/8, all accepted counts, reordered state slots,
entire conv/recurrent pools and stateful HIP graph capture/replay. Optional
AR diagnostics record native kernel names and warmed isolated graph timing.
No model download, serving endpoint or token throughput measurement is performed.
"""

import copy
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

    schedules = [(False, 1)]
    if not getattr(args, "decode_ar_only", False):
        schedules += [(True, k + 1) for k in args.draft_counts]
    for spec, length in schedules:
        tokens, channels = batch * length, 3 * h * d
        slots = tokens + 5
        history_len = length + 2 if spec else args.decode_ar_history
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
        if spec and getattr(args, "decode_spec_null", False):
            indices[-1].zero_()
        if not spec and getattr(args, "decode_ar_padding", False):
            indices[0, 0] = 0
            if batch > 1:
                indices[-1, 0] = -1
        cu = torch.arange(0, tokens + 1, length, device="cuda", dtype=torch.int32)
        if spec and getattr(args, "decode_spec_ragged", False):
            if getattr(args, "decode_mixed", False):
                raise ValueError("Run ragged pure-SD and mixed fixtures separately")
            spans = [1 + n % length for n in range(batch)]
            picked = torch.tensor(
                [n * length + t for n, span in enumerate(spans) for t in range(span)],
                device="cuda",
                dtype=torch.long,
            )
            raw, out_gate = raw[picked], out_gate[picked]
            g, beta, delta = (value[:, picked] for value in (g, beta, delta))
            ratio = compute_rda_value_ratio(beta, delta)
            inputs = (raw, g, out_gate, beta, ratio)
            tokens = picked.numel()
            cu = (
                torch.tensor([0] + spans, device="cuda", dtype=torch.int32)
                .cumsum(0)
                .int()
            )
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
            backend=None,
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
                rocm_decode_backend=(backend or args.decode_backend)
                if fused
                else "legacy",
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
        for accepted in range(1, length + 1) if spec else (1, 1, 1):
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

            gold_out = torch.zeros_like(actual)
            gold_state, gold_conv = initial_state.clone(), initial_conv.clone()
            for n in range(batch):
                offset = accepted - 1 if spec else 0
                conv_slot = int(indices[n, 0])
                state_slot = int(indices[n, offset])
                if conv_slot <= 0 or state_slot <= 0:
                    continue
                history = (
                    initial_conv[conv_slot : conv_slot + 1, :, offset : offset + 3]
                    if spec
                    else initial_conv[conv_slot : conv_slot + 1, :, :3]
                )
                lo, hi = int(cu[n]), int(cu[n + 1])
                convolved, _ = convolution(raw[None, lo:hi], history, weight[:, 0])
                q, k, v = [
                    part.view(1, hi - lo, h, d) for part in convolved.chunk(3, -1)
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
                for token in range(hi - lo):
                    gold_state[indices[n, token]] = states[0, token]
                if spec:
                    gold_conv[conv_slot, :, :2] = history[0, :, 1:]
                    gold_conv[conv_slot, :, 2 : 2 + hi - lo] = raw[lo:hi].T
                else:
                    gold_conv[conv_slot, :, :3] = torch.cat(
                        [history[0, :, 1:], raw[lo:hi].T], -1
                    )

            graph_conv.copy_(initial_conv)
            graph_state.copy_(initial_state)
            graph_out = torch.empty_like(actual)
            assert rocm_kda_graph.try_forward(graph_layer, metadata, inputs, graph_out)
            written = indices.flatten().long()
            written = written[written > 0]
            untouched = torch.ones(slots, device="cuda", dtype=torch.bool)
            untouched[written] = False
            result["cases"].append(
                dict(
                    kind="sd" if spec else "ar",
                    length=length,
                    accepted=accepted,
                    output_vs_existing=error(actual, old_out),
                    state_vs_existing=error(fused_state[written], old_state[written])
                    if written.numel()
                    else {"pass": bool(torch.equal(fused_state, old_state))},
                    output_vs_torch_gold=error(actual, gold_out),
                    state_vs_torch_gold=error(fused_state[written], gold_state[written])
                    if written.numel()
                    else {"pass": bool(torch.equal(fused_state, gold_state))},
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
        if spec and length == 8 and getattr(args, "decode_mixed", False):
            if batch < 2:
                raise ValueError("Mixed validation needs at least two requests")
            source_ids = torch.cat(
                (
                    torch.arange(8, device="cuda"),
                    torch.arange(8, tokens, 8, device="cuda"),
                )
            ).long()
            prefill = getattr(args, "decode_mixed_prefill", 0)
            if prefill:
                if not 1 <= prefill <= 512 or batch < 3:
                    raise ValueError("Use 1..512 prefill tokens and batch >=3")
                source_ids = torch.cat(
                    (
                        source_ids,
                        tokens
                        - length
                        + (torch.arange(1, prefill, device="cuda") % length),
                    )
                )
            order = torch.arange(source_ids.numel(), device="cuda").roll(3)
            source_ids = source_ids[order]
            inverse = order.argsort()
            mixed = copy.copy(metadata)
            mixed.num_spec_decodes, mixed.num_decodes = 1, batch - 1
            if prefill:
                from vllm.v1.attention.backends.utils import (
                    compute_causal_conv1d_metadata,
                )

                mixed.num_prefills, mixed.num_decodes = 1, batch - 2
                mixed.has_initial_state = torch.ones(
                    batch - 1, dtype=torch.bool, device="cuda"
                )
                mixed.has_initial_state[-1] = False
            mixed.num_actual_tokens = source_ids.numel()
            mixed.spec_token_indx = inverse[:8].contiguous()
            mixed.non_spec_token_indx = inverse[8:].contiguous()
            mixed.spec_state_indices_tensor = indices[:1]
            mixed.non_spec_state_indices_tensor = indices[1:, 0]
            mixed.spec_query_start_loc = cu[:2]
            mixed.non_spec_query_start_loc = torch.arange(
                batch, device="cuda", dtype=torch.int32
            )
            if prefill:
                mixed.non_spec_query_start_loc[-1] = batch - 2 + prefill
                mixed.nums_dict, mixed.batch_ptr, mixed.token_chunk_offset_ptr = (
                    compute_causal_conv1d_metadata(
                        mixed.non_spec_query_start_loc.cpu(), device=raw.device
                    )
                )
            mixed.num_accepted_tokens = metadata.num_accepted_tokens[:1]
            data = tuple(
                value.index_select(axis, source_ids)
                for value, axis in zip(inputs, (0, 1, 0, 1, 1))
            )
            captured_layer, captured_conv, captured_state = fixture(True)
            captured_out = torch.empty(
                (1, source_ids.numel(), h, d), dtype=torch.bfloat16, device="cuda"
            )
            captured_layer._forward_step(mixed, *data, captured_out)
            captured_conv.copy_(initial_conv)
            captured_state.copy_(initial_state)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured_layer._forward_step(mixed, *data, captured_out)
            for accepted in range(1, 9):
                mixed.num_accepted_tokens.fill_(accepted)
                if prefill:
                    mixed.has_initial_state[-1] = accepted % 2 == 0
                data[0].add_(0.01)
                indices.copy_(indices.roll(1, 0))
                old, old_conv, old_state = fixture(False)
                native, native_conv, native_state = fixture(True)
                expected = torch.zeros_like(captured_out)
                actual = torch.empty_like(expected)
                old._forward_step(mixed, *data, expected)
                native._forward_step(mixed, *data, actual)
                captured_conv.copy_(initial_conv)
                captured_state.copy_(initial_state)
                graph.replay()
                result["cases"].append(
                    dict(
                        kind="mixed_sd_prefill" if prefill else "mixed_sd_ar",
                        accepted=accepted,
                        output_vs_existing=error(actual, expected),
                        state_vs_existing=error(native_state, old_state),
                        conv_exact={"pass": bool(torch.equal(native_conv, old_conv))},
                        graph_output_exact=error(captured_out, actual, 0, 0, 0),
                        graph_state_exact=error(captured_state, native_state, 0, 0, 0),
                        graph_conv_exact={
                            "pass": bool(torch.equal(captured_conv, native_conv))
                        },
                    )
                )
            if getattr(args, "decode_kernel_witness", False):
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    for _ in range(32):
                        captured_layer._forward_step(mixed, *data, captured_out)
                    torch.accelerator.synchronize()
                names = sorted(
                    {
                        event.name
                        for event in prof.events()
                        if event.device_type == torch.autograd.DeviceType.CUDA
                    }
                )
                if not names:
                    raise RuntimeError("No mixed GPU kernel witness recorded")
                result["mixed_kernel_witness"] = dict(
                    scope="direct diagnostic; no latency claim",
                    kernels=names,
                    prefill_tokens=prefill,
                    backend=args.decode_backend,
                )
        if getattr(args, "decode_timing_replays", 0) or getattr(
            args, "decode_kernel_witness", False
        ):
            if not spec:
                result["ar_diagnostics"] = []
            for backend in dict.fromkeys(("flydsl", args.decode_backend)):
                bench_layer, _, _ = fixture(True, backend=backend)
                bench_out = torch.empty_like(actual)

                def launch(
                    layer=bench_layer, output=bench_out, meta=metadata, data=inputs
                ):
                    assert rocm_aiter_decode.try_forward(layer, meta, data, output)

                # All updates below belong to disposable fixture pools.
                launch()
                torch.accelerator.synchronize()
                row = {
                    "backend": backend,
                    "kind": "sd" if spec else "ar",
                    "length": length,
                }
                if getattr(args, "decode_kernel_witness", False):
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as prof:
                        for _ in range(32):
                            launch()
                        torch.accelerator.synchronize()
                    row["gpu_kernel_names"] = sorted(
                        {
                            event.name
                            for event in prof.events()
                            if event.device_type == torch.autograd.DeviceType.CUDA
                        }
                    )
                    if not row["gpu_kernel_names"]:
                        raise RuntimeError("No GPU kernel witness recorded")
                replays = getattr(args, "decode_timing_replays", 0)
                if replays:
                    if replays < 1 or replays > 10000:
                        raise ValueError("Use 1..10000 decode timing replays")
                    calls = 32
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(calls):
                            launch()
                    for _ in range(20):
                        graph.replay()
                    torch.accelerator.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(replays):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    row.update(
                        microseconds_per_call=start.elapsed_time(end)
                        * 1000
                        / (calls * replays),
                        replays=replays,
                        calls_per_graph=calls,
                        warmup_replays=20,
                        scope="decode conv/recurrence/norm; excludes ratio/projections",
                    )
                result.setdefault("decode_diagnostics", []).append(row)
                if not spec:
                    result["ar_diagnostics"].append(row)
