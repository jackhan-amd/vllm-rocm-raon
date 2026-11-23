# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core arithmetic and forced-acceptance/continuous-batch state diagnostics."""

import torch

from .common import error, provenance, write_json
from .reference import convolution, rda


def run(args, result):
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.models.kimi_k3.amd.ops.kda_prefill import chunk_kda_prefill
    from vllm.models.kimi_k3.amd.ops.third_party.kda import fused_recurrent_kda
    from vllm.models.raon1_mini.kda import compute_rda_value_ratio

    result["source"] = provenance(fused_recurrent_kda)
    h, d = args.heads, args.dim
    torch.manual_seed(args.seed)

    def random(*shape, scale=0.3):
        return (torch.randn(*shape, device="cuda") * scale).bfloat16()

    a = torch.randn(h, device="cuda") * 0.1
    bias = torch.randn(h * d, device="cuda") * 0.1
    cases = result["cases"]
    if args.command == "core":
        for length in [1, 8, 65, 257]:
            q, k, v = [random(1, length, h, d) for _ in range(3)]
            g = random(1, length, h, d)
            beta, delta = random(1, length, h), random(1, length, h)
            initial = torch.randn(1, h, d, d, device="cuda") * 0.01
            expected, states = rda(q, k, v, g, beta, delta, a, bias, initial)
            ratio = compute_rda_value_ratio(beta, delta)
            folded = (v.float() * ratio[..., None]).bfloat16()
            cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
            for backend in ["recurrent", "prefill"]:
                if backend == "recurrent":
                    pool = torch.zeros(length + 1, h, d, d, device="cuda")
                    pool[1] = initial[0]
                    indices = torch.arange(1, length + 1, device="cuda").int()[None]
                    actual, _ = fused_recurrent_kda(
                        q,
                        k,
                        folded,
                        g,
                        beta,
                        a,
                        bias,
                        -5.0,
                        pool,
                        cu,
                        indices,
                        torch.ones(1, device="cuda", dtype=torch.int32),
                    )
                    actual_state = pool[-1:]
                else:
                    actual, actual_state = chunk_kda_prefill(
                        q=q,
                        k=k,
                        v=folded,
                        raw_g=g,
                        raw_beta=beta,
                        A_log=a,
                        g_bias=bias,
                        lower_bound=-5.0,
                        initial_state=initial.clone(),
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                        cu_seqlens=cu,
                        use_fused_chunk=True,
                    )
                cases.append(
                    {
                        "backend": backend,
                        "tokens": length,
                        "output": error(actual, expected),
                        "state": error(actual_state, states[:, -1]),
                    }
                )
                write_json(args.output, result)
        return

    for draft in args.draft_counts:
        t, n, c = draft + 1, 16, 3 * h * d
        slots = (torch.randperm(n * t, device="cuda") + 1).reshape(n, t).int()
        pool = torch.zeros(n * t + 1, h, d, d, device="cuda")
        conv = torch.zeros(n * t + 1, c, t + 2, device="cuda", dtype=torch.bfloat16)
        weight = random(c, 4, scale=0.2)
        previous_state, previous_history = {}, {}
        rounds = args.rounds if args.command == "stress" else max(t + 2, 10)
        for round_id in range(rounds):
            order = torch.randperm(n).tolist()
            active = order[: [16, 15, 8, 1][round_id % 4]]
            # Accepted count includes the mandatory target token: 1..draft+1.
            accepted = [(round_id + seq) % t + 1 for seq in active]
            starts, hist = [], []
            for seq, count in zip(active, accepted):
                if seq not in previous_state:
                    starts.append(torch.zeros(h, d, d, device="cuda"))
                    hist.append(torch.zeros(c, 3, device="cuda"))
                else:
                    starts.append(previous_state[seq][count - 1])
                    hist.append(previous_history[seq][:, count : count + 3])
            batch = len(active)
            raw = random(batch, t, c)
            g = random(batch, t, h, d)
            beta, delta = random(batch, t, h), random(batch, t, h)
            expected_conv, joined = convolution(raw, torch.stack(hist), weight)
            indices = slots[active].contiguous()
            counts = torch.tensor(accepted, device="cuda", dtype=torch.int32)
            cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * t
            conv_out = torch.empty_like(raw.reshape(batch * t, c))
            untouched_mask = torch.ones(pool.shape[0], device="cuda", dtype=torch.bool)
            untouched_mask[indices.flatten().long()] = False
            before = pool[untouched_mask].clone()
            conv_mask = torch.ones(pool.shape[0], device="cuda", dtype=torch.bool)
            conv_mask[indices[:, 0].long()] = False
            conv_before = conv[conv_mask].clone()
            causal_conv1d_update(
                raw.reshape(batch * t, c),
                conv,
                weight,
                None,
                activation="silu",
                conv_state_indices=indices[:, 0].contiguous(),
                num_accepted_tokens=counts,
                query_start_loc=cu,
                max_query_len=t,
                validate_data=False,
                out=conv_out,
            )
            qkv = conv_out.clone()
            ratio = compute_rda_value_ratio(beta, delta)
            value = qkv[:, 2 * h * d :].view(batch, t, h, d)
            value.copy_((value.float() * ratio[..., None]).bfloat16())
            q, k, v = [x.view(1, batch * t, h, d) for x in qkv.split(h * d, -1)]
            actual, _ = fused_recurrent_kda(
                q,
                k,
                v,
                g.reshape(1, batch * t, h, d),
                beta.reshape(1, batch * t, h),
                a,
                bias,
                -5.0,
                pool,
                cu,
                indices,
                counts,
            )
            eq, ek, ev = [
                x.reshape(batch, t, h, d) for x in expected_conv.split(h * d, -1)
            ]
            expected, states = rda(
                eq, ek, ev, g, beta, delta, a, bias, torch.stack(starts)
            )
            for i, seq in enumerate(active):
                previous_state[seq] = states[i].clone()
                previous_history[seq] = joined[i].clone()
            record = {
                "draft_tokens": draft,
                "round": round_id,
                "batch": batch,
                "order": active,
                "accepted": accepted,
                "conv": error(conv_out, expected_conv.reshape(batch * t, c)),
                "output": error(actual.reshape_as(expected), expected),
                "state": error(pool[indices.long()], states),
                "unrelated_rows": {
                    "pass": torch.equal(pool[untouched_mask], before)
                    and torch.equal(conv[conv_mask], conv_before)
                },
            }
            cases.append(record)
            write_json(args.output, result)
            if not all(
                value["pass"]
                for value in record.values()
                if isinstance(value, dict) and "pass" in value
            ):
                torch.save(
                    {
                        "raw": raw,
                        "g": g,
                        "beta": beta,
                        "delta": delta,
                        "initial": torch.stack(starts),
                        "actual": actual,
                        "expected": expected,
                        "state": pool[indices.long()],
                        "expected_state": states,
                        "indices": indices,
                    },
                    str(args.output) + ".failure.pt",
                )
                return
