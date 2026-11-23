# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 SD verifier versus independent Torch math with identical random draws.

Only the random-number helpers are shared. Torch independently computes target
and proposal probabilities, accepted prefixes, residuals, and sampled tokens.
The oracle uses FP64 arithmetic; production retains its own FP32 arithmetic.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import (
    _log1p_neg_stable,
    murmur3_uniform32,
    murmur3_uniform64,
    tl_rand32,
)

from .common import provenance, write_json


@triton.jit
def _draws(
    U,
    G,
    Seeds,
    Map,
    Pos,
    W: tl.constexpr,
    V: tl.constexpr,
    FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, block = tl.program_id(0), tl.program_id(1)
    req = tl.load(Map + row // W)
    seed, pos = tl.load(Seeds + req), tl.load(Pos + row)
    if block == 0:
        tl.store(U + row, tl_rand32(seed, pos, includes_zero=False))
    token = block * BLOCK + tl.arange(0, BLOCK)
    if FP64:
        u = murmur3_uniform64(seed, pos, token)
        noise = -tl.log(-tl.log(u))
    else:
        u = murmur3_uniform32(seed, pos, token)
        noise = -tl.log(-_log1p_neg_stable(u))
    tl.store(G + row * V + token, noise, mask=token < V)


def oracle(target, draft, proposals, lengths, u, noise, greedy, block):
    """Direct probability-space standard/block rejection and Gumbel-max draw."""
    n, width, vocab = target.shape
    k = width - 1
    p = target.double().softmax(-1)
    q = draft.double().softmax(-1) if draft is not None else None
    arange = torch.arange(n, device=target.device)
    selected_p = p[:, :k].gather(2, proposals.clamp_min(0)[..., None]).squeeze(-1)
    selected_q = (
        q.gather(2, proposals.clamp_min(0)[..., None]).squeeze(-1)
        if q is not None
        else torch.ones_like(selected_p)
    )
    # Running joint ratios for block verification, computed in probability space.
    joint, carry = [], torch.ones(n, device=target.device, dtype=torch.float64)
    for i in range(k):
        ratio = carry * selected_p[:, i] / selected_q[:, i]
        carry = torch.where(proposals[:, i] >= 0, ratio.clamp(max=1), carry)
        joint.append(carry)
    joint = torch.stack(joint, 1)
    accepted = torch.zeros(n, device=target.device, dtype=torch.long)
    alive = torch.ones(n, device=target.device, dtype=torch.bool)
    for i in range(k):
        alive &= (i < lengths) & (proposals[:, i] >= 0)
        if greedy:
            good = proposals[:, i] == target[:, i].argmax(-1)
        elif not block:
            good = u[:, i].double() * selected_q[:, i] < selected_p[:, i]
        else:
            ratio = joint[:, i]
            if i + 1 < k:
                if q is not None:
                    mass = (
                        (ratio[:, None] * p[:, i + 1] - q[:, i + 1])
                        .clamp_min(0)
                        .sum(-1)
                    )
                else:
                    mass = ratio * (1 - selected_p[:, i + 1])
                denom = mass + 1 - ratio
                h = torch.where(denom > 0, mass / denom, 1.0)
                next_valid = (i + 1 < lengths) & (proposals[:, i + 1] >= 0)
                h = torch.where(next_valid, h, ratio)
            else:
                h = ratio
            good = u[:, i].double() <= h
        if block and not greedy:
            accepted = torch.where(alive & good, i + 1, accepted)
        else:
            alive &= good
            accepted += alive
    tau = accepted
    bonus = tau == lengths
    proposed_at_tau = proposals[arange, tau.clamp(max=k - 1)]
    placeholder = proposed_at_tau < 0
    residual = p[arange, tau].clone()
    if q is not None:
        multiplier = (
            joint[arange, (tau - 1).clamp_min(0)]
            if block and not greedy
            else torch.ones_like(tau).double()
        )
        multiplier = torch.where(tau > 0, multiplier, 1.0)
        adjusted = (
            multiplier[:, None] * residual - q[arange, tau.clamp(max=k - 1)]
        ).clamp_min(0)
    else:
        adjusted = residual.clone()
        adjusted.scatter_(1, proposed_at_tau.clamp_min(0)[:, None], 0.0)
    residual = torch.where((bonus | placeholder)[:, None], residual, adjusted)
    if greedy:
        sampled = target[arange, tau].argmax(-1)
    else:
        sampled = (residual.log() + noise[arange, tau].double()).argmax(-1)
    output = torch.full((n, width), -1, device=target.device, dtype=torch.long)
    output[:, :k] = proposals
    output[arange, tau] = sampled
    return output, tau + 1


def run(args, result):
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

    result["source"] = provenance(rejection_sample)
    result["scope"] = (
        "V2 standard/block verifier: exact accepted count and emitted token prefix "
        "versus Torch FP64 probability-space gold. Random draws are extracted "
        "using production RNG helpers; RNG correctness is not independently tested. "
        "No watermarking, penalties, draft-model generation or scheduler history."
    )
    for vocab, n in [(33, 256), (8209, 16), (259840, 8)]:
        for k in args.draft_counts:
            width, cap = k + 1, n + 7
            mapping = torch.randperm(cap, device="cuda")[:n].int()
            seeds = (
                torch.arange(cap, device="cuda", dtype=torch.int64) * 1109 + args.seed
            )
            full_pos = (
                torch.arange(width, device="cuda", dtype=torch.int64)[None, :]
                + torch.arange(n, device="cuda", dtype=torch.int64)[:, None] * 17
                + 100
            )
            for fp64 in [False, True]:
                uniforms = torch.empty(n, width, device="cuda")
                noise = torch.empty(
                    n,
                    width,
                    vocab,
                    device="cuda",
                    dtype=torch.float64 if fp64 else torch.float32,
                )
                _draws[(n * width, triton.cdiv(vocab, 1024))](
                    uniforms, noise, seeds, mapping, full_pos, width, vocab, fp64, 1024
                )
                for regime in [
                    "random",
                    "equal",
                    "forced_boundaries",
                    "placeholders",
                    "one_hot",
                    "sparse16",
                ]:
                    target_raw = torch.randn(n, width, vocab, device="cuda")
                    draft_raw = torch.randn(n, k, vocab, device="cuda")
                    if regime == "sparse16":
                        # DFlash2 keeps only a small candidate support in its
                        # full-vocabulary proposal cache. Spread it across blocks.
                        target_raw.mul_(4)
                        top = target_raw[:, :k].topk(16, dim=-1)
                        draft_raw.fill_(-torch.inf)
                        draft_raw.scatter_(
                            2, top.indices, top.values + torch.randn_like(top.values)
                        )
                    lengths = torch.full((n,), k, device="cuda", dtype=torch.int32)
                    if regime == "equal":
                        draft_raw.copy_(target_raw[:, :k])
                    if regime in ["random", "placeholders", "one_hot"]:
                        lengths = torch.arange(n, device="cuda", dtype=torch.int32) % (
                            k + 1
                        )
                    for temperature in [0.0, 0.7, 1.3]:
                        t = temperature or 1.0
                        target, draft = target_raw / t, draft_raw / t
                        proposals = torch.multinomial(
                            draft.softmax(-1).flatten(0, 1), 1
                        ).reshape(n, k)
                        if regime == "forced_boundaries":
                            # Every rejection location, including all accepted.
                            target.fill_(-torch.inf)
                            target[:, :, 0] = 0.0
                            draft.fill_(-torch.inf)
                            draft[:, :, 0] = 0.0
                            proposals.fill_(0)
                            stop = torch.arange(n, device="cuda") % (k + 1)
                            valid = stop < k
                            draft[
                                torch.arange(n, device="cuda")[valid], stop[valid], 0
                            ] = -torch.inf
                            draft[
                                torch.arange(n, device="cuda")[valid], stop[valid], 1
                            ] = 0.0
                            proposals[
                                torch.arange(n, device="cuda")[valid], stop[valid]
                            ] = 1
                        if regime == "placeholders":
                            proposals[
                                torch.arange(n, device="cuda"),
                                torch.arange(n, device="cuda") % k,
                            ] = -1
                        has_q = regime != "one_hot"
                        # Raw draft logits are pre-temperature; target is processed.
                        global_draft = torch.full(
                            (cap, k, vocab), float("nan"), device="cuda"
                        )
                        global_draft[mapping.long()] = draft * t
                        temps = torch.full((cap,), float("nan"), device="cuda")
                        temps[mapping.long()] = temperature
                        counts = lengths + 1
                        mask = (
                            torch.arange(width, device="cuda")[None, :]
                            < counts[:, None]
                        )
                        state_map = mapping.repeat_interleave(counts.long())
                        local = (
                            torch.arange(width, device="cuda").expand(n, -1)[mask].int()
                        )
                        padded = torch.cat(
                            [
                                torch.zeros(n, 1, device="cuda", dtype=torch.long),
                                proposals,
                            ],
                            1,
                        )
                        for block in [False, True]:
                            actual, actual_counts = rejection_sample(
                                target[mask].contiguous(),
                                global_draft if has_q else None,
                                padded[mask].contiguous(),
                                torch.cat(
                                    [
                                        torch.zeros(
                                            1, device="cuda", dtype=torch.int32
                                        ),
                                        counts.cumsum(0).int(),
                                    ]
                                ),
                                full_pos[mask].contiguous(),
                                mapping,
                                state_map,
                                local,
                                temps,
                                seeds,
                                k,
                                use_fp64=fp64,
                                use_block_verification=block,
                            )
                            expected, expected_counts = oracle(
                                target,
                                (
                                    global_draft[mapping.long()].double()
                                    / (
                                        temps[mapping.long()].double()
                                        if temperature
                                        else torch.ones(
                                            n, device="cuda", dtype=torch.float64
                                        )
                                    )[:, None, None]
                                    if has_q
                                    else None
                                ),
                                proposals,
                                lengths,
                                uniforms,
                                noise,
                                temperature == 0,
                                block,
                            )
                            live = (
                                torch.arange(width, device="cuda")[None, :]
                                < expected_counts[:, None]
                            )
                            count_bad = actual_counts != expected_counts
                            token_bad = ((actual != expected) & live).any(-1)
                            good = not bool(count_bad.any() | token_bad.any())
                            case = {
                                "vocab": vocab,
                                "requests": n,
                                "draft_tokens": k,
                                "regime": regime,
                                "temperature": temperature,
                                "fp64_gumbel": fp64,
                                "block": block,
                                "exact_gold": {
                                    "pass": good,
                                    "count_mismatches": int(count_bad.sum()),
                                    "token_mismatches": int(token_bad.sum()),
                                },
                            }
                            result["cases"].append(case)
                            if not good:
                                torch.save(
                                    {
                                        "case": case,
                                        "target": target.cpu(),
                                        "draft": draft.cpu(),
                                        "proposals": proposals.cpu(),
                                        "lengths": lengths.cpu(),
                                        "uniforms": uniforms.cpu(),
                                        "noise": noise.cpu(),
                                        "actual": actual.cpu(),
                                        "counts": actual_counts.cpu(),
                                        "gold": expected.cpu(),
                                        "gold_counts": expected_counts.cpu(),
                                    },
                                    args.output + ".failure.pt",
                                )
                                write_json(args.output, result)
                                return
                        write_json(args.output, result)
