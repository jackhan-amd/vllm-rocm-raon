# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the installed GPU rejection sampler, not a replacement sampler."""

from types import SimpleNamespace

import torch

from .common import provenance, write_json


def run(args, result):
    from vllm.v1.sample.rejection_sampler import rejection_sample

    result["source"] = provenance(rejection_sample)

    # The low-level kernel consumes these four sampling metadata fields.
    def metadata(greedy, batch):
        return SimpleNamespace(
            all_greedy=greedy,
            all_random=not greedy,
            temperature=torch.ones(batch, device="cuda"),
            generators={},
        )

    for length in args.draft_counts:
        proposals, targets, expected, lengths = [], [], [], []
        for rejected in range(length + 1):
            target = list(range(2, length + 2))
            draft = target.copy()
            if rejected < length:
                draft[rejected] = 20
                want = target[: rejected + 1]
            else:
                want = target + [31]
            proposals.extend(draft)
            targets.extend(target)
            lengths.append(length)
            expected.append(want + [-1] * (length + 1 - len(want)))
        logits = torch.full((len(targets), 32), -100.0, device="cuda")
        logits[torch.arange(len(targets), device="cuda"), targets] = 100.0
        batch = len(lengths)
        actual = rejection_sample(
            torch.tensor(proposals, device="cuda", dtype=torch.int32),
            lengths,
            length,
            torch.tensor(lengths, device="cuda", dtype=torch.int32).cumsum(0).int(),
            None,
            logits,
            torch.full((batch, 1), 31, device="cuda", dtype=torch.int32),
            metadata(True, batch),
        )
        result["cases"].append(
            {
                "kind": "all_rejection_positions_and_bonus",
                "draft_tokens": length,
                "exact": {
                    "pass": torch.equal(
                        actual, torch.tensor(expected, device="cuda", dtype=torch.int32)
                    )
                },
            }
        )

    distributions = [
        ([0.55, 0.25, 0.15, 0.05, 0.0], [0.05, 0.15, 0.35, 0.45, 0.0]),
        ([0.1, 0.2, 0.3, 0.4, 0.0], [0.1, 0.2, 0.3, 0.4, 0.0]),
        ([1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]),
    ]
    n = args.samples
    for di, (p_list, q_list) in enumerate(distributions):
        p, q = [torch.tensor(x, device="cuda") for x in [p_list, q_list]]
        for length in args.draft_counts:
            draft = torch.multinomial(q, n * length, replacement=True).int()
            bonus = torch.multinomial(p, n, replacement=True).reshape(n, 1).int()
            actual = rejection_sample(
                draft,
                [length] * n,
                length,
                torch.arange(1, n + 1, device="cuda", dtype=torch.int32) * length,
                q.expand(n * length, -1).contiguous(),
                p.log().expand(n * length, -1).contiguous(),
                bonus,
                metadata(False, n),
            )
            first = actual[:, 0].long()
            assert bool(((first >= 0) & (first < 5)).all())
            second = actual[:, 1].long()
            # Complete early-rejected sequences with an independent AR draw.
            fallback = torch.multinomial(p, n, replacement=True)
            second = torch.where(second < 0, fallback, second)
            observed = torch.bincount(first * 5 + second, minlength=25).float() / n
            expected = (p[:, None] * p[None, :]).flatten()
            limits = 6 * torch.sqrt(expected * (1 - expected) / n) + 2 / n
            deviation = (observed - expected).abs()
            result["cases"].append(
                {
                    "kind": "iid_two_token_joint_distribution",
                    "distribution": di,
                    "draft_tokens": length,
                    "samples": n,
                    "distribution_check": {
                        "pass": bool((deviation <= limits).all()),
                        "max_abs": float(deviation.max()),
                        "criterion": "6 binomial standard errors +2/N",
                        "expected": expected.tolist(),
                        "observed": observed.tolist(),
                    },
                }
            )
            write_json(args.output, result)
    run_v2(args, result)


def run_v2(args, result):
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

    result["v2_source"] = provenance(rejection_sample)

    def invoke(target, draft_logits, proposals, temperature, block):
        n, width, vocab = target.shape
        length = width - 1
        mapping = torch.arange(n, device="cuda", dtype=torch.int32)
        sampled = torch.zeros(n, width, device="cuda", dtype=torch.int64)
        sampled[:, 1:] = proposals
        return rejection_sample(
            target_logits=target.reshape(n * width, vocab).contiguous(),
            draft_logits=draft_logits,
            draft_sampled=sampled.flatten(),
            cu_num_logits=torch.arange(n + 1, device="cuda", dtype=torch.int32) * width,
            pos=(torch.arange(width, device="cuda", dtype=torch.int64) + 100).repeat(n),
            idx_mapping=mapping,
            expanded_idx_mapping=mapping.repeat_interleave(width),
            expanded_local_pos=torch.arange(
                width, device="cuda", dtype=torch.int32
            ).repeat(n),
            temperature=torch.full(
                (n,), temperature, device="cuda", dtype=torch.float32
            ),
            seed=torch.arange(n, device="cuda", dtype=torch.int64) + 42,
            num_speculative_steps=length,
            use_block_verification=block,
        )

    for block in [False, True]:
        for length in args.draft_counts:
            width, n, vocab = length + 1, length + 1, 32
            desired = torch.arange(2, width + 2, device="cuda").expand(n, width)
            logits = torch.full((n, width, vocab), -100.0, device="cuda")
            logits.scatter_(2, desired[..., None], 100.0)
            proposals = desired[:, :-1].clone()
            expected = []
            for stop in range(width):
                if stop < length:
                    proposals[stop, stop] = 20
                expected.append(desired[stop, : stop + 1].tolist())
            sampled, counts = invoke(logits, None, proposals, 0.0, block)
            good = all(
                sampled[i, : int(counts[i])].tolist() == row
                for i, row in enumerate(expected)
            )
            result["cases"].append(
                {
                    "backend": "v2",
                    "block": block,
                    "kind": "every_greedy_rejection_position",
                    "draft_tokens": length,
                    "exact": {"pass": good},
                }
            )
        for di, (p_list, q_list) in enumerate(
            [
                ([0.55, 0.25, 0.15, 0.05, 0.0], [0.05, 0.15, 0.35, 0.45, 0.0]),
                ([0.1, 0.2, 0.3, 0.4, 0.0], [0.1, 0.2, 0.3, 0.4, 0.0]),
                ([1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]),
            ]
        ):
            p, q = [torch.tensor(x, device="cuda") for x in [p_list, q_list]]
            for length in args.draft_counts:
                n, width = args.samples, length + 1
                proposed = torch.multinomial(q, n * length, replacement=True).reshape(
                    n, length
                )
                sampled, counts = invoke(
                    p.log().expand(n, width, -1).contiguous(),
                    q.log().expand(n, length, -1).contiguous(),
                    proposed,
                    1.0,
                    block,
                )
                assert bool(((sampled[:, 0] >= 0) & (sampled[:, 0] < 5)).all())
                second = torch.where(
                    counts >= 2,
                    sampled[:, 1],
                    torch.multinomial(p, n, replacement=True),
                )
                observed = (
                    torch.bincount(sampled[:, 0] * 5 + second, minlength=25).float() / n
                )
                expected = (p[:, None] * p[None, :]).flatten()
                limit = 6 * torch.sqrt(expected * (1 - expected) / n) + 2 / n
                diff = (observed - expected).abs()
                result["cases"].append(
                    {
                        "backend": "v2",
                        "block": block,
                        "distribution": di,
                        "draft_tokens": length,
                        "samples": n,
                        "joint_distribution": {
                            "pass": bool((diff <= limit).all()),
                            "max_abs": float(diff.max()),
                            "expected": expected.tolist(),
                            "observed": observed.tolist(),
                        },
                    }
                )
                write_json(args.output, result)
