# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native route packing versus independent CPU mapping and Torch BF16 bits."""

from collections import Counter

import torch


def _check(ids, weights, experts, unit, packed):
    sorted_ids, sorted_weights, sorted_experts, metadata = packed
    original = ids.cpu().tolist()
    rounded = weights.bfloat16().float().view(torch.int32).cpu().tolist()
    payload = sorted_ids.cpu().tolist()
    weight_bits = sorted_weights.view(torch.int32).cpu().tolist()
    blocks = sorted_experts.cpu().tolist()
    valid = metadata.cpu().tolist()
    tokens, topk = ids.shape
    counts = Counter(v for row in original for v in row if 0 <= v < experts)
    expected_experts = [
        expert
        for expert, count in sorted(counts.items())
        for _ in range((count + unit - 1) // unit)
    ]
    assert valid == [len(expected_experts) * unit, tokens]
    assert blocks[: len(expected_experts)] == expected_experts
    assert all(v == 0 for v in blocks[len(expected_experts) :])
    seen = set()
    for row in range(valid[0]):
        token, slot = payload[row] & 0xFFFFFF, (payload[row] >> 24) & 255
        if token < tokens and slot < topk:
            assert (token, slot) not in seen
            seen.add((token, slot))
            assert original[token][slot] == blocks[row // unit]
            assert weight_bits[row] == rounded[token][slot]
        else:
            assert token == tokens and slot == 255 and weight_bits[row] == 0
    assert len(seen) == sum(counts.values())
    for row in range(valid[0], len(payload)):
        assert payload[row] == -16777216 + tokens and weight_bits[row] == 0
    return {"pass": True, "valid_rows": valid[0], "real_routes": len(seen)}


def run(args, result):
    from vllm.models.raon1_mini.rocm_flydsl_route_prepare import prepare

    result["scope"] = __doc__
    result["oracle"] = (
        "CPU expert counts and route-slot mapping; exact Torch GPU BF16-to-FP32 "
        "bit patterns, including ties, signed zero and subnormals. No TPS claim."
    )
    torch.manual_seed(args.seed)
    for experts in (8, 128):
        for tokens in (1, 8, 17, 32):
            for unit in (16, 32):
                for dtype in (torch.int32, torch.int64):
                    ids = (
                        torch.rand((tokens, experts), device="cuda")
                        .topk(6, -1)
                        .indices.to(dtype)
                    )
                    weights = torch.rand((tokens, 6), device="cuda")
                    weights /= weights.sum(-1, keepdim=True)
                    direct = prepare(ids, weights, experts, unit)
                    case = dict(
                        experts=experts,
                        tokens=tokens,
                        unit=unit,
                        dtype=str(dtype),
                        initial=_check(ids, weights, experts, unit, direct),
                    )
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = prepare(ids, weights, experts, unit)
                    for pattern in (
                        "six_shared",
                        "one_shared",
                        "rotated",
                        "bf16_ties",
                        "invalid_ids",
                        "all_invalid",
                    ):
                        if pattern == "six_shared":
                            ids.copy_(torch.arange(6, device="cuda").expand(tokens, 6))
                        elif pattern == "one_shared":
                            ids.zero_()
                        elif pattern == "rotated":
                            ids.copy_(
                                (
                                    torch.arange(tokens * 6, device="cuda").view(
                                        tokens, 6
                                    )
                                    + 3
                                )
                                % experts
                            )
                        elif pattern == "bf16_ties":
                            bits = torch.tensor(
                                [
                                    0x3F008000,
                                    0x3F018000,
                                    0x00008000,
                                    0x00018000,
                                    -2147385344,
                                    0,
                                ],
                                device="cuda",
                                dtype=torch.int32,
                            )
                            weights.view(torch.int32).copy_(bits.expand(tokens, 6))
                        elif pattern == "invalid_ids":
                            ids[:, 0] = -1
                            ids[:, 1] = experts
                            ids[:, 2] = (
                                (1 << 32) + 3 if dtype == torch.int64 else 2**31 - 1
                            )
                        else:
                            ids.fill_(-1)
                        weights.copy_(weights.roll(1, -1))
                        graph.replay()
                        case[pattern] = _check(ids, weights, experts, unit, captured)
                    result["cases"].append(case)
