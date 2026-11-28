# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native SD padded-state gold, gap preservation and changed-input graph replay.

Standalone correctness tool; no endpoint, model download or performance test.
Uses the independent Torch equations from the existing RDA validation suite.
"""

import argparse

import torch
import torch.nn.functional as F

from .common import environment, error, write_json
from .reference import convolution, rda

H, D = 40, 128
P = H * D
STATE_SIZE = H * D * D
SENTINEL = 12345.0


def run_case(module, batch, length, padding, result):
    tokens, slots = batch * length, batch * length + 5

    def random(*shape):
        return (torch.randn(*shape, device="cuda") * 0.2).bfloat16()

    initial = torch.randn(slots, H, D, D, device="cuda") * 0.01
    initial_conv = random(slots, 3 * P, length + 2)
    backing = torch.full((slots, STATE_SIZE + padding), SENTINEL, device="cuda")
    state = backing[:, :STATE_SIZE].view(slots, H, D, D)
    dense, conv, dense_conv = (
        initial.clone(),
        initial_conv.clone(),
        initial_conv.clone(),
    )
    raw, g = random(tokens, 3 * P), random(1, tokens, H, D).contiguous()
    beta, delta = random(1, tokens, H), random(1, tokens, H)
    ratio = torch.empty(1, tokens, H, device="cuda")
    gate = random(tokens, H, D)
    weight = random(3 * P, 4)
    packed_weight = weight.view(3, P, 4).transpose(1, 2).float().contiguous()
    a_log, bias = (
        torch.randn(H, device="cuda") * 0.1,
        torch.randn(P, device="cuda") * 0.1,
    )
    norm = torch.ones(D, device="cuda")
    indices = torch.arange(1, tokens + 1, device="cuda").int().view(batch, length)
    cu = torch.arange(0, tokens + 1, length, device="cuda", dtype=torch.int32)
    accepted = torch.ones(batch, device="cuda", dtype=torch.int32)
    actual, expected = torch.empty_like(gate), torch.empty_like(gate)

    def update_ratio():
        ratio.copy_(
            torch.exp(
                F.logsigmoid(beta.float() + delta.float()) - F.logsigmoid(beta.float())
            )
        )

    def reset():
        state.copy_(initial)
        dense.copy_(initial)
        conv.copy_(initial_conv)
        dense_conv.copy_(initial_conv)

    def launch(cache, recurrent, output):
        module.forward_sd(
            raw,
            packed_weight,
            cache,
            g,
            beta,
            ratio,
            a_log,
            bias,
            indices,
            accepted,
            cu,
            recurrent,
            gate,
            norm,
            output,
            -5.0,
            1e-6,
        )

    def gold():
        output = torch.zeros_like(actual)
        recurrent, history = initial.clone(), initial_conv.clone()
        for n in range(batch):
            offset = int(accepted[n]) - 1
            first, read = int(indices[n, 0]), int(indices[n, offset])
            lo, hi = int(cu[n]), int(cu[n + 1])
            old = initial_conv[first : first + 1, :, offset : offset + 3]
            values, _ = convolution(raw[None, lo:hi], old, weight)
            q, k, v = [part.view(1, length, H, D) for part in values.chunk(3, -1)]
            value, states = rda(
                q,
                k,
                v,
                g[:, lo:hi],
                beta[:, lo:hi],
                delta[:, lo:hi],
                a_log,
                bias,
                initial[read : read + 1],
            )
            rounded = value.bfloat16().float()
            output[lo:hi] = (
                rounded
                * torch.rsqrt(rounded.square().mean(-1, keepdim=True) + 1e-6)
                * gate[lo:hi].float().sigmoid()
            ).bfloat16()[0]
            for token in range(length):
                recurrent[indices[n, token]] = states[0, token]
            history[first, :, :2] = old[0, :, 1:]
            history[first, :, 2 : 2 + length] = raw[lo:hi].T
        return output, recurrent, history

    def check(kind):
        target, gold_state, gold_conv = gold()
        written = indices.flatten().long()
        untouched = torch.ones(slots, device="cuda", dtype=torch.bool)
        untouched[written] = False
        checks = {
            "dense_output_exact": bool(torch.equal(actual, expected)),
            "dense_state_exact": bool(torch.equal(state, dense)),
            "dense_conv_exact": bool(torch.equal(conv, dense_conv)),
            "torch_output": error(actual, target),
            "torch_state": error(state[written], gold_state[written]),
            "untouched_state_exact": bool(
                torch.equal(state[untouched], initial[untouched])
            ),
            "torch_conv_exact": bool(torch.equal(conv, gold_conv)),
            "gap_sentinel_exact": bool(torch.all(backing[:, STATE_SIZE:] == SENTINEL)),
        }
        result["cases"].append(
            {
                "kind": kind,
                "native_backend": result["active_backend"],
                "batch": batch,
                "length": length,
                "accepted": accepted.tolist(),
                "padding_elements": padding,
                "state_stride": list(state.stride()),
                "checks": checks,
            }
        )
        assert all(v["pass"] if isinstance(v, dict) else v for v in checks.values()), (
            checks
        )

    update_ratio()
    for count in range(1, length + 1):
        accepted.fill_(count)
        reset()
        launch(dense_conv, dense, expected)
        launch(conv, state, actual)
        check("direct")

    # Capture against disposable/reset fixture state; never warm a live cache.
    reset()
    launch(conv, state, actual)
    torch.cuda.synchronize()
    reset()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(conv, state, actual)
    for replay in range(2):
        raw.add_(0.01)
        g.add_(0.015)
        beta.add_(0.02)
        delta.sub_(0.01)
        indices.copy_(indices.roll(1, 1))
        accepted.copy_(
            1
            + (torch.arange(batch, device="cuda", dtype=torch.int32) + replay) % length
        )
        update_ratio()
        reset()
        launch(dense_conv, dense, expected)
        graph.replay()
        check("changed_input_graph")

    # Guard rejection must occur before any cache mutation.
    malformed = {
        "dtype": state.bfloat16(),
        "shape": state[:, :-1],
        "stride": state.transpose(2, 3),
        "alignment_stride": torch.zeros(
            slots * (STATE_SIZE + 1), device="cuda"
        ).as_strided((slots, H, D, D), (STATE_SIZE + 1, D * D, D, 1)),
        "alignment_base": torch.zeros(slots * STATE_SIZE + 1, device="cuda")[1:].view(
            slots, H, D, D
        ),
    }
    before = conv.clone()
    for label, invalid in malformed.items():
        before_state = invalid.clone()
        try:
            launch(conv, invalid, actual)
        except RuntimeError as exc:
            expected_message = "alignment" if label.startswith("alignment") else label
            assert f"Invalid SD state {expected_message}" in str(exc), str(exc)
        else:
            raise AssertionError(f"Guard accepted invalid state {label}")
        assert torch.equal(conv, before), label
        assert torch.equal(invalid, before_state), label
    result["guard_cases"].append(
        {
            "batch": batch,
            "native_backend": result["active_backend"],
            "length": length,
            "padding_elements": padding,
            "rejected": list(malformed),
            "pass": True,
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--lengths", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--padding-elements", type=int, nargs="+", default=[64, 81920])
    parser.add_argument("--seed", type=int, default=1519)
    args = parser.parse_args()
    if (
        any(b < 1 for b in args.batch_sizes)
        or any(t not in (2, 4, 8) for t in args.lengths)
        or any(p < 4 or p % 4 for p in args.padding_elements)
    ):
        parser.error(
            "Use positive batches, lengths 2/4/8, "
            "and positive padding divisible by four"
        )
    from vllm.models.raon1_mini.rocm_hip_rda import load

    result = {
        "environment": environment(),
        "cases": [],
        "guard_cases": [],
        "pass": False,
    }
    try:
        for kind in ("sd", "sd_fused", "sd_vsplit"):
            result["active_backend"] = kind
            module = load(kind)
            for batch in args.batch_sizes:
                for length in args.lengths:
                    for padding in args.padding_elements:
                        torch.manual_seed(args.seed)
                        run_case(module, batch, length, padding, result)
        result["pass"] = True
    finally:
        write_json(args.output, result)


if __name__ == "__main__":
    main()
