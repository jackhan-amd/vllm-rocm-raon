# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Release steering: mixed token alpha, actual decoder boundary and HIP replay."""

from types import SimpleNamespace

import numpy as np
import torch

from .common import error


def run(args, result):
    from vllm.models.raon1_mini import effort_steering as s
    from vllm.models.raon1_mini.model import RaonDecoderLayer

    s.configure(
        SimpleNamespace(
            reasoning_effort_steering={
                "alpha": {
                    "none": 0,
                    "low": 0.3,
                    "medium": 0.1,
                    "high": 0.05,
                    "xhigh": 0.02,
                    "max": 0,
                },
                "default": "high",
                "layers": [12, 14, 16, 18, 20],
            }
        ),
        "release-component-test",
    )
    s.G = torch.zeros(16384, device="cuda", dtype=torch.float32)
    runner = SimpleNamespace()
    torch.manual_seed(args.seed)
    result["scope"] = __doc__
    result["full_model_serving"] = "NOT_COVERED"
    # Real BF16 MoE-output boundary with FP32 checkpoint steering vectors.
    for tokens in [1, 8, 128, 1024]:
        h = torch.randn(tokens, 2560, device="cuda", dtype=torch.bfloat16)
        c = torch.randn(2560, device="cuda", dtype=torch.float32)

        class MoE(torch.nn.Module):
            def forward(self, x):
                return x

        moe = MoE()
        moe.raon_effort_c = c
        fixture = SimpleNamespace(mlp=moe)

        def boundary(x, fixture=fixture):
            return RaonDecoderLayer._mlp_with_effort(fixture, x)

        # A single compiled elementwise graph may fuse this into its consumer.
        compiled = torch.compile(boundary, fullgraph=True)
        compiled(h)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = compiled(h)
        case = {"tokens": tokens}
        for index, alpha in enumerate([0.05, 0.3, 0, 0.02]):
            n0 = tokens // 2
            qsl = torch.tensor([0, n0, tokens], device="cuda", dtype=torch.int32)
            s.fill(
                runner, np.array([alpha, 0.1 if alpha else 0], np.float32), qsl, tokens
            )
            # Independent host expansion from request lengths; no production fill.
            host_alpha = torch.tensor(
                [alpha] * n0 + ([0.1] if alpha else [0]) * (tokens - n0),
                dtype=torch.float32,
                device="cuda",
            )
            expected = h.clone()
            mask = host_alpha != 0
            expected[mask] = (h[mask].float() + host_alpha[mask, None] * c).bfloat16()
            eager = boundary(h)
            graph.replay()
            torch.accelerator.synchronize()
            case[f"eager_gold_{index}"] = error(eager, expected, 0, 0, 0)
            # Compiler fusion may round an FP32 multiply/add differently.
            case[f"graph_gold_{index}"] = error(captured, expected)
            case[f"alpha_mapping_{index}"] = error(s.G[:tokens], host_alpha, 0, 0, 0)
            if not alpha:
                case[f"zero_bits_{index}"] = {
                    "pass": torch.equal(captured.view(torch.int16), h.view(torch.int16))
                }
        result["cases"].append(case)
