# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated warmed steering cost; not model TPS or request latency."""

import argparse
import time
from types import SimpleNamespace

import numpy as np
import torch

from .common import environment, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--implementation", choices=["plain", "compiled"], default="plain"
    )
    args = parser.parse_args()
    from vllm.models.raon1_mini import effort_steering as s

    result = {
        "scope": __doc__,
        "environment": environment(),
        "implementation": args.implementation,
        "cases": [],
    }
    s.G = torch.full((16384,), 0.05, device="cuda")
    compiled = (
        torch.compile(s.steer, fullgraph=True)
        if args.implementation == "compiled"
        else s.steer
    )
    for tokens in [1, 8, 128, 1024, 16384]:
        h = torch.randn(tokens, 2560, device="cuda", dtype=torch.bfloat16)
        c = torch.randn(2560, device="cuda")
        for _ in range(5):
            compiled(h, c)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = compiled(h, c)
        for _ in range(10):
            graph.replay()
        begin, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        begin.record()
        for _ in range(200):
            graph.replay()
        end.record()
        end.synchronize()
        graph_us = begin.elapsed_time(end) * 1000 / 200
        runner = SimpleNamespace()
        qsl = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
        alpha = np.array([0.05], np.float32)
        s.fill(runner, alpha, qsl, tokens)
        torch.accelerator.synchronize()
        started = time.perf_counter()
        for _ in range(50):
            s.fill(runner, alpha, qsl, tokens)
        torch.accelerator.synchronize()
        fill_us = (time.perf_counter() - started) * 1e6 / 50
        assert torch.isfinite(out).all()
        result["cases"].append(
            {
                "tokens": tokens,
                "single_layer_graph_replay_us": graph_us,
                "fill_wall_us_per_step": fill_us,
                "graph_replays": 200,
                "fill_iterations": 50,
            }
        )
    result["status"] = "PASS"
    result["limitations"] = (
        "Isolated timings include replay/launch overhead. Actual whole-model fusion, "
        "other work on the node, generation lengths and prefix-hit mix affect serving."
    )
    write_json(args.output, result)


if __name__ == "__main__":
    main()
