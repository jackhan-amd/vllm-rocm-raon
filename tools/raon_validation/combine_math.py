# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact Torch-gold, changed-route graph replay and timing for FlyDSL combine."""

import argparse
import json
from pathlib import Path

import torch

from vllm.models.raon1_mini.rocm_flydsl_combine import combine_unweighted


def gold(values, ids):
    order = ids.argsort(dim=1, stable=True)
    rows = torch.arange(values.shape[0], device=values.device)
    acc = torch.zeros_like(values[:, 0], dtype=torch.float32)
    for j in range(values.shape[1]):
        acc = acc + values[rows, order[:, j]].float()
    return acc.to(values.dtype)


def timing(fn):
    for _ in range(3):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(8):
            fn()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(40):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / 320


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, choices=[6, 8], default=6)
    args = parser.parse_args()
    torch.manual_seed(1402)
    result = dict(status="RUNNING", topk=args.topk, cases=[], timings=[])
    try:
        with torch.inference_mode():
            for rows in (1, 8, 32, 64, 128, 256, 1024):
                for dtype in (torch.int32, torch.int64):
                    ids = torch.randint(
                        -1, 16, (rows, args.topk), device="cuda", dtype=dtype
                    )
                    values = torch.randn(
                        (rows, args.topk, 2560), device="cuda", dtype=torch.bfloat16
                    )
                    for mode in ("random", "cancellation", "nonfinite"):
                        if mode == "cancellation":
                            values[:, 0] = 256
                            values[:, 1] = -256
                            values[:, 2] = 0.125
                        if mode == "nonfinite":
                            values[:, 3, :4] = float("inf")
                            values[:, 4, :4] = -float("inf")
                            values[:, 5, 4:8] = float("nan")
                        actual = combine_unweighted(values, ids)
                        expected = gold(values, ids)
                        torch.testing.assert_close(
                            actual, expected, rtol=0, atol=0, equal_nan=True
                        )
                        repeat = combine_unweighted(values, ids)
                        torch.testing.assert_close(
                            actual, repeat, rtol=0, atol=0, equal_nan=True
                        )
                        one = combine_unweighted(values[:1], ids[:1])
                        torch.testing.assert_close(
                            actual[:1], one, rtol=0, atol=0, equal_nan=True
                        )
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            replayed = combine_unweighted(values, ids)
                        values.add_(0.25)
                        ids.copy_(ids.flip(1))
                        graph.replay()
                        torch.testing.assert_close(
                            replayed, gold(values, ids), rtol=0, atol=0, equal_nan=True
                        )
                        result["cases"].append(
                            dict(rows=rows, ids_dtype=str(dtype), mode=mode, exact=True)
                        )
                    values.normal_()
                    result["timings"].append(
                        dict(
                            rows=rows,
                            ids_dtype=str(dtype),
                            torch_us=timing(lambda v=values, i=ids: gold(v, i)),
                            flydsl_us=timing(
                                lambda v=values, i=ids: combine_unweighted(v, i)
                            ),
                        )
                    )
            result["status"] = "PASS"
    except BaseException as error:
        result.update(status="FAIL", error=repr(error))
        raise
    finally:
        Path(args.output).write_text(json.dumps(result, indent=2))
    print("COMBINE_GOLD", result["status"], len(result["cases"]), flush=True)


if __name__ == "__main__":
    main()
