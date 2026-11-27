# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only mHC GPU interval inventory; direct traces cannot prove graph gaps."""

import argparse
import hashlib
import json
from pathlib import Path


def native_graph(trace):
    """Join the last native graph launch to exactly its reported dispatches."""
    clients = [
        client
        for client in trace["rocprofiler-sdk-tool"]
        if client.get("buffer_records", {}).get("hip_graph")
    ]
    if len(clients) != 1:
        raise ValueError("Expected exactly one native graph-tracing client")
    client = clients[0]
    launch = max(
        client["buffer_records"]["hip_graph"], key=lambda r: r["start_timestamp"]
    )
    correlation = launch["correlation_id"]["internal"]
    graph_id = launch["graph_exec_id"]["handle"]
    dispatches = [
        record
        for record in client["buffer_records"]["kernel_dispatch"]
        if record["correlation_id"]["internal"] == correlation
        and record.get("graph_exec_id") == graph_id
    ]
    if not dispatches or len(dispatches) != launch["kernel_dispatch_count"]:
        raise ValueError(
            "Native graph dispatch count/correlation evidence is incomplete"
        )
    symbols = {symbol["kernel_id"]: symbol for symbol in client["kernel_symbols"]}
    base = launch["start_timestamp"]
    events = [dict(cat="hip_runtime", name="hipGraphLaunch", ts=0, dur=0)]
    for record in dispatches:
        info = record["dispatch_info"]
        symbol = symbols[info["kernel_id"]]
        events.append(
            dict(
                cat="kernel",
                name=symbol["demangled_kernel_name"],
                ts=(record["start_timestamp"] - base) / 1000,
                dur=(record["end_timestamp"] - record["start_timestamp"]) / 1000,
                pid=info["agent_id"]["handle"],
                tid=info["queue_id"]["handle"],
            )
        )
    return dict(traceEvents=events), dict(
        graph_exec_id=graph_id,
        launch_correlation_id=correlation,
        kernel_dispatch_count=launch["kernel_dispatch_count"],
        graph_node_ids=sorted(record["graph_node_id"] for record in dispatches),
        launch_cpu_duration_us=(launch["end_timestamp"] - base) / 1000,
        first_gpu_start_after_launch_begin_us=min(event["ts"] for event in events[1:]),
    )


def analyze(path, capture_kind, rows, branch):
    raw = path.read_bytes()
    trace = json.loads(raw)
    native = None
    if "rocprofiler-sdk-tool" in trace:
        trace, native = native_graph(trace)
    graph_launches = [
        event
        for event in trace["traceEvents"]
        if event.get("cat") in ("cuda_runtime", "hip_runtime")
        and "GraphLaunch" in event.get("name", "")
    ]
    if (capture_kind == "graph") != bool(graph_launches):
        raise ValueError("Capture-kind annotation disagrees with graph-launch evidence")
    kernels = sorted(
        (event for event in trace["traceEvents"] if event.get("cat") == "kernel"),
        key=lambda event: event["ts"],
    )
    if not kernels:
        raise ValueError("No GPU kernel events; no latency conclusion is possible")
    streams = {(event.get("pid"), event.get("tid")) for event in kernels}
    if len(streams) != 1:
        raise ValueError("Use a single-stream mHC trace for the sequential gap budget")
    records, previous_end = [], None
    for event in kernels:
        name = event["name"]
        if "mhc_pre_gemm" in name:
            stage = "project"
        elif name.startswith("head_"):
            stage = "head"
        elif name.startswith("mix_"):
            stage = "mix"
        elif "MulFunctor" in name or "CUDAFunctorOnSelf_add" in name:
            stage = "synthetic_affine_branch"
        else:
            stage = "unclassified"
        gap = 0 if previous_end is None else event["ts"] - previous_end
        if gap < -1e-9:
            raise ValueError(
                "Overlapping GPU events invalidate a sequential gap budget"
            )
        records.append(
            dict(
                stage=stage,
                name=name,
                gpu_duration_us=event["dur"],
                gap_before_us=max(gap, 0),
            )
        )
        previous_end = event["ts"] + event["dur"]
    return dict(
        schema=1,
        trace_sha256=hashlib.sha256(raw).hexdigest(),
        capture_kind=capture_kind,
        native_graph_evidence=native,
        rows=rows,
        branch=branch,
        graph_launch_count=len(graph_launches),
        kernels=records,
        gpu_duration_sum_us=sum(event["dur"] for event in kernels),
        gpu_first_to_last_span_us=previous_end - kernels[0]["ts"],
        gap_sum_us=sum(record["gap_before_us"] for record in records),
        caveats=[
            "Profiler-instrumented durations are not unprofiled event timings.",
            "GPU intervals alone do not distinguish queue launch latency from work.",
            "Direct-call CPU gaps cannot be attributed to captured graph replay.",
            "An affine branch is not real attention or MoE serving work.",
            "No bandwidth/occupancy/roofline conclusion without hardware counters.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--capture-kind", choices=["direct", "graph"], required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument(
        "--branch", choices=["identity", "affine", "model"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.trace, args.capture_kind, args.rows, args.branch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
