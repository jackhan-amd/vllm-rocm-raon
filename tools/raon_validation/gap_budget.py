# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure activity coverage from selected-region rocprofv3 CSVs, without Torch."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def duration(intervals):
    return sum(end - start for start, end in merge(intervals))


def clip(intervals, start, end):
    return merge((max(start, lo), min(end, hi)) for lo, hi in intervals)


def gaps(intervals, start, end):
    result = []
    cursor = start
    for lo, hi in clip(intervals, start, end):
        if lo > cursor:
            result.append((cursor, lo))
        cursor = hi
    if cursor < end:
        result.append((cursor, end))
    return result


def rows(root, suffix):
    result = []
    for path in sorted(root.rglob("*" + suffix)):
        with path.open(newline="", encoding="utf-8") as stream:
            for index, row in enumerate(csv.DictReader(stream), start=2):
                row["source_file"] = str(path)
                row["source_line"] = index
                result.append(row)
    return result


def interval(row):
    return int(row["Start_Timestamp"]), int(row["End_Timestamp"])


def analyze(case, expected_layers=24):
    window = json.loads((case / "step-window.json").read_text())
    worker = json.loads((case / "worker.json").read_text())
    kernels = rows(case / "sdk", "kernel_trace.csv")
    markers = rows(case / "sdk", "marker_api_trace.csv")
    copies = rows(case / "sdk", "memory_copy_trace.csv")
    apis = rows(case / "sdk", "hip_api_trace.csv")
    roi = [row for row in markers if row["Function"] == "RAON_PROFILE_ROI"]
    if len(roi) != 1:
        raise ValueError(f"Expected one selected ROI, got {len(roi)}")
    start, end = interval(roi[0])
    if end <= start:
        raise ValueError("Empty or inverted selected ROI")
    selected = [row for row in kernels if start <= interval(row)[0] < end]
    names = Counter(row["Kernel_Name"] for row in selected)
    stage1 = sum(n for name, n in names.items() if "mfma_moe1_" in name)
    stage2 = sum(n for name, n in names.items() if "mfma_moe2_" in name)
    forwards = sum(step.get("scheduled_tokens", 0) > 0 for step in window["steps"])
    expected = expected_layers * forwards
    coverage = dict(
        steps=len(window["steps"]),
        nonempty_forwards=forwards,
        expected_moe_dispatches_per_stage=expected,
        observed_stage1=stage1,
        observed_stage2=stage2,
        passed=expected > 0 and stage1 == expected and stage2 == expected,
    )
    report = dict(
        status="PASS" if coverage["passed"] else "INCOMPLETE_GPU_COVERAGE",
        case=str(case),
        worker=worker,
        coverage=coverage,
        scope=(
            "Selected-region activity timestamps in ns. Busy means recorded kernel "
            "or memcpy interval coverage, not CU/MFMA utilization. Kernel durations "
            "are unioned across queues. CPU API overlap is observational, not causal "
            "attribution. Profiler overhead must be compared with the paired control."
        ),
    )
    if not coverage["passed"]:
        return report, []
    kernel_intervals = clip(map(interval, selected), start, end)
    copy_intervals = clip(map(interval, copies), start, end)
    activity = merge(kernel_intervals + copy_intervals)
    idle = gaps(activity, start, end)
    span = end - start
    kernel_ns, copy_ns, busy_ns = map(
        duration, (kernel_intervals, copy_intervals, activity)
    )
    report["activity"] = dict(
        window_ms=span / 1e6,
        kernel_union_ms=kernel_ns / 1e6,
        memcpy_union_ms=copy_ns / 1e6,
        exposed_memcpy_ms=(busy_ns - kernel_ns) / 1e6,
        copy_kernel_overlap_ms=(kernel_ns + copy_ns - busy_ns) / 1e6,
        busy_union_ms=busy_ns / 1e6,
        uncovered_ms=(span - busy_ns) / 1e6,
        uncovered_percent=100 * (span - busy_ns) / span,
        kernel_records=len(selected),
        memcpy_records=len(copies),
        summed_kernel_ms=sum(hi - lo for lo, hi in map(interval, selected)) / 1e6,
    )
    totals = defaultdict(float)
    for row in selected:
        lo, hi = interval(row)
        totals[row["Kernel_Name"]] += hi - lo
    report["top_kernels_by_summed_time"] = [
        dict(name=name, calls=names[name], summed_ms=value / 1e6)
        for name, value in sorted(totals.items(), key=lambda pair: -pair[1])[:30]
    ]
    report["gaps"] = dict(
        count=len(idle),
        under_10us_ms=sum(hi - lo for lo, hi in idle if hi - lo < 10000) / 1e6,
        at_least_100us_ms=sum(hi - lo for lo, hi in idle if hi - lo >= 100000) / 1e6,
        largest=[
            dict(start_us=(lo - start) / 1000, duration_us=(hi - lo) / 1000)
            for lo, hi in sorted(idle, key=lambda pair: pair[0] - pair[1])[:20]
        ],
    )
    graph_scopes = [
        interval(row) for row in markers if row["Function"] == "RAON_FULL_GRAPH"
    ]
    graph_ids = {
        row["Correlation_Id"]
        for row in apis
        if "GraphLaunch" in row["Function"]
        and any(lo <= interval(row)[0] <= hi for lo, hi in graph_scopes)
    }
    graph_members = defaultdict(list)
    for row in selected:
        if row.get("Correlation_Id") in graph_ids:
            graph_members[row["Correlation_Id"]].append(interval(row))
    target_ids = {
        row["Correlation_Id"]
        for row in selected
        if "mfma_moe1_" in row["Kernel_Name"] and row.get("Correlation_Id") in graph_ids
    }
    envelopes = merge(
        (min(lo for lo, _ in members), max(hi for _, hi in members))
        for key, members in graph_members.items()
        if key in target_ids
    )
    inside_graph = sum(duration(clip(envelopes, lo, hi)) for lo, hi in idle)
    execute = {
        int(row["Function"].removeprefix("RAON_EXECUTE_")): interval(row)
        for row in markers
        if row["Function"].startswith("RAON_EXECUTE_")
    }
    sample = {
        int(row["Function"].removeprefix("RAON_SAMPLE_")): interval(row)
        for row in markers
        if row["Function"].startswith("RAON_SAMPLE_")
    }
    between_steps = merge(
        (sample[index][1], execute[index + 1][0])
        for index in sample
        if index + 1 in execute
    )
    idle_between_steps = sum(duration(clip(between_steps, lo, hi)) for lo, hi in idle)
    report["gap_locations"] = dict(
        all_full_graph_launches=len(graph_ids),
        target_full_graphs_with_gpu_records=len(target_ids),
        non_target_full_graphs_with_gpu_records=len(graph_members) - len(target_ids),
        uncovered_inside_target_full_graph_ms=inside_graph / 1e6,
        uncovered_outside_target_full_graph_ms=(span - busy_ns - inside_graph) / 1e6,
        uncovered_between_worker_steps_ms=idle_between_steps / 1e6,
        note=(
            "Target graph envelopes require a MoE stage1 dispatch with the same "
            "HIP correlation ID as a launch within a FULL graph CPU scope. Other "
            "graphs can include the drafter. Between-worker-step gaps include "
            "scheduling, "
            "IPC or output readiness; this does not identify one causal component. "
            "Between-step and outside-graph values overlap and must not be added."
        ),
    )
    synchronization = [
        interval(row)
        for row in apis
        if "Synchronize" in row["Function"] or "WaitEvent" in row["Function"]
    ]
    report["cpu_sync_api"] = dict(
        union_ms=duration(clip(synchronization, start, end)) / 1e6,
        during_gpu_uncovered_ms=sum(
            duration(clip(synchronization, lo, hi)) for lo, hi in idle
        )
        / 1e6,
        note="Includes intentional final boundary drain; overlaps other budgets.",
    )
    last_sample = max((hi for _, hi in sample.values()), default=end)
    nonboundary_sync = [
        interval(row)
        for row in apis
        if ("Synchronize" in row["Function"] or "WaitEvent" in row["Function"])
        and not (
            row["Function"] == "hipDeviceSynchronize"
            and interval(row)[0] >= last_sample
        )
    ]
    report["cpu_sync_api"]["excluding_final_drain_union_ms"] = (
        duration(clip(nonboundary_sync, start, end)) / 1e6
    )
    report["module_load_api_calls"] = sum(
        "ModuleLoad" in row["Function"] for row in apis
    )
    events = []
    for group, category in (
        (selected, "kernel"),
        (copies, "gpu_memcpy"),
        (apis, "cuda_runtime"),
        (markers, "cpu_op"),
    ):
        for row in group:
            lo, hi = interval(row)
            if hi < start or lo > end:
                continue
            args = dict(row)
            args.update(
                correlation=int(row.get("Correlation_Id", 0)),
                **{"External id": int(row.get("Correlation_Id", 0))},
            )
            gpu = category in ("kernel", "gpu_memcpy")
            events.append(
                dict(
                    ph="X",
                    cat=category,
                    name=row.get("Kernel_Name", row.get("Function", row.get("Kind"))),
                    ts=lo / 1000,
                    dur=(hi - lo) / 1000,
                    pid=100000 if gpu else int(row.get("Process_Id", worker["pid"])),
                    tid=int(row.get("Queue_Id", row.get("Thread_Id", 0))),
                    args=args,
                )
            )
    return report, events


def analyze_light(root):
    """Compare stage-event windows with uninstrumented identical prompt subsets."""
    result = {
        "status": "PASS",
        "scope": (
            "Inclusive GPU stream intervals, not kernel-busy union or CU "
            "utilization. Child stages overlap parents. Outside-graph time "
            "includes necessary logits/sampling work. Window ratios estimate "
            "event overhead only where recorded step geometry matches."
        ),
        "cases": [],
    }
    for sd in ("off", "on"):
        for concurrency in (1, 32):
            suffix = f"c{concurrency}-{sd}"
            base = root / "cases" / ("light-control-" + suffix)
            case = root / "cases" / ("light-events-" + suffix)
            plain = json.loads((base / "step-window.json").read_text())
            timed = json.loads((case / "step-window.json").read_text())
            gpu = json.loads((case / "gpu-budget.json").read_text())
            groups = defaultdict(list)
            for row in gpu["intervals"]:
                groups[row["label"]].append((row["start_ms"], row["end_ms"]))

            def geometry(window):
                return [
                    (
                        step.get("scheduled_requests"),
                        step.get("scheduled_tokens"),
                        step["graphs"],
                    )
                    for step in window["steps"]
                ]

            matching = geometry(plain) == geometry(timed)
            statuses = [
                json.loads((directory / "result.json").read_text())
                for directory in (base, case)
            ]
            clean = all(
                item["status"] == "COMPLETE" and not item["cache_artifact_changes"]
                for item in statuses
            )
            graph_ms = duration(
                groups["target_full_graph"] + groups["draft_full_graph"]
            )
            total_ms = gpu["window_gpu_ms"]
            ratio = timed["wall_ms"] / plain["wall_ms"]
            result["cases"].append(
                dict(
                    sd=sd,
                    concurrency=concurrency,
                    steps=len(timed["steps"]),
                    compile_clean=clean,
                    matching_step_geometry=matching,
                    control_window_ms=plain["wall_ms"],
                    event_window_ms=timed["wall_ms"],
                    observed_window_ratio=ratio,
                    event_overhead_percent=(ratio - 1) * 100 if matching else None,
                    gpu_window_ms=total_ms,
                    stage_intervals={
                        label: {
                            "count": len(intervals),
                            "union_ms": duration(intervals),
                            "summed_ms": sum(hi - lo for lo, hi in intervals),
                        }
                        for label, intervals in groups.items()
                    },
                    all_graph_union_ms=graph_ms,
                    outside_graph_intervals_ms=total_ms - graph_ms,
                    outside_worker_intervals_ms=total_ms
                    - duration(
                        groups["worker_execute_inclusive"]
                        + groups["worker_sample_inclusive"]
                    ),
                    hypothetical_speedup_removing_all_outside_graph_work=total_ms
                    / graph_ms,
                )
            )
            if not clean:
                result["status"] = "INVALID_COMPILATION_DURING_MEASUREMENT"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--case", type=Path)
    mode.add_argument("--light-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chrome-trace", type=Path)
    args = parser.parse_args()
    if args.light_root:
        report, events = analyze_light(args.light_root), []
    else:
        report, events = analyze(args.case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    if args.chrome_trace and events:
        args.chrome_trace.parent.mkdir(parents=True, exist_ok=True)
        args.chrome_trace.write_text(
            json.dumps(
                {
                    "traceEvents": events,
                    "displayTimeUnit": "ms",
                    "normalization": (
                        "rocprofv3 ns CSV to Chrome us; CUDA category names are "
                        "PyTorch-compatible labels for original HIP APIs; "
                        "raw rows retained in args"
                    ),
                }
            )
        )
    print(report["status"], report.get("activity", report.get("coverage", "")))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
