# Measured mHC graph stages

Job `89711`, TheraC80 GPU 7, checked the previously prepared graph diagnostics.
No runtime default changed. All three methods below used real layer-24
attention/MLP mHC weights, rows 1/32/256 and scale 1. The branch was a **synthetic
BF16 multiply/add**, not real attention or MoE. Gold/refit checks passed before
each diagnostic. The [receipt](../../tools/raon_validation/results/mhc-graph-latency-20261127.json)
preserves methods, source hashes, graph correlation evidence and results.

## Unprofiled separate-stage graphs

Each stage used three direct warmups, capture of 32 calls, ten warmup replays,
then one GPU-event window of 500 replays. There were no event nodes or profiler
inside these captured stage graphs.

| Rows | HIP project | Fused finish/head | Mix | Whole synthetic block |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4.003 us | 2.466 us | 2.388 us | 13.366 us |
| 32 | 4.023 us | 2.795 us | 2.414 us | 14.242 us |
| 256 | 4.021 us | 2.783 us | 2.423 us | 14.060 us |

These are hot-buffer stage graph costs, including device graph dispatch, not
bare arithmetic durations. The head fixture reused a fixed native projection;
the mix fixture reused fixed native coefficients/branch outputs. Aggregate,
coefficients and mix outputs matched the full native block **exactly** at all
three shapes. The temporary projection substitution exists only in the
diagnostic process and is restored immediately; production code is unchanged.
Separate-stage numbers cannot simply be subtracted from a live model graph.
The whole block contains the two toy branch operations. Historical one-call
graph windows and these 32-call windows must not be compared as an optimization
gain.

Weak scaling with rows suggests a substantial fixed-cost floor over these
shapes. It does not prove how much is dispatch, arithmetic or cache behavior.
Project is the largest of the three measured native stages. A project/head
fusion is worth a bounded small-row experiment, with the existing 16-way
reduction and low-parallelism risks described in the
[feasibility analysis](MHC-LATENCY-AND-NEXT-FUSION-20261127.md).

## Native ROCm graph trace

Torch's graph profile recorded `hipGraphLaunch` but **zero GPU kernel events**.
The CPU interval tool rejected it rather than reporting eager timings as graph
timings. Native `rocprofv3 --hip-runtime-trace --kernel-trace` then collected the
actual graph dispatch records. The tool joins the final launch's correlation ID
and `graph_exec_id` to its reported five nodes; all five were present.

| Rows | Project | Head | Toy branch (two nodes) | Mix | GPU span |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.600 us | 4.640 us | 9.160 us | 4.040 us | 23.440 us |
| 32 | 5.521 us | 4.760 us | 9.280 us | 4.080 us | 23.641 us |
| 256 | 5.680 us | 4.640 us | 9.360 us | 4.080 us | 23.760 us |

Recorded GPU intervals touched with no positive inter-node gap at the trace's
resolution. This does **not** mean graph node scheduling is free: it can be
included within recorded intervals, and these instrumented spans exceed the
unprofiled graph costs. CPU graph-launch API durations were 5.689/5.040/12.750 us
and first-node start offsets 9.261/8.756/16.556 us from launch entry; both are
collector-instrumented and are not unprofiled serving launch latency.

External timing event nodes also worked on HIP. However, five markers per call
inflated phase totals to roughly 38 us. Those marker intervals are preserved
as an overhead diagnostic, **not** substituted for kernel durations or used to
select a faster implementation.

## Next fusion boundary

Actual inference remains `project -> head/aggregate -> attention or MoE -> mix`.
Head and mix cannot cross the actual branch without moving that computation.
Removing the toy branch's two nodes would improve this fixture only.

The next candidate is a separate small-row native project/head realization.
Appending a head to one of the existing 16 K-split CTAs cannot use a local
barrier to wait for other CTAs. A token-owned CTA can avoid that boundary but
reduces the grid and changes dot/reduction scheduling; a last-producer epilogue
requires globally ordered publication/counter logic. A subsequent
[single-CTA HIP experiment](MHC-PROJECT-HEAD-SINGLE-CTA-20261127.md) passed gold
but ran about three times slower and was rejected. A last-producer epilogue was
not implemented. Real model branch timing and hardware counters remain necessary
before claiming a large model improvement.

## Diagnostic commands

```bash
uv run --no-project --python .venv/bin/python -m tools.raon_validation.mhc_math \
  --model /models/Raon-1-mini --backend hip --finisher flydsl --gate flydsl \
  --fuse-finish-head --branch affine --layers 24 --rows 32 --scales 1 \
  --component-graph-replays 500 --output evidence/mhc-stage-32.json
rocprofv3 --hip-runtime-trace --kernel-trace --output-format json \
  --output-directory evidence/native-32 --output-file trace -- \
  uv run --no-project --python .venv/bin/python tools/raon_validation/mhc_math.py \
  --model /models/Raon-1-mini --backend hip --finisher flydsl --gate flydsl \
  --fuse-finish-head --branch affine --layers 24 --rows 32 --scales 1 \
  --external-graph-replay --output evidence/native-32-gold.json
uv run --no-project --python .venv/bin/python \
  -m tools.raon_validation.mhc_trace_budget \
  --trace evidence/native-32/trace_results.json --capture-kind graph \
  --rows 32 --branch affine --output evidence/native-32-intervals.json
```

Use the explicit script path under the external collector to avoid importing a
different installed `tools` package. Collect one profiler at a time per node.
Raw artifacts reside under
`/mnt/hammerspace/jackhan/raon-native-mixed-20261127/mhc` and local
`.amd-validation/mhc-graph-latency-20261127`.
