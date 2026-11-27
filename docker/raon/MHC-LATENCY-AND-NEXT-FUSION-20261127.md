# mHC latency evidence and next fusion boundary

This analysis uses the preserved job-89710 trace and performs no new GPU
execution. The trace is a **direct block call**, not a captured graph replay.
The fixture's branch is BF16 multiply/add; it is not attention or MoE.
The subsequent [job-89711 graph measurements](MHC-GRAPH-LATENCY-20261127.md)
now validate separate-stage graph costs and native graph dispatch evidence.
They also record the Torch profiler's inability to expose graph kernel events.

## What is measured

For rows=256 at the last selected MLP mHC site:

| Stage | Instrumented GPU duration |
| --- | ---: |
| AITER HIP project/statistics | 6.439 us |
| FlyDSL fused finish/head/aggregate | 4.759 us |
| Synthetic branch multiply | 3.039 us |
| Synthetic branch add | 2.959 us |
| FlyDSL residual mix | 3.959 us |
| Sum | 21.155 us |

The gaps between these direct-call GPU intervals are 68.122, 33.081, 42.201
and 27.761 us. The first-to-last interval is 192.320 us. These gaps include
instrumented CPU dispatch/scheduling and are **not** evidence of that much
latency inside a CUDA/HIP graph. The separately measured unprofiled graph block
was 19.941 us, even shorter than the instrumented kernel-duration sum. Subtracting
one measurement from the other cannot identify graph launch overhead.

The current evidence therefore establishes the kernel sequence and a rough
instrumented work distribution. It does not separate graph node dispatch,
memory stalls and arithmetic. A warmed graph-replay trace is required for the
first distinction; hardware counters are required for bandwidth/occupancy or
roofline claims. The [CPU interval receipt](../../tools/raon_validation/results/mhc-latency-20261127.json)
preserves the numbers and caveats.

## Fusion that is and is not possible

The actual dependency is `project/statistics -> head/aggregate -> real
attention or MoE branch -> residual mix`. Head produces the branch input; mix
needs its output. Head and mix cannot be directly fused across the real branch
without moving that whole computation into the fused implementation. Fusing
the toy affine branch would remove two fixture kernels and would not establish
a deployable model improvement.

Finish/head fusion has already removed one real launch while preserving all
90 tested native outputs exactly. Project/head is the next relevant boundary,
but the existing project uses 16 K-split partials. A block-local barrier cannot
make those different CTAs' results globally ready for a head appended to the
existing kernel.

The useful next candidates are:

1. **Small-row single-CTA project/head.** One CTA owns a token's 17 projections,
   square sum, chart and aggregate. This avoids cross-CTA reduction and one
   launch. For rows=1 it also reduces an already small grid, so occupancy and
   latency can worsen. The logical projection is 174,080 MACs per token and
   348,160 BF16 phi bytes; this is a work model, not measured traffic. A staged
   20-KiB input can be reused for dot products and aggregate, but tensor-core
   utilization and register pressure need measurement. Keep the score BF16
   boundary and a batch-independent reduction schedule; compare Torch gold,
   old native outputs, refit and replay before timing.
2. **Preserve split-K and use a last-producer epilogue.** Release/acquire
   publication and an atomic completion counter could let the last producer
   reduce partials and run the head. This retains partial writes and adds
   atomics. A per-replay counter reset must be proven race-free; a separate
   reset launch can erase the saving. This is a more invasive AITER change,
   not a safe replacement for the existing launch boundary.
3. **Head/mix tiling.** Current head/mix use 128 threads and ten 256-channel
   tiles per token. Smaller tiles increase independent work but also repeat
   chart and split reductions more often; larger tiles reduce grid size.
   Their tradeoff cannot be selected from duration alone. The already observed
   AITER project kernel template contains both 64 and 128; their roles must be
   checked in its source before describing a positional argument change as a
   particular tile-size experiment.

No new kernel or default was enabled by this analysis. A single-CTA prototype
is justified only as an opt-in small-row candidate, with its selection bound
chosen from measurements. Even eliminating the measured mHC block entirely is
not a twofold model-speed claim: real model fractions and the branch remain.

## Next GPU validation commands, prepared but not run

The existing gold tool now supports `--profile-graph`. It warms and captures
outside profiling, then records one replay. Without this flag, historical
direct-call profiling is preserved. The output records the profile kind,
rows and branch so the two forms are not confused.

```bash
uv run --no-project --python .venv/bin/python -m tools.raon_validation.mhc_math \
  --model /models/Raon-1-mini --backend hip --finisher flydsl --gate flydsl \
  --fuse-finish-head --branch affine --layers 24 --rows 256 --scales 1 \
  --profile-graph --profile evidence/mhc-graph-256.json \
  --output evidence/mhc-graph-256-gold.json
uv run --no-project --python .venv/bin/python \
  -m tools.raon_validation.mhc_trace_budget \
  --trace evidence/mhc-graph-256.json --capture-kind graph --rows 256 \
  --branch affine --output evidence/mhc-graph-256-intervals.json
```

Repeat the distinct graph-profile condition at rows 1 and 32. Collect one
profiler at a time on the node, since simultaneous profiling previously lost
GPU witnesses. Inspect GPU events and graph-node order before interpreting
gaps; the CPU-only tool refuses empty, multi-stream or overlapping interval
inventories. Follow with a real model graph trace to locate mHC relative to the
actual attention/MoE branch. No GPU allocation was made for these commands.
