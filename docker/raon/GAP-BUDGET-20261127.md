# Warm decode budget (2026-11-27)

Most MXFP8 TP1/DP1 SD-off step time is inside the target FULL graph.
Async scheduling was already enabled; host scheduling does not establish a
missing 2x opportunity.

| Mode | Concurrency | Target graph / step | Draft graph / step | Graph share | Diagnostic overhead |
| --- | ---: | ---: | ---: | ---: | ---: |
| SD off | 1 | 4.362 ms | — | 87.13% | 2.02% |
| SD off | 32 | 8.157 ms | — | 91.12% | 0.44% |
| SD on | 1 | 6.016 ms | 1.060 ms | 86.49% | 0.26% |
| SD on | 32 | 17.614 ms | 1.837 ms | 92.49% | Unavailable: mismatched control geometry |

Each row contains 64 serving steps after warmup. An SD-on step is a verification
iteration, not one output token. SD-on C32 is descriptive only. Graph intervals
include internal dispatch gaps, so their fraction is not CU utilization.
Inclusive execute/sample/graph/logits intervals must not be added.

## Measurement validity

Loading, compilation and capture completed before separate shape/generation
warmups. Additional mixed-batch warmup covered initially missed shapes. All eight
independent GPU replicas reached the barrier before measurement. The accepted
windows had no compilation-cache artifact changes. Events were initialized before
warmup; eight steps were skipped before recording 64 steps. Synchronization was
at window boundaries, never after each step. No extra stateful forward was run
for timing. EOS was respected; no eager fallback was used.

These GSM8K diagnostic subsets are neither the fixed ISL/OSL, 50% prefix-hit
benchmark nor an accuracy evaluation. First collection 89675 was invalid and is
excluded. Accepted allocation 89682 completed 0:0; containers were cleaned and
the node returned.

## Next targets

Removing **all** graph-external work gives arithmetic ceilings of only 1.148x
and 1.097x for SD off C1/C32. These are unrealistic ceilings because required
LM-head and sampling work are included. The remainder outside worker intervals
is not total CPU overhead or idle: preprocessing can be inside execute and other
streams may be unobserved.

Investigate graph-internal conversion/elementwise chains alongside MoE, RDA, mHC
and projections. Detailed C1-off activity contains 1067 dispatches per step,
including 42 BF16-to-FP32 conversions, 36 BF16 fills and 36 log-sigmoid calls.
Counts establish repeated work, not removable latency. RDA gate source is
compatible with part of this pattern, but absent stacks prevent attributing
all conversions to RDA. Copy/cast compute kernels are not DMA transfers.

Detailed ROCprof tracing inflated matching windows by 17–38%; its raw idle
fractions are not unprofiled idle. Each detailed case contains exactly 1536 MoE
stage1 and 1536 stage2 dispatches (64 steps x 24 layers). Nested CPU synchronization
totals overlap GPU execution and cannot be summed into removable wall time.
No 2x improvement or inference-code optimization is claimed here.

## Artifacts

- Source: `216281f592`; target snapshot:
  `3f85cd63ea420100a84378f72ae162af86a441fd`.
- Image plus recorded source overlay:
  `docker.io/hanjack/vllm-rocm@sha256:c41349b2bca3335d3a7eacceb6e4d9d21c4fa4602f7d24e454a8ccb9f739c281`.
- Thera raw cases, argv, warmup evidence:
  `/mnt/hammerspace/jackhan/raon-gap-budget-20261127`.
- Authoritative event budget: `analysis/low-overhead-validated.json`.
- TraceLens appendix: `analysis/tracelens/analysis.md`. Region classification
  lacks shapes/stacks; numeric heuristic savings are unsupported. Its report
  validator flags deliberate null savings versus raw heuristic scores. Those
  scores are not performance claims.
- [Compact evidence](../../tools/raon_validation/results/gap-budget-20261127.json).

```bash
python -m tools.raon_validation.gap_budget \
  --light-root /mnt/hammerspace/jackhan/raon-gap-budget-20261127 \
  --output low-overhead-budget.json
python -m tools.raon_validation.gap_budget \
  --case /path/to/gap-r1-trace-c32-off \
  --output sdk-budget.json --chrome-trace normalized.json
```

The CPU-only tool unions clipped intervals and rejects incomplete MoE coverage.
A synthetic graph probe verified 80/80 expected kernels first. SDK activity
collection avoids kernel replay because RDA updates live state. Future vLLM
profiling should use the documented `--disable-signal-handlers` shutdown option:
[ROCm inference profiling](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-10.2.0/how-to/using-rocprofv3-with-inference-frameworks.html).
