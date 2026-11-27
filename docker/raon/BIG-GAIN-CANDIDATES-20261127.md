# Large remaining decode opportunities

The first target is a native **small-batch MoE schedule**, followed by the
**whole RDA path**. Another 2x end-to-end gain has not been demonstrated.
The new evidence does not support spending the next iteration primarily on
projection dispatch, sampling, or additional N-tile sweeps.

The [RDA native-port audit](RDA-NATIVE-PORT-AND-BUDGET-20261127.md) adds a complete
graph budget, pinned CUDA AR/SD references, HIP port requirements and a measured
mixed-batch native-coverage gap. The gate-only roofline must not be applied to
the recurrence kernel. All earlier fusion/tile tasks remain pending.

## First iteration: executed results

Slurm 89601 added opt-in native FlyDSL route packing plus BF16 probability
rounding in one dispatch, retaining router int32 IDs for the native ordered
combine. The combine's compiled launcher now separates int32/int64 signatures;
an independent cancellation regression reproduced old output 16 against Torch
gold 12, and the fixed launcher returned 12. Route ABI checks, full-shape
Torch-gold comparisons and actual model graph replay passed.

With default M32 expert GEMMs, single warmed SD-off diagnostic measurements
improved C1 by 5.9%, C8 by 4.2% and C32 by 3.0%. The compact record is
`tools/raon_validation/results/moe-smallm-20261127.json`. This is a useful
increment, not evidence for another 2x. M16 failed to beat matched M32 controls,
and direct route GEMV was slower even with 24 distinct-address weight sets.

A proposed RDA direct-FULL wrapper was withdrawn: V2 intentionally records
FULL capture with context mode NONE, and the existing native RDA path already
records its kernels there. A missing activation witness rejected the proposed
serving trials before any performance measurement. Python callbacks remain
relevant to PIECEWISE; they must not be reported as a newly removed FULL-decode
bottleneck. Subsequent analysis separates RDA input projection, value-ratio
gate and native conv/recurrence/norm, with an uninstrumented control.

The eight-server component diagnostic completed. Last replay sums across the
18 RDA layers were:

| Recorded region | C1 | C32 |
| --- | ---: | ---: |
| Main input projection | 0.435 ms | 0.514 ms |
| Value-ratio Torch chain | 0.360 ms | 0.467 ms |
| Native conv/recurrence/norm wrapper | 0.466 ms | 1.196 ms |

Each row is a separate execution with 18 external event pairs. Instrumented
model medians were 4.561/4.583/4.622 ms at C1, against control 4.333 ms; C32
medians were 8.440/8.546/8.460 ms, against control 8.246 ms. This appreciable
measurement overhead prevents treating the component values as exact removable
latencies or adding their percentages. The core wrapper is the strongest C32
lead; gate fusion can remove a multi-launch chain at both sizes. Main projection
already selects native FlyDSL where supported. Test kernel fusion and the
recurrence's V-row schedule, rather than adding another redundant FULL wrapper.

## Measurement and limits

Slurm 89552, TheraC80 / MI355X; published image
`sha256:c41349b2bca3335d3a7eacceb6e4d9d21c4fa4602f7d24e454a8ccb9f739c281`,
MXFP8 snapshot `3f85cd63ea420100a84378f72ae162af86a441fd`, SD off,
independent TP1/DP1 processes. Existing precision, native kernels, prefix cache,
FP8 KV, and resource settings were retained. Uniform decode used
`FULL_AND_PIECEWISE`, compilation mode 0, with breakable graphs enabled;
prefill retained PIECEWISE. No enforce-eager or implicit fallback.

External timing events were inserted into actual graph capture; no extra model
forward or live-state update was performed. After 64 serving decode replays,
events were synchronized and read once. C1/C32 denote requested concurrency and
selected graph size. Natural EOS and a 256-token diagnostic cap were used.
Captured route slots can include graph padding.

All-module nested timing added about **1.4 ms** relative to the control without
module events. Its mHC-exclusive percentage is therefore **withdrawn**. Two
additional passes instrumented only MoE or only RDA to avoid nested markers.
These isolated passes still have event overhead. The following values describe
the **last observed replay**, not an average across all prompts or an E2E TPS
budget. Different rows are separate executions and must not be summed exactly.

| Inclusive block | Graph tokens | Block time | Model graph time | Approximate model share |
| --- | ---: | ---: | ---: | ---: |
| All 24 MoE blocks | 1 | 1.881 ms | 4.931 ms | 38% |
| All 18 RDA blocks | 1 | 1.455 ms | 4.824 ms | 30% |
| All 24 MoE blocks | 32 | 4.962 ms | 9.244 ms | 54% |
| All 18 RDA blocks | 32 | 2.487 ms | 9.097 ms | 27% |

RDA includes projections, gate preparation, convolution, recurrence, norm and
output projection. MoE includes routing, sorting, activation quantization, both
expert GEMMs, the shared-expert join, and combine. Shared-expert time is not an
additional disjoint component.

The no-module-event controls measured model GPU medians of 4.603 / 8.646 ms
(C1/C32, first eight observations excluded). Outside the graph, logits including
LM head took approximately 0.196 / 0.273 ms; the complete sampler took
0.223 / 0.291 ms. Top-k/top-p plus Gumbel is contained inside sampler time.
The active V2 path uses Triton top-k/top-p, not the PyTorch full-sort fallback.
These GPU intervals exclude client, scheduler, and other host work.

## 1. Small-batch MoE: the strongest structural lead

The live MXFP8 adapter fixes `block_m=32`. A last-replay snapshot of actual
checkpoint routing across all 24 MoE layers showed:

| Graph tokens | Routed slots | Padded M32 rows | Logical row occupancy |
| --- | ---: | ---: | ---: |
| 1 | 144 | 4,608 | 3.125% |
| 32 | 4,608 | 57,696 | 7.987% |

C1 selects six experts, one row each. The C32 graph selected 65–91 distinct
experts per layer. This is **logical row occupancy**, not measured MFMA
utilization, wasted latency, or a 12x/32x speedup estimate. Weight reads, tile
masking, and fixed costs remain.

Prioritize a route-aware, small-M native FlyDSL/AITER schedule; evaluate a
direct route/slot schedule or fused expert chain against the current sorted
M32 chain. Reducing intermediate traffic and dispatch stages belongs in this
same experiment. Merely changing N or enabling another graph flag does not
address the sparse M dimension.

Pinned AITER `69753cf7f4323ef8ec511d1e3a5d273e818a7029` does not expose a
drop-in MXFP8 + SiTUv2 one-stage configuration in `fused_moe_1stage_dict`.
Its FP8/FP8 stage registry starts at M32; the v2 stage2 registry explicitly
excludes FP8/FP8 with sort block M16. Removing that check would not implement
the missing layout/kernel. MXFP4 has smaller-M native candidates, but uses
distinct packing and must be assessed through its own Quark/AITER path.
The BF16-activation/MXFP4-weight one-stage SiTU entry is not equivalent to
silently changing the current A4W4 activation contract.

Retain SiTU parameters, per-32 E8M0 scaling, probability-fold position, BF16
rounding boundaries, expert-ID-ordered FP32 combine and single shared-expert
execution. Compare expert outputs and whole-block outputs against Torch gold
before enabling a candidate. No such replacement is claimed implemented here.

## 2. RDA: optimize the complete block

The recurrence kernel alone substantially understates RDA cost. Inspect the
projection sequence and gate preparation together with the native conv/update/
norm chain. `kda.py::compute_rda_value_ratio` still launches the FP32 cast,
log-sigmoid, subtraction and exponential chain outside the recurrence.
`rocm_flydsl_decode.forward` separately prepares cache indices, initializes an
intermediate, invokes convolution and recurrence, and applies gated norm.

The next native fusion candidate should absorb gate/index preparation and
remove intermediates only where every output lane is provably written. Separately
evaluate the projection layout and recurrence's V-row schedule. A gate-only
rewrite is a component experiment, not a claim that the entire RDA block will
be 2–3x faster. Preserve FP32 state, vector decay, BF16 seams, NULL/PAD handling
and SD acceptance/state-index refresh; test state and output against Torch gold.

## Why both are needed

At a representative model fraction of 54%, making MoE twice as fast gives
`1 / (0.46 + 0.54 / 2) = 1.37x` for the model portion, before host/output
costs. Therefore MoE alone at 2x cannot deliver another 2x E2E. MoE and RDA
together are the appropriate large targets; approximately 3x component goals
are exploration targets, not predictions or measured gains. mHC remains a
secondary fusion opportunity, with its intrusive attribution excluded above.

## Evidence

Compact verified receipt:
`tools/raon_validation/results/large-blocks-20261127.json`.
Raw reports, server argv/logs, actual request outputs, instrumentation and
controllers remain under
`/home/jackhan/bench-results/raon-sd-off-native-20261127`.
The receipt includes SHA-256 hashes, actual native/FULL graph witnesses and
request completion counts. All block-diagnostic containers were removed.
This is not an accuracy result, an MXFP4 measurement, or the fixed-ISL/OSL
benchmark with 50% measured token-based prefix hits.
