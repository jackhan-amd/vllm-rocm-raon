# RDA native port and whole-model optimization tasks

The major targets remain the **RDA block and MoE expert chain**. Gate/mHC
fusion work is retained, but the preceding roofline experiment covered the
RDA **value-ratio gate**, not the state-update recurrence. Its very low CU
utilization must not be attributed to the RDA recurrence.

This audit performs CPU analysis of existing traces and pinned source. No GPU
allocation, kernel implementation, backend change or new speed measurement was
made. The HIP port described below is not implemented or validated yet.

## Whole target-graph budget

These are instrumented elapsed graph partitions, not unprofiled E2E fractions.
Attention/MLP branches include input norm and their projections and exclude mHC
wrappers. Shared-expert overlap is contained in the MoE branch rather than added
as a second component. Each graph's partitions reconstruct its kernel envelope
exactly in integer nanoseconds. The detailed tracer substantially inflates small
kernel times. No scaling to low-overhead timings is applied.

| Component | SD off C1 | SD off C32 | SD on C32 FULL-graph subset |
| --- | ---: | ---: | ---: |
| RDA branch | 35.91% | 31.31% | 43.78% |
| MoE branch | 34.32% | 48.93% | 42.11% |
| mHC wrappers | 16.65% | 10.70% | 6.19% |
| MLA branch | 11.46% | 7.97% | 7.06% |
| First Dense MLP | 1.17% | 0.75% | 0.51% |
| Prefix/suffix/inter-site gaps | 0.50% | 0.33% | 0.34% |
| Profiled target-graph envelope | 6.030 ms | 9.559 ms | 18.751 ms |
| Included FULL graphs / sampled steps | 64/64 | 64/64 | 39/64 |

SD-on iterations verify multiple tokens; they are not output-token latencies.
The SD-on subset excludes fragmented/mixed steps and cannot establish an all-step
on/off speed comparison. Final LM-head/logits and sampling are outside these
target graphs. The checkpoint has 18 RDA and 7 MLA layers, with one MLP in every
layer; the attention-layer count ratio is not its fraction of execution time.

Nested RDA subsets, in milliseconds per included target graph:

| RDA work | SD off C1 | SD off C32 | SD on C32 FULL subset |
| --- | ---: | ---: | ---: |
| Four logical projections, including GEMM epilogues | 0.766 | 0.822 | 1.593 |
| Value-ratio gate | 0.600 | 0.605 | 0.605 |
| Native recurrence | 0.096 | 0.766 | 5.126 |
| Native conv | 0.086 | 0.091 | 0.163 |
| Native gated norm | 0.090 | 0.091 | 0.097 |
| BF16 fills | 0.175 | 0.181 | 0.180 |
| Index preparation and other work | 0.257 | 0.343 | 0.352 |
| Input norm | 0.094 | 0.094 | 0.093 |

These kernel unions are nested within the RDA branch, not additional disjoint
opportunities. Some large projection shapes add a Tensile GSU epilogue dispatch;
four logical GEMMs do not imply exactly four GPU dispatches. Their times do not
predict removable latency, especially for tiny kernels affected by tracing.

## Actual current paths and a native-coverage gap

| Phase | Current implementation | Limitation |
| --- | --- | --- |
| Fused prefill | Kimi native HIP prologue/chunk, with the RDA V correction in the caller | Not the decode kernel |
| Pure AR decode | AITER FlyDSL conv + local AITER-GDN-derived FlyDSL recurrence + FlyDSL gated norm | Three compute kernels plus gate/index/fill preparation |
| Pure SD verify | Same FlyDSL chain with accepted-count rollback and candidate state writes | Not a single fused Kimi HIP RDA kernel |
| Prefill + SD mixed step | Native HIP prefill plus vendored Triton SD recurrence | Current native decode adapter declines mixed batches |

The SD-on C32 trace has **25/64 steps with both HIP prefill prologue and Triton
recurrence**. Exact HIP launch-correlation joins map all relevant kernels to
their CPU worker-execute scopes, with zero unmatched events. Those steps contain
450 Triton recurrence calls (25 x 18 RDA layers). The other 39 steps contain 702
FlyDSL recurrence calls (39 x 18). C1 has 62 native decode steps and two prefill
steps, without mixed Triton recurrence.

The implicated zero-based C32 step indices are:
`14,17,19,21,23,24,26,28,29,30,31,32,33,38,40,43,44,51,53,54,55,57,58,59,61`.
Source confirms `rocm_aiter_decode.try_forward()` returns false for prefill or
mixed AR/SD metadata, and `_forward_step()` subsequently uses the legacy AMD
branches. This is an actual coverage gap, not evidence that Triton is slower for
these differently shaped steps. It needs an optimized mixed-batch integration
with separate prefill/decode state ownership and validated scatter/metadata.

## CUDA RDA sources exist

Pinned upstream `krafton-ai/vllm@raon` commit:
`7dcdc8a35f1d76a03fa74af6c022ff6be0ef3bf0` (fetched for this audit).

- [AR fused CUDA RDA](https://github.com/krafton-ai/vllm/blob/7dcdc8a35f1d76a03fa74af6c022ff6be0ef3bf0/vllm/models/raon1_mini/kda_sgl_fused_kernel.cu):
  SGLang KDA-derived conv/RDA-value-scale/recurrence/gated-norm fusion with a
  40-head instantiation. Its default numerics keep some intermediates in FP32;
  it is a semantic/scheduling reference, not a bitwise numerical oracle.
- A separate `kda_sgl_fused_bf16_kernel.cu` variant is present. Precision
  guarantees must be checked in the implementation rather than inferred from
  that filename.
- [SD fused CUDA RDA](https://github.com/krafton-ai/vllm/blob/7dcdc8a35f1d76a03fa74af6c022ff6be0ef3bf0/vllm/models/raon1_mini/kda_sgl_fused_spec_kernel.cu):
  packed/ragged verify up to eight positions, accepted-state rollback,
  per-token checkpoint writes and explicit BF16 seams. Its optional small-grid
  splitting uses CUDA thread-block clusters/distributed shared memory.
- Local HIP reference:
  `csrc/kimi_k3/fused_kda_decode_kernel_rocm.cu`, plus
  `vllm/models/kimi_k3/amd/ops/kda_decode.py`. Latest upstream places the HIP
  source under `csrc/libtorch_stable/kimi_k3/`.

The HIP KDA entry currently instantiates 12/24/48/96 heads and rejects SD.
It does not implement independent RDA write gating. Removing the head or
capability guard would not implement the missing semantics.

## Port design

Use CUDA RDA for its equations and cache protocol, and Kimi HIP for gfx950
wave64/DPP reduction, vectorized state traffic, register/LDS scheduling and
conv/norm fusion. Implement a native HIP derivative manually; no hipify, PTX
translation or silent Triton substitute.

1. **AR native HIP RDA operation.** Add a 40-head instantiation and an explicit
   independent RDA write-gate input. Fold ratio preparation into the native
   frontend where viable. Keep conv output, scaled V and recurrence-output BF16
   boundaries required by the current model. Keep the original KDA API behavior
   unchanged; expose the RDA contract separately rather than overloading beta.
   Adapt AR slot-zero validity and negative padding: the Kimi kernel currently
   treats slot zero as NULL, whereas the active RDA AR path permits it.
2. **SD native HIP RDA operation.** Port the CUDA SD cache protocol, including
   all accepted counts and each candidate's FP32 state. Start with one CTA/head
   to avoid assuming CUDA cluster primitives exist on AMD. Compare that with
   a staged native prologue, V-split recurrence and norm for small grids; a
   one-kernel port is not automatically faster if it reduces parallelism or
   spills registers. Never remove required rollback checkpoints for speed.
3. **Mixed-batch native integration.** Dispatch eligible AR/SD sub-batches to
   native kernels while prefill retains HIP chunk execution. Correct packed
   token indirection, state ownership and output scatter are required. Capture
   must refresh accepted counts/indices and must not repeat live cache updates.

RDA uses independent erase and write gates. With `e=sigmoid(beta)` and
`r=exp(logsigmoid(beta+delta)-logsigmoid(beta))`, the current update uses
`v_scaled=BF16(v_conv_BF16*r)` before the erase-gated correction. Replacing beta
by `sigmoid(beta+delta)` changes the erase term; moving V scaling before
conv/SiLU changes the model. Algebraically cancelling the ratio can also remove
an observable BF16 rounding boundary. Torch gold remains the numerical oracle.

## Recurrence-specific opportunities

The current FlyDSL implementation owns 16 V rows per CTA and uses eight CTAs
per head. Q/K normalization and vector decay are recomputed across the V row
groups although they are common to a token/head. This is source-level repeated
work, not a prediction of 128x kernel speedup. Kimi HIP stages common data once
per head block; a native shared prologue can provide the same benefit to V-split
schedules.

State float4 address mapping is another concrete difference. For 16 K lanes,
current FlyDSL uses `8*lane + 4*j + element`, while Kimi HIP uses
`4*lane + 64*j + element`, with `j=0,1` and `element=0..3`. Both cover K=128.
The current mapping touches four 128-byte regions per float4 instruction; the
HIP mapping touches two fully covered regions. This mechanical address check
does **not** establish twice the HBM transactions: cache/write combining and
compiled code require counter/ISA validation.

The FP32 state is 64 KiB per head, or 2.5 MiB per sequence/layer at 40 heads.
One AR read plus write across 18 RDA layers is 90 MiB/sequence/step. A full
eight-position SD verify with one initial read and eight checkpoint writes is
405 MiB/sequence/iteration. These are logical traffic models, not measured HBM
bytes; live rows, cache hits and padding affect actual traffic. The current
FlyDSL already holds state across SD tokens, so a port cannot claim to eliminate
eight initial-state reads that it never made.

## Implementation and acceptance tasks retained

All rows below remain work to execute; this source audit does not mark them done.

| Task | Current status | Acceptance evidence |
| --- | --- | --- |
| RDA recurrence/conv/norm hardware counters | Pending; previous roofline was gate-only | Real shapes, FP32 state, cache-regime notes, roofline and transaction metrics |
| Native HIP AR RDA derivative | Port sources and semantic differences identified | Independent Torch output/state gold, native kernel witness, changed-input/state graph replay |
| Native HIP SD RDA derivative | CUDA SD reference and AMD scheduling alternatives identified | All accepted counts, ragged queries, reordered slots, NULL/PAD and every checkpoint |
| Mixed prefill/SD native path | Gap reproduced in 25 C32 steps | Mixed-batch native witnesses, exact single cache advance, correct scatter and graph metadata refresh |
| RDA gate fusion and safe initialization removal | Retained | Stable extreme-value gold, BF16 seams, complete write coverage |
| mHC tile 64/128, finish/head fusion, head/mix tiles | Retained | Real weights, Torch gold/refit/replay, total-site timing |
| MoE small-M expert-chain schedule | Retained as a major target | Keep SiTU/quant/ordered-combine contracts; do not repeat already losing M16/direct-GEMV trials unchanged |
| MLA, projection/Dense MLP and LM head | Retained in the complete budget | Actual backend witnesses and phase-specific timings before candidate selection |

Shared RDA/mHC arithmetic does not change expert packing. MXFP4 and MXFP8 must
retain their distinct Quark/AITER/FlyDSL contracts and each receive acceptance
coverage. All unvalidated candidates stay opt-in; running evaluations/defaults
remain unchanged. Warmup and compilation remain outside performance windows.

At the illustrative C32 SD-off graph fractions, RDA alone—even at zero cost—
would yield only about 1.46x for that graph. RDA and MoE together occupy about
80% there (about 86% in the SD-on FULL subset). These instrumented fractions
motivate joint work; they do not predict a 2x E2E gain or validate the roughly
10% small-fusion expectation. Kernel and model measurements are both required.

## Evidence

- [Source and trace audit receipt](../../tools/raon_validation/results/rda-native-audit-20261127.json).
- Existing trace scripts/results on Thera:
  `/mnt/hammerspace/jackhan/raon-gap-budget-20261127/analysis/whole-model-budget.json`,
  `rda-whole-audit.json`, and `rda-step-coverage.json` in the same directory.
- Preserved upstream source snapshots:
  `.amd-validation/raon-rda-native-audit-20261127/sources` in the local workspace.
- [Earlier gate/mHC roofline](KERNEL-ROOFLINE-20261127.md).
