# Current-image profile and native fusion candidates

The first implementation candidate is a Raon-specific native FlyDSL mHC
head/chart/mix path. RDA value-ratio preparation and output initialization are
the next targets. The arithmetic-only MLA BMM rewrite remains lower priority.
This review measures and identifies existing work; **no new optimization kernel,
speedup or model-accuracy result is claimed**.

## Accepted collection

Slurm **89470**, TheraC80, completed in **00:06:55**, exit **0:0**. All eight
owned containers were cleaned up. There is no retained GPU allocation.

- Image: `docker.io/hanjack/vllm-rocm@sha256:f683e05a88ace2c842316fca4772f0f5fe72c1b5b0f2db69b7bb127a46ac9026`.
- Runtime source: `fcee6105d3cc5e3d2814a10b87e205610caa9e7a`; unchanged in this
  analysis round. AITER/FlyDSL packages and checkpoint weights are unchanged.
- Eight independent TP1/DP1 replicas: SD off/on, requested concurrency 1/32,
  and initial/prefill versus delayed-decode profiling windows.
- MXFP8 target snapshot `3f85cd63ea420100a84378f72ae162af86a441fd` and the reviewed
  DFlash2 snapshot `74f2835bd683458f6124f48784b078b06770d824`, seven draft tokens.
- Dtype auto, FP8 E4M3 KV, prefix caching, model length 262144, batched tokens
  16384, sequences 128 and memory fraction 0.85 were retained. Compilation mode
  remained 0/PIECEWISE; no enforce-eager fallback or Torch.compile experiment.
- Existing native mHC projection/finisher, direct MoE graph, split MLA and named
  native draft backend were enabled. `VLLM_ROCM_USE_AITER=1` and
  `HIP_FORCE_DEV_KERNARG=1` were verified in every case's environment record.

The earlier GSM8K inputs, temperature 1, top-p 0.95 and top-k 20 were reused.
Separate warmup preceded replay profiling. Diagnostic requests respected EOS and
had a 128-token cap. This is **not** the fixed 8K/32K/128K input benchmark, a
50% measured prefix-hit experiment, an output-TPS comparison, or an accuracy
evaluation. Prefix caching remained enabled.

The initial windows include admission ramp-up and mixed prefill/decode; they
must not be presented as pure prefill measurements. The delayed windows contain
only generation. SD-on C32 fell from 32 to 29 active requests as EOS completed
some requests; C32 denotes the requested maximum, not constant occupancy.

## GPU-event validation and profiler workaround

An earlier collection completed requests but produced CPU-only traces. It is
excluded from GPU performance analysis. The same image and MI355X reproduced
the upstream spawn/profiler problem:

| Parent initializes GPU, then spawns a profiler worker | Session 1 GPU events | Session 2 GPU events |
| --- | ---: | ---: |
| Default profiler-library registration | 0 | 0 |
| `ROCPROFILER_SET_ROCPROFILER_REGISTER_LIBRARY=0` | 10 | 10 |

The opt-out was set before the top-level Python process. All eight accepted
serving replay traces then passed an explicit nonzero GPU-kernel-event check.
The independent reproducer and workaround are described in
[PyTorch issue 182373](https://github.com/pytorch/pytorch/issues/182373).
This is a **profiling fix**, not a serving speed optimization or a precision
change. The image's inference kernels were not patched to obtain these traces.

V2 does not wire the existing Torch capture option in the same way as the legacy
runner. A diagnostics-only entry point observed the original native warmup and
actual serving-graph capture calls exactly once. It omitted export of discarded
memory-sizing graphs and used explicit module scopes instead of recording every
Python interpreter frame. It never added a forward or a live cache update.
Replay retained the normal Torch profiler's shapes and stacks.

## Steady-decode kernel inventory

Every percentage below divides by **summed profiled GPU kernel durations**.
These are not end-to-end critical-path fractions or speedup estimates. Each
decode trace has 16 CPU execute scopes (GPU annotation copies are not counted).

| Kernel family | SD off C1 | SD off C32 | SD on C1 | SD on C32 |
| --- | ---: | ---: | ---: | ---: |
| Copy/cast | 16.88% | 16.51% | 19.49% | 10.48% |
| Generic elementwise | 13.84% | 9.26% | 10.17% | 5.12% |
| Generic reduction | 0.17% | 0.12% | 0.06% | 0.03% |
| RDA/conv recurrence name family | 1.19% | 6.58% | 4.04% | 21.78% |
| MLA A4W4 BMM | 0.81% | 0.71% | 0.65% | 0.70% |
| MoE GEMM | 8.89% | 24.09% | 11.93% | 23.64% |
| Other GEMM, ownership not fully reconstructed | 29.03% | 21.13% | 27.07% | 19.95% |

Copy, elementwise and reduction still total **29.72% at SD-on C1** and
**15.64% at SD-on C32**. Thus the concern about miscellaneous small operations
remains relevant after the finisher update. The former broad reduction category
is now very small. Historical mixed prefill/decode traces are not controlled
before/after comparisons with these steady windows.

The RDA name family is not the entire RDA layer. For example, its FlyDSL conv
`update_kernel_0` and gated norm `norm_0` were initially in the generic-name
bucket. Similarly, native `finish_0`, ordered `combine_0`, and draft
`gather_1`/`metadata_0` must not be labeled unoptimized because their names were
unrecognized. Per-kernel inventories are retained with the evidence.

## 1. mHC chart fusion: a concrete first candidate

The effective mHC path is:

1. AITER HIP projection/statistics.
2. Native FlyDSL finisher.
3. Triton `_mhc_site_head`.
4. Torch `einsum("im,bmk,jk->bij", Q, z, Q).contiguous()` for the Raon chart.
5. The attention/MLP branch.
6. Triton `_mhc_site_mix`.

It is neither a wholly native FlyDSL path nor the wholly Torch fallback.
The existing fused path in `mhc.py` still leaves this chart contraction in Torch.
That distinction matters when reviewing the remaining native coverage.

In **direct runner warmup**, exact GPU External-ID joins connected the two FP32
GEMM specializations and one FP32 direct-copy specialization exclusively to
`RaonMHC` module scopes across all 25 decoder layers. Recorded `aten::bmm` shapes
match the chart contractions. This is direct-execution evidence, not invented
per-layer attribution of replay events.

The same three kernel classes in steady SD-on replay account for:

| Chart-related classes | Calls across 16 steps | C1 GPU-kernel sum | C32 GPU-kernel sum |
| --- | ---: | ---: | ---: |
| Two FP32 GEMM specializations | 1600 | 4.42% | 2.42% |
| FP32 direct-copy specialization | 1600 | 4.15% | 2.11% |
| **Total** | **3200** | **8.57%** | **4.53%** |

The counts agree with 50 mHC sites per step, two GEMMs and two copies per site:
**200 kernel launches per step**. The whole-model direct-warmup mapping and
replay class counts support this candidate; individual replay nodes still lack
complete graph ownership.

The chart table overlaps the coarse family table: its FP32 copies are already
inside Copy/cast and its GEMMs are inside Other GEMM. Do not add these shares
together as independent opportunities.

Port the existing Raon head/chart equations to FlyDSL and fold the chart into
the head, then provide a native mix using those coefficients. Preserve the
current HIP projection/finisher, BF16 score seam, FP32 intermediates, Helmert
basis, rho clamp, residual matrix and branch-output rounding. Do not replace
Raon's chart with AITER's different Sinkhorn gate. Compare real checkpoint
weights against both the current production path and independent Torch mHC
math, including changed-input graph replay and in-place refit.

The two remaining Triton head/mix classes additionally occupy 4.11% at C1 and
2.46% at C32. Merely renaming their implementation does not establish a gain;
the concrete opportunity is removing the chart's intermediate work and then
measuring the complete site.

## 2. RDA value-ratio preparation and initialization

`compute_rda_value_ratio()` in `kda.py` still computes
`exp(logsigmoid(beta + delta) - logsigmoid(beta))` through separate casts and
Torch operations before the native decode callback. The trace has **576**
log-sigmoid kernels: two per RDA layer, 18 layers and 16 steps. This one class
occupies 1.38% at C1 and 0.68% at C32; its casts/add/sub/exp are additional work.

A native FlyDSL frontend can fold this expression while retaining stable
log-sigmoid math, BF16 projection inputs and FP32 ratio output. Test extremes,
noncontiguous projection views, changed-input replay and the downstream
convolution/RDA/norm chain against Torch gold. Do not replace the expression
with an unstable sigmoid quotient.

The adapter also zeroes the output before dispatching to FlyDSL, which creates a
zeroed intermediate and then runs a norm kernel that writes every actual output
element. The observed 576 BF16-fill kernels agree with two initializations per
RDA layer/step. The outer output zero is a removal candidate. Retain required
intermediate PAD/NULL initialization until full write coverage is proven, and
validate every accepted count and untouched state row before adoption.

## 3. C32 recurrence, then scheduling

The measured 21.78% recurrence-family share keeps C32 RDA near the top of the
list. Separate repeated head/token Q/K normalization and gate preparation from
speculative full-state checkpoint traffic. Shared preparation can avoid
recomputing the same quantities across V rows, but its extra buffer traffic
must be measured. Checkpoint removal is not an acceptable shortcut: preserve
rollback states, accepted counts, reordered slots and exactly one cache advance.

Input/metadata staging and MLA callback/output copies remain candidates, but
the full copy bucket must not be attributed to metadata. MLA's arithmetic-only
A4W4 BMM remains below 1%; the previous generic BF16 wide-tile screen also gave
no basis to enable wider dispatch. Precision and MoE packing are unchanged.
The proposed mHC/RDA frontend work is independent of expert weight packing;
MXFP4 end-to-end acceptance still needs its own validation.

## Trace-quality limit and retained evidence

TraceLens `c80f7de688c485fd6924da9a4de345a785bac60e` generated all eight reports,
but the gate remains `GRAPH_UNDER_RECORDED` (reported graph-replay fraction
85.85%). Its capture
merger reports missing target capture batch sizes for V2 execution roots, so
providing capture files did not establish complete replay ownership. The
prescribed deterministic fallback ran. Agentic attribution and roofline gain
estimates were skipped. This review does not adopt the fallback's heuristic
impact estimates or add inclusive CPU timings into a purported critical path.

The machine-readable review is
[`current-kernel-decomposition-20261127.json`](../../tools/raon_validation/results/current-kernel-decomposition-20261127.json).
The complete report/inventory bundle is retained at
`/home/jackhan/bench-results/raon-other-kernels-20261126/fresh-capture-gpu/current-profile-evidence.zip`,
SHA-256 `0a79ecb47948688ac3a13280b522bde18ec756af6c3f947f1967d70a6477a332`.
It includes commands, environment records, nonzero GPU-event guards, probe
results, input-trace hashes, module mapping and the TraceLens reports. Raw
capture/replay traces remain beside the cases on Thera.
The public JSON uses hashes and readable prefixes for long GPU symbols; their
full measured spellings remain in the archived mapping file.

Earlier setup failures are preserved separately: 89424 (CRLF shell), 89425
(diagnostic multiprocessing entry point), and 89442 (memory pressure, setup
recovery and CPU-only GPU-invalid profiles). None supplies GPU performance
evidence for this review. No new native optimization has been enabled or
published as a result of this analysis.
