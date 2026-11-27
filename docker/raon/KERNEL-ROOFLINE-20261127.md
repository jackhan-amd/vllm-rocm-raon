# RDA gate and native mHC roofline (2026-11-27)

Both selected parts are below whole-device compute and bandwidth ceilings in
these isolated diagnostics. RDA gate fusion has stronger support than a claim
that its seven kernels are already saturating the GPU. mHC has scope for
small-shape scheduling/tile work; no candidate speedup has been measured.

## Scope and validity

- MI355X/gfx950, pinned ROCm 10.2 image, stateless TP1/DP1 component calls.
- Eight independent GPU tasks: RDA and mHC at 1, 8, 32 and 256 rows.
- RDA uses BF16 `[1,T,40]` inputs; beta retains the production projection
  row stride of 20656. The seven-operation Torch gate is unchanged.
- mHC uses real layer-0 attention-site checkpoint weights and synthetic
  BF16 `[T,4,2560]` activations. A fixed BF16 branch output excludes actual
  Attention/MoE computation while retaining project/finish/head/mix work.
- Twenty warm calls precede the baseline. One event interval measures 512
  graph replays without the counter profiler. Counter collection is separate:
  iteration multiplexing over nine counter sets, kernel iterations 101:200.
  All unique selected kernels have 100 sampled dispatches. All eight workload,
  counter and analysis exits passed. The counter run serializes streams.
- Components match the current workspace source hashes. No inference kernel,
  default runtime switch, quantization policy or accuracy evaluation changed.
- Accepted job 89705 completed 0:0 in 00:29:29; container cleanup passed.
  Job 89704 failed before workload execution because the image cache was absent.
  Setup/analysis invocation errors were resolved without recollecting counters.

## Primary results

The values below are native Compute Profiler metrics for **serialized isolated
kernel dispatches**, not utilization of the serving process or latency shares.
HBM percentages divide measured kernel bandwidth by the empirical roof on
that task GPU, approximately 6.95–6.97 TB/s. L2 roofs are about 34.4–34.5 TB/s.

| Component | CU utilization C1 | CU utilization C32 | C32 HBM GB/s | C32 fraction of empirical HBM roof |
| --- | ---: | ---: | ---: | ---: |
| rda cast (strided) | Merged cast name at C1 | 0.1561% | 8.024 | 0.1151% |
| rda add/sub | 0.0443% | 0.1050% | 10.823 | 0.1552% |
| rda cast | 0.0570% | 0.1087% | 3.380 | 0.0485% |
| rda exp | 0.0512% | 0.1024% | 9.518 | 0.1365% |
| rda logsigmoid | 0.0729% | 0.1984% | 6.340 | 0.0909% |
| mhc finish | 0.0171% | 0.5668% | 43.100 | 0.6196% |
| mhc head | 0.5268% | 14.2535% | 222.701 | 3.2015% |
| mhc mix | 0.4792% | 13.0606% | 432.723 | 6.2206% |
| mhc mHC project | 4.0463% | 4.2794% | 390.846 | 5.6186% |

The C32 mHC projection attains about 14.89 TFLOP/s of BF16 MFMA work,
approximately 0.61% of its empirical BF16 MFMA roof. Head/mix use vector
arithmetic; comparing them to the BF16 matrix peak would be misleading.

The roofline point can lie on the memory side of the ridge while remaining
far below the sloping bandwidth roof. That is not evidence of saturated
HBM. For these tiny kernels, limited parallelism, dispatch/dependency
latency and instruction latency remain relevant. Ordinary FLOP roofline
does not fully model exp/log/SFU limits. CU utilization is an active-cycle
metric, not a percentage of peak FLOPs or a removable latency fraction.

## Supporting larger shape

At 256 rows, mHC head/mix CU utilization rises to about 27%, and
projection/mix reach about 2.28/2.24 TB/s HBM bandwidth (roughly 33/32%
of the measured HBM roof). The large shape needs its own tile decision;
a C1 winner must not automatically replace its configuration.

## One unprofiled graph interval per component/shape

These isolated graph-replay timings include graph submission/pacing effects.
They are not per-node times inside a complete model graph or serving TPS.

| Rows | RDA gate graph microseconds | mHC graph microseconds |
| --- | ---: | ---: |
| 1 | 14.177 | 11.884 |
| 8 | 14.592 | 12.724 |
| 32 | 15.864 | 12.694 |
| 256 | 16.325 | 16.860 |

The earlier 0.60 ms RDA-gate total came from a detailed whole-model trace
with substantial observer overhead. It must not be treated as a predicted
fusion saving or scaled using one whole-window overhead ratio.

## Reviewed candidates

1. **RDA seven-to-one native FlyDSL gate.** Read BF16 strided beta and delta,
   compute the stable FP32 log-sigmoid difference and exp in one kernel,
   and emit FP32 ratio. Start with wave-size/head-aware tiles for 40 heads.
   Preserve stable log-sigmoid math; do not replace it with a sigmoid quotient.
   Validate extremes, noncontiguous input, changed-input graph replay and
   downstream RDA Torch gold before measuring end-to-end benefit.
2. **Native AITER projection tile 64 versus current 128.** The installed
   `mhc_pre_gemm_sqrsum(..., tile_k=128, w_preshuffle_bf16=1)` API explicitly
   supports `tile_k` 64 or 128. This is an existing HIP candidate, not a
   Triton substitution. Check output/partial-buffer layout and real-weight
   BF16 rounding parity before timing; no superiority is assumed.
3. **Fuse mHC finish into head.** Eliminate a launch and score/mean
   materialization, while preserving the 16-way sum order and explicit
   BF16 score rounding before FP32 gate math. Extra reduction work repeated
   across head tiles can offset savings; evaluate total-site timing.
4. **Bounded head/mix tile review.** Compare wave/block sizes 64/128 and
   channel tiles 128/256/512 where divisibility and write ownership hold.
   Smaller tiles expose more blocks but repeat chart math; larger tiles
   reduce repetition but may reduce parallelism or increase registers.
   Head and mix remain separated by the actual Attention/MoE branch.

These shared BF16/FP32 components are independent of expert weight packing.
The diagnostics are based on the MXFP8 runtime shapes; they do not establish
MXFP4 end-to-end performance or correctness of a future implementation.

## Artifacts and limitations

- [Compact counter receipt](../../tools/raon_validation/results/kernel-roofline-20261127.json).
- Raw scripts, source fingerprints, commands, native CSV metrics and four
  interactive roofline HTML charts:
  `/mnt/hammerspace/jackhan/raon-roofline-20261127/runs/89705`.
- Portable archive: `/mnt/hammerspace/jackhan/raon-roofline-20261127/roofline-artifacts-89705.zip`.
- Profiler: `rocm-profiler==10.2.0`, Compute Profiler 3.10.0, SDK 1.4.2.
  Analysis uses an isolated environment with the tool requirements pinned.
  The SDK frontend selected the wrong Python environment for dependency
  verification; explicit execution of the installed analysis script with
  its own venv resolved this without modifying profiler source.
- The analyzer warns about an unavailable empirical MFMA F6F4 peak.
  It is unrelated to the FP32/BF16 paths reported here.
- Reused inputs and one mHC site do not reproduce full-model cache
  interference. Counter multiplexing samples metrics across dispatches.
  No occupancy-to-speedup conversion or candidate gain is claimed.
- [Profiler collection and roofline method](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-10.2.0/how-to/profile/mode.html).
