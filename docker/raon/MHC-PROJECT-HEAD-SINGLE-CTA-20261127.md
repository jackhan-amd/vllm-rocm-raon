# Rejected mHC single-CTA project/head candidate

**Performance rejected. Do not add this candidate to a serving optimization
profile.** The experimental flag is
`RAON_ROCM_MHC_PROJECT_HEAD=hip_single_cta`; its default remains `legacy`.
The successful native AITER/FlyDSL chain is unchanged by default.

A new local HIP kernel owns one token per 256-thread CTA. Four waves divide
17 phi projections from the original BF16 `[17,10240]` matrix. Each dot uses
16 ordered K=640 segments and a fixed wave reduction. The same CTA computes
mean square, rounds scores to BF16, evaluates Raon's chart/gates and writes
the BF16 aggregate plus 20 FP32 coefficients. The branch and FlyDSL mix remain
outside it. Direct `hipcc` compilation targets gfx950 without hipify; upstream
AITER was not modified.

The prototype preserves the score boundary and a fixed cross-batch schedule,
**not** AITER's MFMA accumulation order. It uses scalar FP32 products and wave
reductions, with no MFMA. The aggregate retains its explicit FMA chain. Live
phi/coefficient buffers support in-place refits and captured replay.

## Correctness passed, performance failed

Job `89711`, TheraC80 GPU 7: real checkpoint layers 0/12/24, attention and MLP
sites, rows 1/8/32 and scales 0.125/1/8 passed **54 Torch-gold cases and six
refit cases**. Changed-input replay, repeat identity and batch identity passed.
BF16 RMS/peak limits were unchanged. Outputs matched the unfused native
comparator exactly in 25/54 cases; all comparisons passed the numerical screen.

| Rows | Existing fused native | Single-CTA HIP | Regression |
| --- | ---: | ---: | ---: |
| 1 | 14.194 us | 45.590 us | 3.21x slower |
| 8 | 15.073 us | 46.114 us | 3.06x slower |
| 32 | 15.558 us | 46.320 us | 2.98x slower |

These whole mHC blocks include a synthetic BF16 affine branch, not attention
or MoE. Each condition used three direct warmups, capture, ten warmup replays
and one GPU-event window of 500 replays. Compilation and profiling were outside
measurement. No model benchmark or evaluation was performed.

The profiler witnessed `mhc_project_head_single_cta`, two toy branch operations
and FlyDSL mix. The new kernel alone took an instrumented 43.199 us. Removing
one launch did not offset the changed work/schedule. A smaller grid and the
replacement of AITER MFMA by scalar reductions are plausible mechanisms; no
counter-based bandwidth/occupancy attribution was collected.

This particular fusion schedule is rejected. A future project/head design
must retain useful parallelism and optimized matrix arithmetic, or validate a
different globally synchronized epilogue. It needs independent gold and timing.
The inspected AITER API also confirms its positional 128 is `tile_k`, documented
as 64 or 128; no tile-K sweep was run here.

## Reproduction and preservation

```bash
export RAON_MHC_HIP_BUILD_DIR=/cache/raon-mhc-hip
uv run --no-project --python .venv/bin/python tools/raon_validation/mhc_math.py \
  --model /models/Raon-1-mini --backend hip --finisher flydsl --gate flydsl \
  --fuse-finish-head --project-head-hip --branch affine --layers 0 12 24 \
  --rows 1 8 32 --scales 0.125 1 8 --timing-replays 500 \
  --profile evidence/project-head-profile.json \
  --output evidence/project-head-math.json
```

The [negative-result receipt](../../tools/raon_validation/results/mhc-project-head-single-cta-20261127.json)
contains hashes, acceptance counts, timings and actual kernel names. Raw source,
build command/log/binary and results are preserved in
`/mnt/hammerspace/jackhan/raon-native-mixed-20261127/mhc/final-artifacts.tar.gz`
and local `.amd-validation/mhc-graph-latency-20261127/final-artifacts.tar.gz`.
The compiled module fingerprint is `11dcd38253488c9b9f46`.
