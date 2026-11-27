# Experimental native HIP RDA SD

Subsequent [mixed wiring and shape dispatch](NATIVE-RDA-MIXED-AND-DISPATCH-20261127.md)
adds expanded-pool AR and native mixed coverage. The limitations below describe
this initial SD prototype; full-model acceptance remains pending.

`RAON_ROCM_KDA_DECODE_BACKEND=hip_rda` adds pure speculative decode to the
opt-in native AR prototype. The SD operation uses three native HIP kernels:
conv with accepted-history rollback, recurrence retaining FP32 state across
tokens and storing every candidate checkpoint, and gated output RMSNorm.
The recurrence reuses Kimi HIP's DPP reductions and float4 state mapping.
No hipify, PTX translation or Triton compute is used in this selected operation.
RDA ratio calculation and projections remain outside the operation.

The BF16 conv/SiLU, post-conv RDA ratio, BF16 V and pre-norm BF16 boundaries
are preserved. NULL/PAD prefixes do not update caches. AR/SD staged weight
buffers have separate lifetime caches so one graph cannot invalidate the other.

## Scope and remaining work

This is an experimental **pure SD** candidate, with lengths 2 through 8.
The retained `hip_rda_ar` selector still explicitly rejects SD. Existing mixed
prefill/decode batches retain the adapter's announced existing path; mixed
native coverage is not implemented. AR with an expanded speculative conv pool
is also not supported by the native AR kernel, which requires three history
taps. Do not enable this selector for general production SD serving yet.
Defaults remain unchanged. Ragged/NULL-suffix SD, cross-mode shared-layer
integration and full-model MXFP4/MXFP8 evaluation remain acceptance work.

## Finite MI355X results

Job `89710`, TheraC80, same pinned image as the
[AR prototype](NATIVE-HIP-RDA-AR-20261127.md). Seven independent GPUs passed
batch 1/2/3/4/8/16/32: AR capture/two changed-input replays and SD lengths
2/4/8 with every accepted count, reordered slots, exact conv histories and
untouched recurrent rows, output/state versus independent Torch gold, and exact
graph/direct output/cache comparisons. Existing tolerance was unchanged.

The profiler witnessed `sd_conv`, `sd_recurrence` and `sd_norm`. Some initial
profiler runs returned no GPU events and were recorded as diagnostic errors,
not numerical passes. Gold was rerun without profiler; batch32 witness was
collected with 32 disposable launches per profile. This is evidence of capture
instability, not proof of a profiler concurrency defect.

| Batch | Verify length | Existing FlyDSL adapter (us) | HIP adapter (us) | Speedup |
| --- | --- | ---: | ---: | ---: |
| 1 | 2 | 19.188 | 16.081 | 1.19x |
| 1 | 4 | 24.878 | 25.766 | 0.97x |
| 1 | 8 | 36.561 | 35.793 | 1.02x |
| 32 | 2 | 88.365 | 61.520 | 1.44x |
| 32 | 4 | 163.763 | 102.712 | 1.59x |
| 32 | 8 | 321.080 | 182.405 | 1.76x |

One event timing window per successful condition, after compile and profiler:
32 operations/graph, 20 warmup replays, 100 measured replays. Disposable state
pools only. This is a warmed single-layer adapter comparison including its
initialization operations, not full-model TPS or accuracy. Single-layer cache
reuse is more favorable than real model traffic. The final Python-only staging
cache lifetime change was followed by batch1/32 gold regression; timings predate
that change. The small-batch SD path still needs a different schedule/fusion;
native HIP by itself does not establish a speedup.

```bash
uv run --no-project --python .venv/bin/python -m tools.raon_validation aiter-rda \
  --decode-backend hip_rda --decode-batch-size 32 \
  --output evidence/hip-rda-sd-gold.json
```

Add `--decode-kernel-witness --decode-timing-replays 100` for sequential
profiling and isolated timings. Avoid concurrent profiler sessions. The
[receipt](../../tools/raon_validation/results/native-hip-rda-sd-20261127.json)
and raw `/mnt/hammerspace/jackhan/raon-hip-rda-sd-20261127` preserve the evidence.
The next RDA work is expanded-pool AR and native mixed-batch dispatch, then a
small-grid SD schedule and real-model measurements. The concurrent
[mHC finish/head experiment](MHC-FINISH-HEAD-20261127.md) is a separate change.
