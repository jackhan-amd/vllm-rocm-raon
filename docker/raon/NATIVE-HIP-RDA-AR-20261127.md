# Native HIP RDA AR prototype

The opt-in `hip_rda_ar` backend derives from vLLM's existing Kimi-K3 HIP decode
kernel. It fuses conv update, RDA recurrence and gated output RMSNorm for
40 heads of dimension 128 on gfx950. It compiles directly with `hipcc`; neither
hipify nor a CUDA/PTX backend is involved. The upstream Kimi kernel is unchanged.

The derivative preserves separate erase/write gates: it applies the FP32 RDA
ratio after BF16 conv/SiLU, rounds V back to BF16, updates FP32 recurrent state,
then rounds the output to BF16 before norm. The seven-operation ratio calculation
and all projections remain outside this kernel. This is AR support only. Pure
SD raises an explicit error; mixed/prefill batches retain the adapter's reported
existing path. SD rollback/checkpoints and mixed-batch native coverage remain
pending. No default runtime profile was switched.

## MI355X validation

Thera job `89709`, node `TheraC80`, eight independent devices, Torch
`2.14.0+rocm10.2.0`, pinned image digest
`sha256:c41349b2bca3335d3a7eacceb6e4d9d21c4fa4602f7d24e454a8ccb9f739c281`.

- Initial positive-slot gold passed at batch 1/2/3/4/8/16/32/64.
- The final source passed positive-slot tests at batch 1/32, and NULL/negative
  padding at batch 1/3/32/64. Each final case includes capture plus two replays
  with changed raw inputs and reordered slot metadata, exact untouched caches
  and conv histories, and output/state against independent Torch gold.
- Final FlyDSL AR/SD regression passed at batches 1/32: SD lengths 2/4/8 and
  every accepted count, plus the AR NULL/padding tests.
- The original BF16 screen remains 1% relative RMS and 2% relative peak plus
  `1e-5` absolute tolerance. It was not relaxed. Numerical PASS is not bitwise
  equality with Torch. Graph/direct output and cache comparisons are exact.
- The profiler witnessed `raon_rda_ar_fusion_kernel<...>` as the sole GPU kernel
  in the new operation. The FlyDSL comparison witnessed conv, recurrence, norm
  and the surrounding initialization/index operations.

### Boundary defect found and corrected

The initial port inherited an incorrect assumption from the earlier candidate:
that AR physical slot zero was live. `utils.NULL_BLOCK_ID` is zero, and
`gdn_attn.py` fills padded AR metadata with it. The legacy recurrent path already
skips it. Both native HIP and the existing FlyDSL/AITER-derived candidate now
skip zero and negative slots, write zero output, and leave those caches intact.
The earlier port audit's AR-slot-zero statement is superseded by this check.

### Warm isolated graph timings

| AR batch | Existing FlyDSL adapter | HIP fused adapter | Operation speedup |
| --- | ---: | ---: | ---: |
| 1 | 15.997 us | 4.783 us | 3.34x |
| 32 | 53.408 us | 30.187 us | 1.77x |

Each condition used one event timing window: 32 operations per captured graph,
20 separate warmup replays, then 100 measured replays. Compilation and profiler
witness collection occurred before measurement. Stateful repetitions used only
disposable synthetic fixture pools. This compares the entire selected adapter,
including its required fills/index preparation; it excludes projections and
ratio calculation. It does not measure output tokens/sec, model accuracy or an
end-to-end gain. Reusing a single layer's cache is more cache-friendly than a
full model with 18 RDA layers and MoE weights. Full-model measurements are needed
before enabling this backend in a performance profile.

## Reproduction

Inside the installed ROCm runtime with one idle visible GPU and `hipcc`:

```bash
export RAON_HIP_RDA_BUILD_DIR=/cache/raon-hip-rda
uv run --no-project --python .venv/bin/python -m tools.raon_validation aiter-rda \
  --decode-backend hip_rda_ar --decode-ar-only --decode-batch-size 32 \
  --decode-kernel-witness --decode-timing-replays 100 \
  --output evidence/hip-rda-ar-b32.json
uv run --no-project --python .venv/bin/python -m tools.raon_validation aiter-rda \
  --decode-backend hip_rda_ar --decode-ar-only --decode-ar-padding \
  --decode-batch-size 32 --output evidence/hip-rda-ar-padding.json
```

`RAON_ROCM_KDA_DECODE_BACKEND=hip_rda_ar` selects the experimental runtime path.
Warm it before capture so compilation and FP32 weight staging occur outside
capture. Missing compiler, unsupported shape/device or unsupported pure SD fail
explicitly. The `.hip` source is included in package data. MXFP4/MXFP8 MoE
packing and serving precision are unaffected; this shared BF16 attention
diagnostic does not substitute for full-model evaluation of both checkpoints.

The [validation receipt](../../tools/raon_validation/results/native-hip-rda-ar-20261127.json)
records source hashes and compact results. Raw artifacts remain at
`/mnt/hammerspace/jackhan/raon-hip-rda-20261127` and in the local
`.amd-validation/raon-hip-rda-20261127` directory. The next implementation work
is native SD with accepted-count rollback and every checkpoint, then mixed
prefill/SD dispatch. Gate fusion, mHC work and the MoE small-M schedule remain
listed in the [complete optimization budget](RDA-NATIVE-PORT-AND-BUDGET-20261127.md).
