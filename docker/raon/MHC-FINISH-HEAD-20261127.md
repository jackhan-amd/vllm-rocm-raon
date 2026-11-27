# Native mHC finish/head fusion

`RAON_ROCM_MHC_FUSE_FINISH_HEAD=1` is an opt-in extension to the existing
native mHC chain. It requires `RAON_ROCM_MHC_BACKEND=hip`,
`RAON_ROCM_MHC_FINISHER=flydsl`, and `RAON_ROCM_MHC_GATE=flydsl`. Defaults are
unchanged. It applies to the shared BF16 mHC block; it changes neither MXFP4 nor
MXFP8 expert packing.

The AITER HIP projection produces 16 FP32 score/square partials. The candidate
reduces those partials inside the FlyDSL head/aggregate kernel, retaining the
original reduction order and the BF16 score rounding boundary. This removes
the separate FlyDSL finish launch and its compact score/mean intermediates.
The branch still separates head/aggregate from residual mix; no cross-branch
fusion or elimination of mHC math is claimed.

## What the CUDA names mean

`mhc.py` uses `Tensor.is_cuda` and `torch.cuda` for device and graph operations.
ROCm PyTorch exposes the same APIs for HIP. These names alone do not exclude AMD.
The inspected file contains no PTX, inline assembly or NVIDIA SM capability
guard. Its trainer-fused path invokes local Triton kernels and is separately
selected; it is not the active AITER HIP/FlyDSL path. The native path is
`RaonMHC.forward -> rocm_mhc_gate.forward -> AITER HIP project -> FlyDSL
head/aggregate -> branch -> FlyDSL mix` when this candidate is enabled.

## MI355X evidence

Thera job `89710`, node `TheraC80`, GPU 7, pinned ROCm image/runtime. GPU 7 was
reserved for mHC while the SD investigation used GPUs 0 through 6.

Real checkpoint layers 0/12/24, both attention and MLP mHC sites, rows
1/8/32/64/256 and input scales 0.125/1/8 passed all 90 independent Torch-gold
cases. The BF16 affine branch was multiply by 0.75 then add 0.125. Existing
1% RMS / 2% relative peak plus `1e-5` absolute limits were unchanged.
Changed-input graph replay, exact repeats and six in-place refit cases passed.
All 90 fused outputs were also bit-identical to the unfused native chain.

| Rows | Existing native block | Fused finish/head | Speedup |
| --- | ---: | ---: | ---: |
| 1 | 16.080 us | 14.389 us | 1.118x |
| 8 | 16.189 us | 15.058 us | 1.075x |
| 32 | 16.503 us | 15.545 us | 1.062x |
| 64 | 17.162 us | 16.680 us | 1.029x |
| 256 | 21.286 us | 19.941 us | 1.067x |

These are whole synthetic-branch mHC block GPU-event timings at the last tested
MLP site, not full-model tokens/sec or model accuracy. Each condition used three
direct warmups, capture, ten warmup replays and one window of 500 measured graph
replays. Compilation and the profiler were outside measurement. The profiler
witnessed one native `mhc_pre_gemm_sqrsum_kernel`, one `head_0.kd`, one
`mix_0.kd`, and the affine branch's two Torch operations; no separate finish
kernel remained. This is a modest site-level improvement, not evidence of a
twofold model gain. Further mHC tile changes were not included in this result.

The first JIT attempt required declaring branch outputs before the FlyDSL
conditional. A subsequent gold failure exposed an interface defect: square
partials are 2D, while the fused head uses flat indices. Passing
`mean.view(-1)` fixed the stride interpretation. The existing unfused native
chain passed the same 90-case gold, and the corrected candidate became exactly
equal to that chain. Initial failed results remain preserved separately.

## Reproduction

Inside the installed pinned runtime on an idle visible GPU:

```bash
uv run --no-project --python .venv/bin/python -m tools.raon_validation.mhc_math \
  --model /models/Raon-1-mini --backend hip --finisher flydsl --gate flydsl \
  --fuse-finish-head --branch affine --layers 0 12 24 \
  --rows 1 8 32 64 256 --scales 0.125 1 8 --timing-replays 500 \
  --profile evidence/mhc-finish-head-profile.json \
  --output evidence/mhc-finish-head.json
```

The [receipt](../../tools/raon_validation/results/mhc-finish-head-20261127.json)
records timings, source/artifact hashes and kernel names. Raw gold and profiler
artifacts are under
`/mnt/hammerspace/jackhan/raon-hip-rda-sd-20261127/mhc` and local
`.amd-validation/mhc-finish-head-20261127`.
