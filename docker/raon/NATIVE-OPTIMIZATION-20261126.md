# Native Raon optimization screen — 2026-11-26

MI355X, one independent TP1/DP1 replica per condition; original release MXFP8,
SD off/on, 128 GSM8K prompts at C=1/8/32. Each point was measured once after
separate warmup. Output tokens/s only; EOS respected, cap 4096, T=1, top-p=.95,
top-k=20, thinking enabled. All 30 points passed token-total and compiler-clean
checks. Full-set accuracy and the separate long-ISL/50%-prefix workload were not run.

| SD | C | Baseline | Native final | Ratio |
| --- | --- | ---: | ---: | ---: |
| off | 1 | 75.6 | 146.9 | 1.94x |
| off | 8 | 386.6 | 698.5 | 1.81x |
| off | 32 | 877.7 | 1309.1 | 1.49x |
| on | 1 | 189.7 | 277.3 | 1.46x |
| on | 8 | 946.5 | 1306.9 | 1.38x |
| on | 32 | 1599.0 | 1902.8 | 1.19x |

The final candidate combines native HIP mHC GEMM/statistics, ordered FlyDSL
MoE reduction, direct enclosing-graph MoE capture and narrowed live MLA attention
callbacks. The gate, expert probabilities and checkpoint precision are retained.
The existing native Kimi/AITER attention kernel and KV owner remain in use.

The component image checks passed: real-weight mHC 126 cases plus 6 refits,
ordered combine 42, MoE direct graph 6, MLA partition-only fixture 9. HIP mHC
is numerically within Torch-gold limits but has 120/126 exact batch-identity
cases; this does not establish universal bit parity. The FlyDSL mHC alternative
has 126/126 in that finite fixture. The MLA partition fixture substitutes
attention math and is not an additional native attention-kernel gold.

All new switches remain opt-in:

```bash
export RAON_RUNTIME_PROFILE=mxfp8-flydsl
export RAON_ROCM_MHC_BACKEND=hip
export RAON_MOE_COMBINE_UNW_FUSED=flydsl
export RAON_ROCM_MOE_DIRECT_GRAPH=1
export RAON_ROCM_MLA_SPLIT_GRAPH=1
export VLLM_USE_V2_MODEL_RUNNER=1
```

Use the existing reviewed server arguments; source the FlyDSL runtime profile.
This does not enable Torch.compile or eager execution. Do not apply the MXFP8
direct-MoE switch to the MXFP4 native runner. Intermediate combine-only SD-on
results regressed at C=8, so isolated-kernel speed is not a universal recommendation.

Published image: `docker.io/hanjack/vllm-rocm:raon-native-3ce9e8d1ce-20261126`

`sha256:74a98d145aecfc637b3a269e4c8d896b2ac842a5fd26fc3a95af4f3e95c4b6db`

Source: `3ce9e8d1ce2ac363369031c735ed44c5b9ac3f0e`. The published image passed independent
component checks; its generation smoke is retained with the raw artifacts.
Performance used working-tree overlays before non-math formatting/type cleanup.

See [machine-readable evidence](../../tools/raon_validation/results/native-paths-20261126.json)
for all intermediate points, token lengths, acceptance lengths, commands and limits.
Raw requests, profiler traces and failed preparation attempts remain under
`/home/jackhan/bench-results/raon-manual-opt-20261126`.

Next targets: the remaining native MLA-attention callback and KDA replay metadata/
copies. Do not remove their live-metadata refresh or repeat live state updates
during graph warmup. The 32-step traces include profiler overhead and prefill;
their inclusive CPU times and GPU kernel-duration sums are not E2E critical paths.

## Precision audit and the next native-kernel gap

The target checkpoint is MXFP8, but that does not describe every runtime-internal
projection. This runtime defaults `VLLM_ROCM_USE_AITER_FP4BMM=True` on CDNA4.
For BF16 `kv_b_proj` weights, `MLAAttention` quantizes the absorbed `W_K` and
`W_V` matrices to MXFP4 during preparation. Both were actually observed calling
AITER's `batched_gemm_a16wfp4` in the final SD-off/on traces.

Despite the A16 API name, the inspected AITER device kernel calls
`_mxfp4_quant_op` on its BF16 activation tile and then
`tl.dot_scaled(..., "e2m1", ..., "e2m1")`. These absorbed projections therefore
use internal MXFP4 activations and weights, FP32 accumulation and BF16 output.
The MLA attention core still uses the native AITER HIP implementation and FP8 KV;
this is a separate precision boundary. All measured profiles retained the same
behavior, so the relative speed comparison did not introduce this quantization.
This screen did not measure the accuracy impact of the inherited BMM default.

The pinned AITER tree has a gfx950 FlyDSL A8W8 batched API, a dense A16W4 API
without activation quantization, and a gfx1250-only batched A8W4 API. None is a
drop-in gfx950 replacement for the observed A4W4 operation. Preserve the current
E2M1/E8M0 packing, activation quantization and output scaling when building that
replacement; do not silently switch to A8 activations or unquantized weights.

Relevant inspected paths are `vllm/model_executor/layers/attention/mla_attention.py`
(`is_aiter_triton_fp4_bmm_enabled`, weight preparation and `_v_up_proj`) and the
pinned AITER `ops/triton/_triton_kernels/gemm/batched/batched_gemm_a16wfp4.py`.
The residual SD-only Triton callers also include DFlash2 convolution, draft
attention/cache update and rejection sampling. These are separate from Raon's
native RDA decode and its native MoE GEMMs.

## Follow-on native mHC finisher

Commit `f6319bb27c` adds `RAON_ROCM_MHC_FINISHER=flydsl` to fuse split
reductions, the BF16 score cast and mean normalization after the HIP GEMM.
It remains opt-in. The full real-weight block gold passed 126 cases and 6
refits. With the same one-trial speed workload, the follow-on results are:

| SD | C | Prior native | With finisher | Ratio |
| --- | --- | ---: | ---: | ---: |
| off | 1 | 146.9 | 160.9 | 1.10x |
| off | 8 | 698.5 | 701.6 | 1.00x |
| off | 32 | 1309.1 | 1439.0 | 1.10x |
| on | 1 | 277.3 | 293.0 | 1.06x |
| on | 8 | 1306.9 | 1299.2 | 0.99x |
| on | 32 | 1902.8 | 2407.0 | 1.26x |

This is another incremental candidate, not an additional twofold gain. Different
natural lengths and acceptance remain a limitation, especially for small ratios.
See [follow-on evidence](../../tools/raon_validation/results/native-finisher-20261126.json)
and the [residual SD-on Triton callers](../../tools/raon_validation/results/mhc-hip-combine-split-profile-r1-on-triton-callers.json).

The follow-on image is
`docker.io/hanjack/vllm-rocm:raon-native-f6319bb27c-20261126`, digest
`sha256:21ae335dd3d6aa57d0d75b5ee044c43336019d3ee726e5966c9eba54c729a329`.
Its [receipt](image-native-finisher-20261126.json) records exact source hashes,
fresh component gold and three natural-EOS generation smokes for each SD mode.
Enable the finisher explicitly; the image retains the old default.

The next low-scope serving candidate is the DFlash2 draft's native attention
backend: the recorded run used `ROCM_ATTN` Triton attention/cache helpers even
though target MLA used native AITER. The runtime's per-kind overrides are
`full_attention` and `sliding_window`; `ROCM_AITER_FA` advertises non-causal and
FP8-cache support. That capability check alone is not validation of this draft's
exact masking/cache contract. Check draft attention against Torch gold and SD
cache replay before measuring this as a separately reviewed configuration.
