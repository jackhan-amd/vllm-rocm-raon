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
