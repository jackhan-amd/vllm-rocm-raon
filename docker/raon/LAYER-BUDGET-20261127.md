# Layer timing inside the warm target graph (2026-11-27)

MXFP8, TP1/DP1, MI355X. Existing post-warmup activity was analyzed on CPU;
no GPU allocation, runtime change or additional benchmark was performed.

**These are instrumented ROCprof times, not unprofiled timings.** Matching
SD-off windows previously showed 38% / 17% whole-window tracing overhead
at C1 / C32. No overhead correction or proportional scaling is applied.
The graph-envelope means below are not the prior low-event 4.36 / 8.16 ms
target-graph timings. Profiler cost can affect small kernels differently.

Layer numbers are one-based; checkpoint indices are one less. Layer 1 has
a Dense FFN and all other layers have MoE. Asterisks identify the five
reasoning-effort steering layers. Each attention/MLP site includes its
own mHC wrapper, input normalization and projection work.

## SD-off layer distribution

Each entry averages 64 target FULL graphs. Units: milliseconds per step.

| Layer | Attention | C1 attention site | C1 MLP site | C1 total | C1 graph share | C32 attention site | C32 MLP site | C32 total | C32 graph share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | RDA | 0.142 | 0.090 | 0.232 | 3.85% | 0.188 | 0.092 | 0.280 | 2.93% |
| 2 | RDA | 0.138 | 0.101 | 0.239 | 3.97% | 0.186 | 0.219 | 0.405 | 4.24% |
| 3 | RDA | 0.140 | 0.101 | 0.241 | 3.99% | 0.188 | 0.226 | 0.414 | 4.33% |
| 4 | MLA | 0.120 | 0.104 | 0.223 | 3.70% | 0.129 | 0.231 | 0.360 | 3.77% |
| 5 | RDA | 0.142 | 0.103 | 0.245 | 4.07% | 0.189 | 0.228 | 0.417 | 4.36% |
| 6 | RDA | 0.140 | 0.100 | 0.240 | 3.98% | 0.186 | 0.210 | 0.396 | 4.15% |
| 7 | RDA | 0.140 | 0.099 | 0.239 | 3.97% | 0.187 | 0.213 | 0.400 | 4.19% |
| 8 | MLA | 0.119 | 0.100 | 0.219 | 3.63% | 0.130 | 0.208 | 0.338 | 3.54% |
| 9 | RDA | 0.141 | 0.100 | 0.240 | 3.99% | 0.187 | 0.207 | 0.395 | 4.13% |
| 10 | RDA | 0.141 | 0.099 | 0.240 | 3.98% | 0.187 | 0.205 | 0.392 | 4.10% |
| 11 | RDA | 0.140 | 0.099 | 0.239 | 3.97% | 0.186 | 0.193 | 0.379 | 3.97% |
| 12 | MLA | 0.119 | 0.099 | 0.218 | 3.62% | 0.128 | 0.190 | 0.319 | 3.34% |
| 13* | RDA | 0.141 | 0.129 | 0.269 | 4.47% | 0.187 | 0.228 | 0.415 | 4.34% |
| 14 | RDA | 0.141 | 0.100 | 0.240 | 3.98% | 0.186 | 0.195 | 0.381 | 3.99% |
| 15* | RDA | 0.141 | 0.129 | 0.270 | 4.47% | 0.186 | 0.232 | 0.419 | 4.38% |
| 16 | MLA | 0.119 | 0.099 | 0.218 | 3.62% | 0.130 | 0.196 | 0.326 | 3.41% |
| 17* | RDA | 0.141 | 0.129 | 0.270 | 4.47% | 0.187 | 0.248 | 0.435 | 4.55% |
| 18 | RDA | 0.141 | 0.100 | 0.241 | 3.99% | 0.186 | 0.208 | 0.395 | 4.13% |
| 19* | RDA | 0.140 | 0.129 | 0.269 | 4.46% | 0.186 | 0.243 | 0.430 | 4.49% |
| 20 | MLA | 0.119 | 0.100 | 0.219 | 3.63% | 0.130 | 0.197 | 0.327 | 3.42% |
| 21* | RDA | 0.141 | 0.129 | 0.270 | 4.48% | 0.187 | 0.236 | 0.423 | 4.42% |
| 22 | RDA | 0.140 | 0.100 | 0.240 | 3.98% | 0.186 | 0.206 | 0.392 | 4.10% |
| 23 | RDA | 0.141 | 0.100 | 0.241 | 3.99% | 0.186 | 0.214 | 0.399 | 4.18% |
| 24 | MLA | 0.119 | 0.099 | 0.218 | 3.62% | 0.128 | 0.220 | 0.348 | 3.64% |
| 25 | MLA | 0.118 | 0.100 | 0.218 | 3.62% | 0.129 | 0.214 | 0.343 | 3.59% |

## Additive site budget

| Region | C1 ms | C32 ms |
| --- | ---: | ---: |
| Attention sites (18 RDA + 7 MLA) | 3.3624 | 4.2675 |
| MLP sites (1 Dense + 24 MoE) | 2.6373 | 5.2593 |
| Embedding / graph prefix | 0.0145 | 0.0154 |
| Final reduction / norm / graph suffix | 0.0154 | 0.0164 |
| Total target graph envelope | 6.0297 | 9.5587 |

## Repeated fusion candidates and existing mHC cost

The following kernel unions are subsets of the site budget above; do not
add them again. Their duration is not an achievable saving. Asynchronous
work can overlap other categories. No replacement kernel was measured.

| Subset | Calls per graph | C1 ms | C1 graph share | C32 ms | C32 graph share |
| --- | ---: | ---: | ---: | ---: | ---: |
| RDA ratio: cast/cast/add/logsigmoid/logsigmoid/sub/exp | 126 | 0.6005 | 9.96% | 0.6050 | 6.33% |
| RDA BF16 output fills | 36 | 0.1748 | 2.90% | 0.1807 | 1.89% |
| Reasoning-effort steering chain | 30 | 0.1468 | 2.43% | 0.1475 | 1.54% |
| Existing mHC project/finish/head/mix | 200 | 1.0037 | 16.65% | 1.0232 | 10.70% |

The subtraction is emitted as a second CUDAFunctor_add<float> kernel.
All 18 ratio chains match the seven-operation sequence before native RDA
decode in every included target graph. The five steering layers each
have six separate kernels immediately before MLP mHC mix: compare,
BF16-to-FP32 cast, multiply, add, cast back and where. This agrees with
`compute_rda_value_ratio()` and `effort_steering.steer()` source. The C1
steering-chain total is 0.1468 ms, explaining the approximately 0.03 ms
extra in each of those layers. Preserve alpha-zero bit selection and
the BF16 rounding boundary when considering fusion.

## SD-on target FULL-graph subset

Only 62/64 C1 and 39/64 C32 target steps were single FULL graph launches.
The remaining steps have fragmented dispatch and are excluded here.
This is not a complete SD-on step average, an on/off speed comparison,
or milliseconds per output token. Draft execution is not included.

| Layer | Attention | C1 target ms | C32 target ms |
| --- | --- | ---: | ---: |
| 1 | RDA | 0.262 | 0.605 |
| 2 | RDA | 0.329 | 0.854 |
| 3 | RDA | 0.311 | 0.863 |
| 4 | MLA | 0.279 | 0.565 |
| 5 | RDA | 0.313 | 0.865 |
| 6 | RDA | 0.303 | 0.847 |
| 7 | RDA | 0.304 | 0.841 |
| 8 | MLA | 0.267 | 0.538 |
| 9 | RDA | 0.298 | 0.837 |
| 10 | RDA | 0.296 | 0.827 |
| 11 | RDA | 0.290 | 0.820 |
| 12 | MLA | 0.265 | 0.530 |
| 13 | RDA | 0.326 | 0.851 |
| 14 | RDA | 0.290 | 0.817 |
| 15 | RDA | 0.323 | 0.853 |
| 16 | MLA | 0.264 | 0.528 |
| 17 | RDA | 0.326 | 0.854 |
| 18 | RDA | 0.289 | 0.810 |
| 19 | RDA | 0.316 | 0.853 |
| 20 | MLA | 0.262 | 0.532 |
| 21 | RDA | 0.324 | 0.847 |
| 22 | RDA | 0.293 | 0.834 |
| 23 | RDA | 0.295 | 0.834 |
| 24 | MLA | 0.267 | 0.539 |
| 25 | MLA | 0.262 | 0.542 |

## Attribution and validation

Target graph ownership requires a HIP graph-launch correlation ID inside
a recorded FULL-graph CPU scope and 24 MoE stage1 kernels. Every included
graph must contain 50 native mHC projection starts, 50 finish/head/mix
kernels, 18 recurrence and 7 MLA attention kernels in the checkpoint order,
and 24 correctly positioned MoE stage1/stage2 pairs. All checks passed.

Consecutive mHC starts partition attention and MLP sites in layer order.
The last layer ends at its final mix; prefix and suffix remain separate.
No kernel assigned to a layer may extend beyond its ending boundary.
For every graph, prefix + 25 layer spans + suffix exactly reconstructs
the first-to-last target kernel envelope in integer nanoseconds.
Layer spans include graph-internal gaps; they are not sums of overlapping
kernel times or measurements of CU utilization. Classification uses
source order and checked native markers rather than Torch replay stacks.

- Source: `216281f592`; target snapshot `3f85cd63ea420100a84378f72ae162af86a441fd`.
- [Raw per-layer result](../../tools/raon_validation/results/layer-budget-20261127.json).
- [Warmup and overhead evidence](GAP-BUDGET-20261127.md).
- Raw SDK CSVs and CPU analysis script on Thera:
  `/mnt/hammerspace/jackhan/raon-gap-budget-20261127/layer-budget.py`.
- Remote result: `analysis/layer-budget.json` under that same directory.
