# Native HIP V-split SD candidate and crossover

`rocm_hip_rda_sd_vsplit.hip` uses three native HIP operations: shared convolution
and Q/K-normalization/decay prologue, V-split recurrence, and gated output norm.
The prologue computes common gates once; eight independent 16-V-row CTAs per
head consume them. At batch1 the recurrence grid grows from 40 to 320 CTAs on
the actual 256-CU MI355X. It holds two float4 vectors of FP32 state per thread
across tokens, writes every candidate checkpoint and preserves the BF16 seams.
This is an AMD HIP implementation, without hipify or Triton compute.

Each prologue CTA handles one sequence/head and processes its token groups in
parallel, with 128 threads per candidate. A barrier completes all old-history
reads before any convolution-cache write, avoiding inter-CTA cache races. The
prologue also initializes the intermediate output, so the entry has three
compute launches and no separate fill launch. State/checkpoint ownership is
disjoint across V tiles. The final norm reads all 128 V values per head.

## Validation and adapter timings

Job89711, TheraC80 GPU0. Six batches 1/2/4/8/16/32 passed 102 finite gold cases:
independent Torch output/state, complete conv histories and untouched state,
AR capture/replay plus SD lengths2/4/8 at every accepted count, and exact
graph/direct state/output/cache checks. The separate witness run passed and
recorded shared prologue, V-split recurrence and norm kernels. The selected
module is `c8e21e49c71aeb3b1409`. This snapshot test uses a private loader;
production selector and mixed/ragged integration are validated separately.

| Batch | Verify length | FlyDSL adapter (us) | V-split HIP adapter (us) | Speedup |
| --- | --- | ---: | ---: | ---: |
| 1 | 2 | 19.180 | 8.935 | 2.15x |
| 1 | 4 | 24.846 | 11.517 | 2.16x |
| 1 | 8 | 36.579 | 16.125 | 2.27x |
| 32 | 2 | 88.143 | 56.657 | 1.56x |
| 32 | 4 | 163.234 | 103.307 | 1.58x |
| 32 | 8 | 318.284 | 204.488 | 1.56x |

Each condition used one event window after separate compile and warmup:
32 operations/graph, 20 warmup replays and 100 measured replays. State updates
only touch disposable synthetic fixtures. Witness profiling is separate.
These numbers exclude projection/ratio and do not establish model TPS,
accuracy or full-model improvement. Batch32 is slower than the separately
measured [single-kernel candidate](NATIVE-HIP-RDA-SD-FUSED-20261127.md): at
length8, 204.488 us versus 148.161 us. V-split is therefore not a universal win.

## Direct native-entry crossover

The same fixed synthetic input/initial-state fixture was used for each pair
below. Both native C++ entries were captured directly, with the same 32/20/100
graph procedure and one event window each. There is no repeated FlyDSL timing
or duplicate numerical gold sweep here. Outputs were checked finite after the
window; the preceding 102-case sweeps establish numerical coverage separately.

| Batch | Steps | One-kernel HIP (us) | V-split HIP (us) | V-split faster by over 5% |
| --- | --- | ---: | ---: | --- |
| 2 | 2 | 9.319 | 10.164 | No |
| 4 | 2 | 10.385 | 12.111 | No |
| 8 | 2 | 14.047 | 20.755 | No |
| 16 | 2 | 25.793 | 31.822 | No |
| 2 | 4 | 16.155 | 13.293 | Yes |
| 4 | 4 | 16.186 | 15.768 | No |
| 8 | 4 | 22.557 | 30.776 | No |
| 16 | 4 | 48.200 | 53.948 | No |
| 2 | 8 | 28.193 | 18.955 | Yes |
| 4 | 8 | 28.053 | 26.076 | Yes |
| 8 | 8 | 39.599 | 48.722 | No |
| 16 | 8 | 87.678 | 102.355 | No |

Combining these exact pairs with batch1 adapter evidence gives a conservative
experimental V-split choice set `(batch, steps)={(1,4),(1,8),(2,4),(2,8),(4,8)}`.
Do not interpolate to unmeasured batches or ragged query lengths. Retain the
single-kernel native candidate for other unmeasured shapes. This is a proposed
opt-in dispatch table, not authorization to alter an active serving profile.
One-window comparisons do not establish statistical confidence; the 5% margin
avoids choosing the roughly 2.6% batch4/steps4 difference.

## Compiler resources and interpretation

| Kernel | VGPR | SGPR | LDS bytes | Private bytes | VGPR/SGPR spills |
| --- | ---: | ---: | ---: | ---: | --- |
| Common prologue, steps2/4/8 | 48 | 50 | 144/288/576 | 0 | 0/0 |
| V-split recurrence | 47 | 34 | 0 | 0 | 0/0 |
| Norm | 11 | 26 | 272 | 0 | 0/0 |

The prologue launches 256/512/1024 threads for steps2/4/8; recurrence and norm
use256. Recurrence has 8x the CTAs of the single-kernel candidate and lower
VGPR use, but the old candidate also had zero spill. Call this a scheduling
experiment, not a spill fix. The new FP32 common buffer costs 640 KiB per
eight-token request and is consumed by V tiles. Its traffic, cache behavior and
the larger prologue blocks are plausible contributors to the batch32 loss;
hardware counters have not separated them. No measured CU-busy or occupancy
claim follows from metadata alone.

The [receipt](../../tools/raon_validation/results/native-hip-rda-sd-vsplit-20261127.json)
contains exact hashes, gold, witness, native-entry measurements and resource
metadata. Raw artifacts are under
`/mnt/hammerspace/jackhan/raon-native-mixed-20261127/{sd-vsplit,sd-crossover}`
and local `.amd-validation/sd-vsplit-20261127`. Standalone tests do not establish
MXFP4/MXFP8 model quality, arbitrary ragged/NULL suffix behavior or shared-layer
mixed-batch integration. Keep the candidate opt-in pending those checks.
