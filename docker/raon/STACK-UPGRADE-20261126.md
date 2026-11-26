# AITER/FlyDSL candidate, 2026-11-26

This candidate preserves the graph/projection runtime image and upgrades AITER
to commit `69753cf7f4323ef8ec511d1e3a5d273e818a7029` with its required FlyDSL
`0.5.3` wheel. Torch, ROCm, Triton, vLLM and checkpoint bytes stay fixed.
The historical Dockerfiles and pins remain reproducible baselines.

```bash
podman build --network=host \
  -f docker/Dockerfile.rocm-raon-stack-20261126 \
  -t raon-stack-candidate:aiter69753cf7-flydsl053 .
```

The installer uses the existing `/opt/raon/.venv`, installs with `uv` and
`--no-deps`, verifies the FlyDSL wheel hash and AITER Git commit, and records
versions and submodules under `/opt/raon/stack-upgrade/stack-evidence`.
Native AITER modules compile on first use against the image's ROCm/Python.
Use fresh writable runtime caches for this candidate; do not reuse compiled
AITER/FlyDSL modules from the baseline.

Status: published candidate with open numerical findings. Slurm job `88770`
completed on TheraC91, MI355X, TP1/DP1. The image is retained at:

```text
docker.io/hanjack/vllm-rocm:raon-rocm102-aiter69753cf7-flydsl053-candidate-20261126
docker.io/hanjack/vllm-rocm@sha256:9f94e15c16a2774e67ab7e841fd5e91a36f7e63c0d0a37b39227014c4cf1b537
```

The installed AITER distribution identifies itself as
`0.1.1.dev1+g69753cf7f` because the pinned source checkout has no release tags.
Use the Git commit as its identity, rather than interpreting this as an older
official release. FlyDSL is `0.5.3`.

## Speed screen

Output tokens/second; original MXFP8 checkpoint, GSM8K first 128 prompts,
temperature 1, top-p .95, top-k 20, thinking enabled, maximum 4096 new tokens,
natural EOS. Each condition was measured once after separate warmup; no JIT
cache changes or compilation warnings occurred during measurement. Separate
servers collected four bounded CPU/GPU traces at C=1/32, SD off/on.

| SD | Concurrency | Previous stack | Candidate | Change |
| --- | ---: | ---: | ---: | ---: |
| off | 1 | 75.56 | 75.45 | -0.14% |
| off | 8 | 386.88 | 403.36 | +4.26% |
| off | 32 | 878.53 | 747.22 | -14.95% |
| on | 1 | 191.47 | 195.45 | +2.08% |
| on | 8 | 899.49 | 920.54 | +2.34% |
| on | 32 | 1451.56 | 1627.45 | +12.12% |

The previous measurements ran on TheraC81, job `88723`. These are descriptive
single-run comparisons: concurrent stochastic decoding changed output lengths.
For example, SD-off C=32 wall time stayed about 80.8 vs 80.6 seconds, while total
output changed from 70,997 to 60,218 tokens. Do not attribute its TPS difference
directly to kernel latency. At SD-off C=1 all 128 texts, output lengths and finish
reasons matched the previous stack exactly. This short-prompt screen does not
reproduce the fixed-ISL/OSL workload with measured 50% prefix-cache hits.

## Numerical findings

Projection, RDA and MXFP8 MoE fixtures passed. MXFP4 eager-vs-gold and graph
numeric tolerance checks passed, but exact repeat/replay checks failed; its
overall fixture status remains FAIL. This is an adapter fixture, not a full
MXFP4 model evaluation.

Native MLA exceeded the unchanged 2% FP32 relative-RMS threshold on both stacks
(roughly 2.1% on the failing cases). This is still an open numerical finding.
With identical saved inputs/references, actual outputs across the real 40 heads
were identical in 11 of 12 cases; the remaining case differed in 4/40,960 BF16
elements by at most one BF16 step (maximum absolute difference 0.00006103515625).
Stack agreement does not turn either FP32-gold failure into a pass. No full
model accuracy score was measured in this upgrade screen.

## Profiling evidence and limits

TraceLens identified graph-under-recorded traces: graph-replay fractions were
87.3% / 81.2% for SD off/on at C=1. CSV/XLSX export completed but both wrapper
shells returned exit 1 without a final traceback. Full agent analysis stopped;
deterministic fallback reports and a direct GPU-kernel inventory were retained.
Operator shapes, roofline efficiency and CPU critical-path attribution are not
established. The following shares use the **sum of observed GPU kernel
durations**, not end-to-end latency:

| Kernel | SD off | SD on |
| --- | ---: | ---: |
| `matmul_kernel_persistent.kd` | 30.21% | 24.01% |
| `__amd_rocclr_copyBuffer.kd` | 7.28% | 5.06% |
| BF16-to-FP32 copy kernel | 5.68% | 4.26% |
| `hgemm_bf16_skinny_t8x64x128x2_w1x1x4_0.kd` | 5.65% | 8.30% |

These first 32 profiled engine steps include prefill and are not a full-request
decode-only profile. The persistent GEMM is a likely mHC hotspot: `mhc.py` uses
the fixed-schedule Triton implementation with `RAON_BATCH_INVARIANT` defaulting
to 1. This switch differs from `VLLM_BATCH_INVARIANT=0` and the disabled optional
MLP/MLA projection persistent flags. Graph capture metadata is needed to confirm
per-call attribution before replacing it with a native kernel. Preserve its
batch-invariance contract and verify Torch gold when testing a replacement.
Copies, conversions and indexing are the next candidates to localize. These
observations do not establish a threefold speedup opportunity.

TraceLens source was `c80f7de688c485fd6924da9a4de345a785bac60e`. A local copied
analysis tool needed a Windows `csv.field_size_limit` portability fix. The
MI355X extension used public theoretical peaks, not measured achievable rates;
no roofline result is accepted from this degraded analysis. User TraceLens
source and runtime math were not changed. Compile experiments remain deferred.

Machine-readable measurements, image identity and numerical findings are in
[`stack-upgrade-20261126.json`](../../tools/raon_validation/results/stack-upgrade-20261126.json).
Raw results and all four traces remain under
`/home/jackhan/bench-results/raon-stack-upgrade-20261126-r2` on Thera. Both jobs
completed with Slurm exit `0:0`; owned containers were removed and GPUs released.

The Kimi sparse-MLA flags do not automatically apply to Raon's dense MLA. Inspect
the upgraded libraries' actual dispatch and retain native-kernel witnesses.
