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

Status: candidate, GPU validation and performance results pending. A successful
image build is not numerical or performance acceptance. The submitted experiment
checks projection/RDA/MXFP8/MXFP4 graph numerics, then measures the existing native
MXFP8 SD-off/on GSM8K first-128 workload at concurrency 1/8/32 once after warmup.
Separate servers collect bounded CPU/GPU traces; their timings are diagnostic.
The running persistent-kernel comparison is unchanged.

The Kimi sparse-MLA flags do not automatically apply to Raon's dense MLA. Inspect
the upgraded libraries' actual dispatch and retain native-kernel witnesses.
