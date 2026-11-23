# Raon-1 mini on ROCm

This fork adds the ROCm integration currently being evaluated for
`KRAFTON/Raon-1-mini-preview` on AMD Instinct MI355X, with TP1/DP1.
The vLLM base is `5df63a7203cdb9328d824a27b682e77c7b9928be`.
The Raon model implementation is derived from the
[KRAFTON fork](https://github.com/krafton-ai/vllm/tree/335a45e3f9d8a7f5f76c884734b5797ec9623fb3).
The upstream Apache-2.0 license and source attributions are retained.

## Integration

- RDA transforms the write gate and uses vLLM's Kimi AMD KDA kernels.
- MLA uses the Kimi AMD wrapper, AITER MLA decode and AITER prefill attention.
- Routed experts preserve the checkpoint's row-wise MXFP8 bytes and use
  HIP Triton `dot_scaled`, with native gfx950 scaled MFMA.
- mHC uses HIP-compatible floating-point primitives.
- DFlash2 shares cache pages across target MLA, RDA state and draft GQA.

The functional changes are in vLLM. AITER and FlyDSL are pinned dependencies;
this fork does not require modifications to their source or compiler.
The imported Raon package also retains source for other upstream execution
paths. The ROCm profile in `docker/raon/runtime-env.sh` selects the tested path.
No hipify step is used.

## Build

From the repository root:

```bash
docker build -f docker/Dockerfile.rocm-raon \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  -t hanjack/vllm-rocm:raon-rocm10.2 .
```

This uses the latest published ROCm vLLM image resolved on 2026-11-21,
pinned by digest, plus AITER `0.1.29.post1` and FlyDSL `0.5.0`.
The image tag is `rocm10.2.0_ubuntu24.04_py3.14-pytorch_2.14.0_vllm-0.32.0`;
the actual installed vLLM version must identify base commit `5df63a720`.
The build refuses an incompatible native wheel. It preserves Torch, Triton
and vLLM native extensions while installing the matching Python integration.

To reuse the immutable dependency stack used in the evaluation:

```bash
docker build -f docker/Dockerfile.rocm-raon-validated-base \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  -t hanjack/vllm-rocm:raon-evaluation-base .
```

The Dockerfiles use a virtual environment over the image's system packages.
Build evidence is retained under `/opt/raon/build-evidence` and
`/opt/raon-library-patch/build-evidence` when dependencies are rebuilt.
Image construction alone does not validate GPU execution or model quality.

Two built images are retained on Docker Hub. Both completed the four-task
GPU smoke check with SD off and DFlash2; this is not a full quality pass.
The long-generation DFlash2 failure described below remains unresolved.

| Stack | Image tag in `docker.io/hanjack/vllm-rocm` | Source commit |
| --- | --- | --- |
| Latest pinned dependencies | `rocm10.2-raon-680cfcee1-lateststack-20261121` | `680cfcee1` |
| Evaluation dependency stack | `rocm10.2-raon-a7653665e-evalbase-20261121` | `a7653665e` |

Immutable image digests and the scope of validation are recorded in
[`docker/raon/published-images.json`](docker/raon/published-images.json).

## Serve

```bash
docker run --rm --network=host \
  --device=/dev/kfd --device=/dev/dri --shm-size=8g \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v raon-aiter-cache:/var/cache/aiter \
  -v raon-triton-cache:/var/cache/triton \
  hanjack/vllm-rocm:raon-rocm10.2 \
  KRAFTON/Raon-1-mini-preview \
  --revision 1b4f494d71dff50318288309f64c72aa8c11340e \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 --data-parallel-size 1 \
  --max-model-len 262144 --max-num-batched-tokens 16384 \
  --max-num-seqs 16 --gpu-memory-utilization 0.80 \
  --enforce-eager --no-enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 --mamba-cache-mode none \
  --attention-config '{"backend_per_kind":{"mla_attention":"ROCM_AITER_MLA"},"mla_prefill_backend":"ROCM_AITER_FA"}' \
  --additional-config '{"kda_prefill_backend":"fused"}'
```

For rootless Podman, add `--group-add=keep-groups` and use its supported
device-access policy. Authenticate with Hugging Face first if model access
requires it. Keep access tokens outside the image and Git repository.

For experimental DFlash2 diagnostics, add the following. Long MMLU-Pro
generations have failed with repeated output; DFlash2 is not quality-validated.

```bash
--speculative-config '{"method":"dflash","model":"KRAFTON/Raon-1-mini-preview-dflash2","revision":"164b3dda020f145a4650b3c050841e15250c2f77","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

These are initial correctness settings. Prefix caching and CUDA graphs are
disabled, and this fork's initial target is the released MXFP8 checkpoint.

## Evaluation status

Reproducible RDA, accepted-state rollback, long-update, FP8 MLA, rejection
sampler and instrumented model diagnostics are registered under
[`tools/raon_validation`](tools/raon_validation/README.md). They retain
individual numerical errors and first-failure evidence separately from
quality evaluation and performance measurements.

On 2026-11-23, the MI355X component checks passed 8 RDA arithmetic cases,
30 rollback cases, 3,072 repeated state-update cases and 36 legacy/V2 sampler
cases. Full-model SD correctness is **not established**: native FP8 MLA Q2/H48
produced a NaN, and teacher-forced raw logits exceed the diagnostic tolerance.
The opt-in `VLLM_ROCM_AITER_MLA_PAD_H40_TO_64=1` avoids the unstable H48
dispatch in the tested captured-input replay; it does not resolve every SD
failure. A separate draft layer-25 NaN was traced to expired SWA cache
values contaminating the ROCm prefix-prefill value reduction. The prefix
kernel now skips/masks those loads; the actual captured-input replay and
eight regression cases pass. Long full-model validation after this fix and
the raw-logit discrepancy remain open. Numerical criteria are recorded with the
[diagnostic tools](tools/raon_validation/README.md). The Dockerfiles now include
the tools and this option; existing published image digests predate them.

Both SD off and DFlash2 completed the same four basic tasks with natural EOS:
multiplication, the first eight primes with explanation, Korean translation,
and continuation of a square-number sequence. The answers were correct.
Greedy token differences are recorded separately from task correctness.

The same four tasks are available in
`examples/offline_inference/raon_rocm_smoke.py`. Inside the image, run
`source /opt/raon/runtime-env.sh` and then
`/opt/raon/.venv/bin/python /opt/raon/smoke.py`.
Pass `--draft KRAFTON/Raon-1-mini-preview-dflash2` for DFlash2.
This checks natural termination; inspect the printed answers for correctness.

At 2026-11-21 00:41 UTC, SD off had completed 493 of 12,032 MMLU-Pro questions:
423 correct (85.80% on the completed subset), all with natural EOS and no
unparsed answers. This partial score is not the full benchmark result.
SD off continues across two-hour allocations with its original runtime.

DFlash2 evaluation was stopped after all 16 active questions began emitting
token ID 0 continuously for tens of thousands of tokens. Its 117 completed
questions (102 correct) do not establish a valid quality result. The saved
outputs are retained for diagnosis and are excluded from automatic continuation.
The cause has not been established; the short smoke checks did not catch it.
The failure was observed in the original immutable evaluation image, not a
full MMLU-Pro run of either newly built image. The published integration has
equivalent executable Python source, and neither image has a full quality pass.

The evaluation uses five-shot CoT,
thinking enabled, temperature 1, top-p 0.95, top-k 20 and presence penalty 1.5.
Each request can use the remaining portion of the 262,144-token context,
with `ignore_eos=False` and `min_tokens=0`. Completed questions are preserved
across two-hour allocations. Inflight outputs can resume from saved accepted
tokens by re-prefilling the prefix, with segment seeds and history recorded.
Such resumed samples are identified; numerical identity with uninterrupted
generation is not claimed.

The full 12,032-question score has not been established. The model card's
published MMLU-Pro score has an incompletely specified protocol and is
descriptive context rather than an exact reproduction target.

This is an experimental runtime-snapshot branch, not an upstream-ready PR.
The imported preview implementation still has upstream lint findings,
including optional-path imports, legacy API policy checks and unreachable
FlashKDA references in the ROCm adapter. Full upstream pre-commit is not
passing; Windows Node/npm bootstrap also encountered a certificate error.
The original Docker helpers pass Ruff and both Dockerfiles parse. Before the
2026-11-23 diagnostic changes, all 63 imported runtime Python files were
checked for functional AST equivalence with the evaluated container after
formatting and attribution comments. The current overlay also includes
`vllm/envs.py`, the MLA backend and the prefix-prefill kernel for the separately
tested fixes.

`docker/raon/runtime-source-provenance.json` records the hashes of the Python
files captured from the evaluated container before any repository formatting.
The active evaluation keeps its original immutable image and input sources.
