# Opt-in native Raon stack (2026-11-28)

This candidate removes intermediate attention-projection/mHC, MoE/mHC and
RDA write-projection/gate-ratio launches. The model weights and precision are
unchanged. It also repairs SD's logical-token/padded-buffer boundary. The
accepted baseline and its image digest remain unchanged.

The deployment source is frozen `77f28c0949`, SD overlay
`3a5543b0ec1a677dd8f1a6331bdb14f506f559f6b17ddec1298474d8b4aebd40`,
and the 13-file candidate recorded in
`tools/raon_validation/results/native-stack-20261128/tested-source-sha256.json`.
Those files are kept byte-for-byte as tested, including their experimental
formatting. The existing dirty development checkout was not used as a build
context. No AITER or FlyDSL compiler change is required by this stack.

## Scope and activation

* Attention output projection + mHC: native AITER/FlyDSL MFMA epilogue,
  at most 32 actual rows; original BF16 output rounding is retained.
* MoE combine + shared output + steering + mHC: one native FlyDSL kernel,
  at most 128 actual rows. Native expert GEMMs and routing stay unchanged;
  shared computation occurs once. This new boundary is MXFP8-only.
* RDA write projection + gate ratio: one native AITER-derived FlyDSL kernel
  per 32-row chunk, at most 64 actual rows. It preserves the BF16 delta seam
  and FP32 ratio and never updates the recurrent cache.
* Dense BF16 and MXFP8 MoE use one common actual-shape policy across C=1/8/32.
  No per-concurrency environment profile is selected.
* These are local fusion boundaries, not a persistent whole-model monokernel.
  The MXFP4 model and the later preshuffle/large-M experiment are not certified
  by the results in this directory.

Use `runtime-native-fusion-env.sh` through `run-native-fusion.sh`; it adds the
tested geometry worker extension. The opt-in flags include
`RAON_ROCM_ATTN_PROJECT_MHC=flydsl`, `RAON_ROCM_MOE_MHC_FINISH=flydsl`, and
`RAON_ROCM_RDA_PROJECT_RATIO=flydsl`. Their defaults in the model modules stay
legacy. `RAON_KDA_DECODE_GRAPH`, `RAON_MOE_OUTER_GRAPH`, and
`RAON_MOE_SHARED_SIDE_STREAM` are all enabled in the reviewed native profile.
The image is built with `Dockerfile.native-stack`. Its build checks every
tested overlay and writes a complete installed-model source manifest.

The server/client settings remain the reviewed TP1/DP1 profile: dtype auto,
FP8 E4M3 KV, prefix caching, max length 262144, batched tokens 16384,
max sequences 128, memory utilization .85, Raon reasoning/tool parsers, thinking
enabled and history-thinking truncation disabled. SD uses DFlash2 with seven
draft tokens and probabilistic proposal/standard rejection sampling. No eager
fallback is enabled. Eight replicas use independent GPUs, never native DP8/EP.
Exact executed server argv and workload contracts are in the evidence packet.

## Validation and measured limits

* SD boundary repair: 48 native/Torch cases and 192 direct/replay records,
  including changed accepted counts, indices and logical lengths. Unpadded
  native output/state/conv agree with repaired padded execution; padding and
  state-stride sentinels are checked.
* MoE/mHC finish: 36 native-chain/Torch cases and 108 changed-input graph
  replays. BF16 seams, stable expert order and optional steering are covered.
* RDA projection/ratio: 88 real-checkpoint-weight cases and 264 changed-input
  graph replays; maximum relative RMS versus Torch FP32 equations is
  `1.1686859029893618e-6`. This is an operator check, not full-model quality.
* Job 90043: all 30 single measurements valid, after separate warmup and
  outside profiler windows. Eight additional GPU traces prove actual native
  kernel activation. No valid performance point was repeated.
* 14/30 cells reach 90% of the published B300 reference. GSM8K AR C=1 reaches
  272.07 output tokens/s; the prior projection candidate was 217.60. C=1 SD
  HumanEval/MT-Bench remain below target. This is not an acceptance of all cells.
* The current model is final Raon-mini; the
  [published reference](https://huggingface.co/KRAFTON/Raon-1-mini-preview#speculative-decoding)
  uses preview. Cross-checkpoint ratios are descriptive, not hardware rankings.
* Prior optimized source completed MMLU-Pro: 10029/12032 = 83.3527%, versus
  frozen baseline 10021/12032. That score excludes the new MoE finish,
  geometry and RDA ratio fusion. Fresh full-set job 90045 evaluates this
  candidate separately: 9989/12032 = 83.0203%, all12032 natural EOS, zero
  unparsed answers, audit/cleanup passed. The observed change is -0.3324pp
  against the prior optimized source and -0.2660pp against the original frozen
  baseline. Paired questions gained344/lost384; the descriptive paired95%
  interval is [-0.7719,+0.1070]pp (not repeated-run variance). This neither
  proves equivalence nor establishes a regression. No automatic quality
  noninferiority decision is made.

The separately preserved image is
`docker.io/hanjack/vllm-rocm:raon-mini-mxfp8-native-fusion-20261128-rc1`,
digest `sha256:acb0ae5d9c477c237b726e3949e8b74aa6465a94afb28d62af9bb05cd32d56a8`.
It checks19 tested source files at build time and records88 installed model
files. Source label `77f28c0949-native-164e2fa16bce` identifies the frozen
commit and candidate archive; it is not a claim that the candidate was part of
that frozen commit. The experimental mHC zero-low derivative is excluded.

The fixed ISL/OSL workload with a measured50% token-based prefix hit ratio is
separate from these public natural-prompt tasks. Prefix caching is enabled;
the actual hit fractions and output lengths are recorded, not forced to50%.

See `speed30.json`, `gpu-traces8.json`, `workload-effects.json`, and the raw
evidence packet under `tools/raon_validation/results/native-stack-20261128/`.
Kernel-time shares are sums of device events and may overlap streams; they
are not wall-time percentages. Actual API request decode time per C=1 SD
iteration is about 6.63 to 6.77 ms across all five datasets. Weighted AL varies
substantially and must be kept separate from mean per-request AL.

## AL diagnosis

RNG/checkpoint controls, exact-prefix SD/AR/fresh scoring, and native RDA
variants are diagnostic runs, not throughput or accuracy measurements.
Reconstructing processed top-k/top-p probabilities agrees with 189 saved live
verification rows to `1.2e-7`; replacing SD target scores with incremental AR
scores has no consistent overlap gain on these realized prefixes. This is not
a counterfactual AL prediction or full target hidden-state parity.

Independent RNG families do not explain the gap. Coupled draft/target noise
fails an independent output-marginal Monte Carlo test and is not deployable.
The published NVIDIA runtime commit is unknown; this result does not establish
which sampler produced the public table. K32/K64 draft proposals and different
native RDA SD kernels show no useful HumanEval AL gain on paired full prompts.
KV BF16 was rejected by the existing qualified cache-page alignment guard;
the guard was retained and no BF16-KV inference result is claimed.
