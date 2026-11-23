# Raon ROCm correctness diagnostics

Run inside the pinned Raon ROCm environment with one visible GPU. Source
`docker/raon/runtime-env.sh` before starting Python. Preserve the image's
`PYTHONPATH`: the ROCm SDK places AMD-SMI there. The tools now refuse an
unrecognized ROCm platform. Use a virtual environment with the installed
Torch/vLLM/native extensions available; no model download is performed.

| Command | Independent check |
| --- | --- |
| `core` | RDA recurrent and fused prefill output/final state versus direct FP32 erase/write equations; lengths 1, 8, 65, 257 |
| `rollback` | Every previous accepted count for draft lengths 1, 3, 7; convolution history, candidate recurrent states, reordered batches 16/15/8/1 and untouched cache rows |
| `stress` | Repeat the rollback checks for 1,024 rounds per draft length by default |
| `mla` | FP8 MLA query lengths 1/2/4/8, shuffled KV indices and causal masks against dense FP32 attention of the same dequantized inputs |
| `sampler` | Legacy and actual V2 rejection samplers, including standard/block verification; every greedy rejection position and bonus token; IID two-token joint distributions including equal/disjoint supports and zero probabilities |
| `sampler-gold` | V2 accept/reject length and emitted tokens versus independent Torch probability-space math with identical supplied random draws |
| `model` | Real eager inference with finite-value checks on live RDA inputs/output, decoder output, target/draft logits and rejection-sampler inputs; optional greedy prefix comparison |
| `teacher` | Fixed accepted prefix, full vocabulary raw logits and per-layer hidden vectors versus AR; optional forced rejection/all-acceptance positions |
| `replay` | Captured finite Q/KV and semantic CSR inputs versus dense FP32 attention, repeat stability, and H48/H64 dispatch without model loading |
| `draft-replay` | Captured ROCm draft GQA cache/query inputs versus independent dense sliding-window attention, without model loading |

From the repository root in an installed development environment:

```bash
.venv/bin/python -m tools.raon_validation core --output evidence/core.json
.venv/bin/python -m tools.raon_validation rollback --output evidence/rollback.json
.venv/bin/python -m tools.raon_validation stress --rounds 1024 --output evidence/stress.json
.venv/bin/python -m tools.raon_validation mla --output evidence/mla.json
.venv/bin/python -m tools.raon_validation sampler --samples 65536 --output evidence/sampler.json
.venv/bin/python -m tools.raon_validation sampler-gold --output evidence/sampler-gold.json
```

Numerical gates are declared before execution: BF16 RDA/convolution use
RMS error <= `1e-5 + 0.01 * reference_RMS` and maximum absolute error <=
`1e-5 + 0.02 * reference_peak`. The FP8 MLA screen uses 0.02 and 0.05
respectively. These are diagnostic tolerances, not a general accuracy
guarantee. Nonfinite tensors always fail. Cache rows outside the write set
must remain exactly unchanged. The sampler uses a fixed seed and 65,536
trials with a per-bin bound of six binomial standard errors plus `2/N`.
The checker includes a deliberately corrupt tensor that must fail.

The RDA reference computes `sigmoid(beta + delta) * v` and
`sigmoid(beta) * predicted_value` separately. It does not call the production
value-ratio transform or native gate/recurrence helpers. Actual convolution
and value-fold rounding boundaries are included in the comparison.
Previous accepted counts include the mandatory target token: count 1 means
zero draft tokens accepted; count `draft+1` means all draft tokens accepted.

## Model diagnostics

Provide a JSON array of up to 160 objects with numeric-string `id` and either
`prompt_token_ids` or `prompt`. Keep model paths/revisions fixed. For example:

```bash
.venv/bin/python -m tools.raon_validation model \
  --model /models/target --draft-model /models/draft \
  --draft-tokens 7 --temperature 1 --prompts-json prompts.json \
  --generation-seconds 1500 --output evidence/model-k7.json
```

Use `--draft-tokens 0 --temperature 0` to capture an AR reference. Then run
`--temperature 0 --draft-tokens 1` (or 3/7) with
`--reference-json evidence/ar.json`. The tool checks prompt identity and
reports the first differing token in the generated common prefix. A prefix
comparison is not a full vocabulary-logit comparison or a formal proof.
Top-five token log probabilities are retained for investigating differences.

`--max-tokens` defaults to the remaining 262,144-token context. A smaller
explicit value is useful for bounded prefix comparisons and is recorded.
EOS is honored. Time-bounded unfinished generations are `INCOMPLETE`.
Warmup and padded token rows are excluded from live finite-value checks.
A nonfinite failure writes `OUTPUT.failure.pt` with current layer inputs,
selected cache rows before the operation, metadata and accepted outputs.
These dumps can contain prompts/model-derived tensors; retain them privately.
Instrumentation changes execution timing; these runs are not performance
measurements and must not be merged into MMLU-Pro scores.

## Teacher forcing and MLA failure replay

The teacher comparison is **SD versus AR**, not native kernels versus Torch
gold. An end-to-end raw-logit difference is not by itself a local kernel
accuracy failure. Use the operation-level gold checks below for that question.

Use one prompt, temperature zero, an ordinary non-EOS token, and the same
`--max-tokens` for both runs. Choose the token from the model's tokenizer.
For example, if token 1000 is non-EOS in the pinned tokenizer:

```bash
.venv/bin/python -m tools.raon_validation teacher \
  --model /models/target --prompts-json one-prompt.json \
  --draft-tokens 0 --force-token 1000 --max-tokens 512 \
  --output evidence/teacher-ar.json
.venv/bin/python -m tools.raon_validation teacher \
  --model /models/target --draft-model /models/draft \
  --prompts-json one-prompt.json --draft-tokens 7 --force-token 1000 \
  --accepted-drafts 3 --max-tokens 512 \
  --reference-json evidence/teacher-ar.json --output evidence/teacher-k7-a3.json
.venv/bin/python -m tools.raon_validation replay \
  --dump evidence/model-k1.json.failure.pt --output evidence/replay.json
.venv/bin/python -m tools.raon_validation draft-replay \
  --dump evidence/model-k7.json.failure.pt --dim 320 --heads 8 --window 2048 \
  --output evidence/draft-replay.json
```

The teacher hook records raw logits before replacing processed sampling
logits with a point mass. `--accepted-drafts N` also replaces draft proposals
after executing the real draft; the acceptance histogram must show `N+1`.
These are deliberate test-only controls, not serving options or quality
evaluation. The raw full-vocabulary and hidden-vector screens retain the
1% RMS/2% peak criteria above. Matching sampled text is insufficient.
`OUTPUT.logits.pt` and `OUTPUT.hidden.pt` retain the compared tensors.
Coverage is a sparse set of positions, not every generated position.

Replay supports the captured TP1 Raon H40, D576/V512, page-size-one native
decode inputs. It gathers the logical KV rows into fresh storage, zero-pads
physical sequence tails, and **rebuilds** all AITER work/reduction metadata.
Opaque captured metadata can contain device addresses and must never be
reused in another process. Results compare only the first 40 live heads.
The original allocation's metadata and cache layout are not reproduced.

`mla --scratch-poison --kv-padding zero` initializes native FP32 scratch
allocations with NaNs and checks live-head output stability. Repeat with
`--mla-heads 64`. Dense attention of the same dequantized inputs is the
acceptance reference. Alternative references that quantize softmax P are
reported only to diagnose FP8 rounding; they cannot turn a failure into PASS.
Scratch sensitivity is evidence for further investigation, not by itself
proof of a particular uninitialized read.

The draft replay supports the captured TP1 Raon GQA layout, head dimension
320 and window 2,048. It repacks only the captured physical pages and rebuilds
the logical block table. The first failing request is checked against dense
FP32 attention that reads only live windows. The original output's nonfinite
status is reported separately. No model weights are loaded.

The captured draft layer 25 failure had finite Q/K/V and finite KV throughout
the live window; expired cache pages contained NaNs. The ROCm prefix-prefill
context loop loaded those expired values before masking scores, allowing
`0 * NaN` to contaminate the value reduction. This fork skips expired tiles
and masks boundary K/V loads. The original captured-input replay failed;
the patched replay passed with 0.168% relative RMS error. Eight regression
cases cover query lengths 2/8, causal/non-causal attention, BF16/FP8 cache,
whole expired pages and the masked part of a boundary tile. A subsequent
post-fix diagnostic passed all 160 requests at each of SD 1/3/7, with natural
stop and no detected NaN or token-zero repetition. This establishes the
measured stability scope, not SD quality or distributional equivalence.

## Actual-input PyTorch FP32 gold

```bash
.venv/bin/python -m tools.raon_validation model \
  --model /models/target --draft-model /models/draft \
  --prompts-json one-prompt.json --draft-tokens 7 --max-tokens 128 \
  --mla-heads 64 --light --torch-gold --output evidence/gold-k7.json
```

Repeat with draft counts 0/1/3/7. Use one request per process. The opt-in
instrumentation observes actual production decode calls without replacing
their outputs. Both RDA and MLA checks must execute for coverage to pass.
Comparisons are recorded in `OUTPUT.gold.json` and the main result. The first
failing part saves private tensors in `OUTPUT.gold-rda.pt` or
`OUTPUT.gold-mla.pt`.

* RDA: check the post-convolution recurrent operation, before gated RMSNorm
  and output projection, from the exact live initial state. One FP32 Torch
  reference computes original erase/write equations on V before value folding;
  a second uses native stored BF16 folded V to isolate kernel arithmetic from
  transform rounding. Check both outputs and candidate states.
* MLA: check latent attention before output projection with explicit FP32
  Torch QK, causal masking, softmax and PV, using the same stored/dequantized
  Q and KV as the native call. Gold does not round probabilities to FP8.

The 2026-11-23 actual-input run checked all 18 RDA and seven MLA target layers,
sampling calls 1/2/4/8/16 per layer in bounded 128-token generation probes.
All 360 RDA cases and 140 MLA cases passed the unchanged numerical screens.
Maximum output relative RMS errors were 0.258% and 0.694%, respectively.
See [the complete operation-level record](results/torch-gold-20261123.json).
The current optimized production paths are retained on this evidence.

Each check starts from its operation's current native state. It does not
independently rebuild prefill, convolution, all projection/GEMM operations,
or the full request's history, and does not establish lossless SD or a quality
score. Earlier synthetic MLA edge-case failures and end-to-end AR/SD
differences remain preserved as separate evidence.

`--mla-backend TRITON_MLA` and `--common-rda` are diagnostic controls for the
earlier failed path-alignment experiment. The latter uses the generic
recurrent kernel for AR and fixes recurrent launch settings for both paths.
Neither is a recommended serving change. `--torch-gold` requires the production
RDA and AITER MLA paths. Those trials did not improve raw-logit agreement.

## SD verifier and resampler gold

```bash
.venv/bin/python -m tools.raon_validation sampler-gold \
  --output evidence/sampler-gold.json
```

`sampler-gold` directly checks accepted lengths and every emitted token,
including rejection recovery and the bonus token, against independent Torch
FP64 probability-space equations. It covers standard and block verification,
draft lengths 1/3/7, temperatures 0/0.7/1.3, both native Gumbel precision modes,
full and one-hot proposal distributions, equal distributions, every forced
rejection position, placeholders, ragged draft lengths including zero, and
shuffled persistent request-state indices with unused slots poisoned by NaNs.
An additional sparse 16-candidate proposal matches DFlash2's sparse cache shape.
Vocabulary sizes 33/8,209/259,840 exercise small, multi-block and real-model
vocabulary reductions.

The production RNG helpers supply the same uniforms/Gumbel draws to both
implementations; the oracle does not share the verifier's probability,
acceptance, residual or token-selection implementation. RNG correctness itself
is outside this test. No tolerance or mismatch exclusion is applied to accepted
counts and emitted token IDs. The 2026-11-23 run passed all 648 conditions and
60,480 requests with zero count or token mismatches. See
[the full verifier record](results/sampler-gold-20261123.json).
Draft-model generation, scheduler/cache history, watermarking and upstream
logit processing remain separate checks.

## Bounded SD configuration timings

`python -m tools.raon_validation.benchmark --help` exposes a separate offline
engine timing tool. It does not install numerical guards or replace outputs.
It warms each batch shape, records two measured repetitions by default,
retains accepted output IDs and emission events, and snapshots native SD
proposal/acceptance counters before and after each point. Its default points
are concurrency 1/16 with a 512-token cap; EOS is honored. Prompt identity,
observed output lengths and stop reasons must accompany comparisons.

These are offline queue-to-output timings, not HTTP latency or quality scores.
Natural EOS can produce different output lengths even with the same inputs;
report the lengths and repetition spread alongside throughput/TTFT/TPOT.
Benchmark results do not establish semantic quality or a universally optimal
SD configuration. Existing AR evaluation is kept separate.

`--requests 64 --concurrencies 16 32 64` uses the same 64 requests at every
point, admitting a new request when one finishes. The engine's maximum sequence
count is the largest requested concurrency. In the measured MI355X run, the
SD7/probabilistic/standard profile achieved median output rates of
641/915/1,487 tokens/s at those concurrencies (two repetitions, same GPU).
Pooled request p50 offline TTFT was 130/295/1,051 ms; p50 TPOT was
21.7/23.0/23.0 ms. These are throughput/latency tradeoffs on this closed-loop
workload, not an HTTP SLA. On the separate 16-request timing workload, SD1/3
and block verification did not improve throughput robustly over
SD7/probabilistic/standard. The requested greedy point was overridden to
probabilistic by the model config; it cannot establish greedy performance.
The tool now records and checks the effective speculation fields before
warmup/measurement. Natural EOS changed observed output lengths;
the raw lengths and repetition spread are retained in
[the measurement record](results/sd-performance-20261123.json).

## Compare ongoing SD off/on evaluations

```bash
.venv/bin/python -m tools.raon_validation.evaluation \
  --sd-off-root /evidence/original-ar \
  --sd-on-root /evidence/patched-sd \
  --window-seconds 300 --output evidence/evaluation-comparison.json
```

This CPU-only tool reads the existing `state/PROFILE/mmlu/completed` records
and `jobs/JOB/PROFILE.log` files; it never changes or pauses an evaluation.
It selects the latest common allocation unless `--job` is supplied, verifies
the protocol and paired prompt hashes, and reports both cumulative and common
completed-question accuracy. A second paired subset excludes resumed requests.
Generation rates are time-weighted over the same log window. Different active
questions, cache histories and the original AR versus patched SD runtime mean
this is a live descriptive comparison, not an isolated performance A/B.
Raw prompts, output text and token IDs are excluded from its JSON output.
See the [2026-11-23 live comparison](results/evaluation-comparison-20261123.json):
874 paired questions scored 83.75% AR and 84.32% SD7. The common five-minute
window measured 231 versus 436 output tokens/s (1.89x). These are partial,
descriptive results with the limitations above.
The [later checkpoint snapshot](results/evaluation-comparison-20261123T1127Z.json)
compares 1,308 common questions: AR 82.80%, SD7 83.18%, with 236/546 output
tokens/s in its five-minute window. Use an explicit `--job` for the last fully
reported allocation while a new allocation is still initializing.

## Experimental Quark MXFP4 conversion

### Profiling and the first GEMM tuning screen

After sourcing the corresponding runtime profile and configuring isolated
caches as below, collect warmed traces on one idle GPU:

```bash
.venv/bin/python -m tools.raon_validation.profile_model \
  --precision mxfp4 --audit-dir evidence --output evidence/profile-result.json \
  --model /models/raon-quark-mxfp4 --draft-model /models/raon-draft \
  --prompts-json prompts.json --concurrencies 16 --draft-tokens 7
.venv/bin/python -m tools.raon_validation.gemm_tune \
  --output evidence/gemm-gold.json
```

`profile_model` runs three full-cap warmup batches, then records separate
one-token and 64-token-cap traces. The former includes prefill and first-token
sampling; the latter includes prefill and short decoding. Both honor EOS.
Triton cache misses and AITER builds are forbidden during capture, and FlyDSL
uses run-only mode. These traces are instrumented and cannot establish serving
throughput. Per-step CPU ranges and input shapes help attribute GPU launches.
Add `--python-only` for separate cProfile captures (`.pstats` plus cumulative
and self-time tables), without the GPU trace recorder. Those durations are also
instrumented; normalize by actual engine steps before comparing paths.

`gemm_tune` screens the existing `RAON_KDA_PROJ_GEMM_TUNE=1` setting, which
changes the BF16 small-M launch tile and wave count while retaining the K-loop.
It compares native baseline/candidate outputs bitwise, checks per-row batch
invariance, records errors against Torch FP64 products of identical BF16 inputs,
and times warmed API calls with GPU events and a Triton cache-miss guard.
This gfx950 screen uses deterministic random inputs, transposed linear-weight
layouts, seven projection shapes, BF16/FP32 stores and the requested M values.
It is not an all-input GEMM proof, an attention/SD gold test, or model quality
acceptance. A passing screen permits separate end-to-end trials; no runtime
default is changed by these tools.

The [2026-11-23 initial optimization record](results/local-optimization-20261123.json)
retains the completed C1 profile summary and 84-case screens on each of two
GPUs (168 checks). Every candidate output matched the native baseline bitwise;
both baseline and candidate passed the tested batch-invariance checks. Full
model performance is measured separately with same-GPU ABBA trials. Both C16
GPU traces and Python profiles were subsequently captured in allocation 86666.
The earlier C16 attempt that exceeded its lease remains an incomplete attempt.
The first short ABBA comparison measured pooled medians of 34.25/37.84 output
tokens/s (flag 0/1) at C1 and 402.17/357.79 at C16. Each flag has two engine
starts and three official repetitions per start, with EOS honored. The C1 point
has only one request per repetition. These descriptive +10.47%/-11.04% changes
are not a universal tuning recommendation; longer queued-request trials run
separately. Native compilation metadata confirms eight-wave baseline launches
and additional four-wave launches with the candidate. Existing full-context
quality evaluations use their unchanged frozen sources and baseline settings.

### Excluding compilation from SD timing

Use the separate compiler-guarded timing tool after sourcing the appropriate
Raon runtime profile. Give each engine its own cache directories, set before
starting Python. Run the profiles sequentially on the same idle GPU:

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export TRITON_CACHE_DIR="$PWD/evidence/triton"
export AITER_JIT_DIR="$PWD/evidence/aiter"
export FLYDSL_RUNTIME_CACHE_DIR="$PWD/evidence/flydsl"
export TORCH_EXTENSIONS_DIR="$PWD/evidence/torch-extensions"
.venv/bin/python -m tools.raon_validation.compile_clean \
  --precision mxfp4 --audit-dir evidence --warmup-tokens 512 \
  --priming-repetitions 2 --repetitions 5 \
  --model /models/raon-quark-mxfp4 --draft-model /models/raon-draft \
  --prompts-json prompts.json --output evidence/timing.json \
  --concurrencies 16 --output-tokens 512 --draft-tokens 7
```

For MXFP8, source its runtime profile and change the precision/model paths.
The initial full-cap warmup and two priming batches are excluded. During
official repetitions, a Triton JIT cache miss or AITER build raises an error,
and FlyDSL runs with `FLYDSL_RUNTIME_RUN_ONLY=1`. Code-cache snapshots occur
outside the timer. The portable tool hashes artifacts and rejects additions,
modifications and deletions. This covers those APIs and cache roots, not every
possible compiler invocation. A blocked build invalidates the measurement;
do not promote a partial result. Use `official_compile_free` to select points;
the underlying benchmark also retains warmup/priming points in its JSON.

The [2026-11-23 GPU4 comparison](results/compile-clean-sd-20261123.json) used
a frozen experimental wrapper with these compiler guards and size/mtime cache
snapshots. Both profiles passed all three official repetitions without cache
changes or blocked attempts. Median output throughput was 466.07 tokens/s for
MXFP8 SD7 and 397.33 for MXFP4 SD7 (-14.75%). Both honored EOS, so output
lengths differed. These are short C16, 512-token-cap, offline engine timings
including prefill, with four unchanged evaluation engines on other GPUs.
They do not replace full-length evaluation or establish a universal slowdown.
The initial warmup measured 213.75/173.55 tokens/s; that difference combines
compilation, lazy initialization and other warmup effects. Its duration is not
a measurement of compilation alone. The portable tool is a later refactor;
its CPU cache-audit tests passed, but this recorded GPU run used the frozen
wrapper whose hash is included in the result.

The pinned `haanjack/quanto` revision is
`30e613245063db738570212307e799b3f0a63ffa`. Use its source with the image's
AMD Quark 0.12.post1, Torch 2.14.0+rocm10.2.0 and Python 3.14.10. Quanto's
older Quark submodule has a Python <3.13 build restriction and is not the
Quark implementation used in this experiment. Install quanto in a separate
environment without resolving or replacing Torch:

```bash
uv venv --system-site-packages .venv
uv pip install --python .venv/bin/python --no-deps --no-build-isolation /path/to/quanto
.venv/bin/python -m tools.raon_validation.quantize \
  --model /models/released-raon-mxfp8 --output /models/raon-quark-mxfp4
.venv/bin/python -m tools.raon_validation.quantized \
  --source /models/released-raon-mxfp8 --model /models/raon-quark-mxfp4 \
  --output evidence/mxfp4-export-audit.json
```

The Raon adapter registers a model template and temporarily connects Quark's
input-format recovery to row-wise E4M3/E8M0 decoding. It restores the released
MXFP8 weights before quanto's file-to-file RTN conversion. Only routed expert
weights are converted; attention, dense/shared FFN, routers, norms, embeddings
and output head remain unchanged. This is requantization of the released
MXFP8 checkpoint. It does not recover the unavailable original BF16 weights.
RTN uses no calibration/evaluation data. Quanto's MXFP4 scheme declares both
MXFP4 weights and dynamic MXFP4 activations (W4A4).

The actual three-tensor source restoration matches independent Torch FP64
arithmetic exactly. All 9,216 packed expert weights passed shape/scale checks;
all 1,002 other tensors have identical dtype, shape and bytes. Safetensors
storage changed from 32,702,145,088 to 19,867,486,640 bytes. These export
checks do not establish native GEMM correctness or model accuracy. Fresh
native-path smoke and MMLU-Pro run separately from the MXFP8 evaluation.

For this checkpoint only, source `docker/raon/runtime-mxfp4-env.sh` and select
the public `aiter` MoE backend. Quark's activation metadata selects its native
MXFP4/MXFP4 implementation. The profile disables the custom
MXFP8 loader/forward overrides and selects AITER's a4w4 SiTUv2 path. The vLLM
model config preserves Quark metadata and backend selection. No AITER/FlyDSL
source or compiler patch is required. The four-task example accepts
`--moe-backend aiter` and a local `--model` directory.
Future images built from these Dockerfiles select this experimental profile
with `RAON_RUNTIME_PROFILE=mxfp4`; the default remains the MXFP8 profile.

## Experimental MLA workaround

On the tested gfx950 AITER build, real H40 padded to H48 at query length two
produced allocation-sensitive native outputs, including a live model NaN.
The captured finite Q/KV replay passed the screen at H64 and failed at H48.
This fork provides an opt-in workaround:

```bash
export VLLM_ROCM_AITER_MLA_PAD_H40_TO_64=1
```

Set it before starting the engine. It changes decode metadata and query
padding together; FP8 prefill keeps its separate H48 path. It uses existing
AITER kernels and requires no hipify or compiler change. It is disabled by
default in generic vLLM; this fork's dedicated Raon runtime profile enables it.
`model/teacher --mla-heads 64` enables this actual vLLM option and
refuses to run if the patch is missing. Other head counts are unchanged.
The workaround is not an SD correctness or model-quality acceptance: the
teacher-forced raw-logit comparison still fails. See
[the MI355X diagnostic record](results/mi355x-20261123.json) for measured scope.

Both Raon Dockerfiles include these tools under `/opt/raon/tools/raon_validation`.
After sourcing `/opt/raon/runtime-env.sh`, change to `/opt/raon` and use its
virtual-environment Python with the commands above. Previously published
image digests predate these tools and the opt-in workaround.

`python -m tools.raon_validation.prompts --help` prepares the same five-shot
MMLU-Pro formatting and round-robin ordering used by the evaluation. It
requires locally pinned Parquet data, the evaluator CoT template and tokenizer.
Use `--reference-manifest` to require identical prompt-token hashes.
PyArrow is needed only for this input-preparation helper.

Each JSON result records the source hash, hardware/runtime, fixed arguments,
individual comparisons and exit status. `PASS` covers only executed cases;
it does not establish all-input correctness, model quality, full scheduler
lifetime correctness, or cross-device bitwise equivalence. Outputs with
changed RNG consumption must be assessed distributionally, not by seed alone.
