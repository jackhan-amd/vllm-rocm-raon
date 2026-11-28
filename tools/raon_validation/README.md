# Raon ROCm correctness diagnostics

The [mixed RDA and shape dispatcher](../../docker/raon/NATIVE-RDA-MIXED-AND-DISPATCH-20261127.md)
adds `hip_rda_fused_sd`, `hip_rda_vsplit_sd` and `hip_rda_auto_sd`. Use
`--decode-ar-history 10`, `--decode-mixed --decode-mixed-prefill 129`,
`--decode-spec-ragged` or `--decode-spec-null` for separate boundary fixtures.
Mixed integration references the existing implementation; pure component gold
uses independent Torch equations. These diagnostics do not measure model TPS.

The [native HIP SD experiment](../../docker/raon/NATIVE-HIP-RDA-SD-20261127.md)
adds pure SD gold and native-kernel timing with `--decode-backend hip_rda`.
Expanded-pool AR and mixed-batch native coverage remain pending; this selector
is experimental and defaults remain unchanged.

The [native mHC finish/head experiment](../../docker/raon/MHC-FINISH-HEAD-20261127.md)
adds `mhc_math --fuse-finish-head` for real-weight Torch gold, refit/replay,
direct comparison with the unfused native chain and warmed block graph timing.
The runtime switch is opt-in and remains off by default.

For CPU-only warm serving interval analysis, use
`python -m tools.raon_validation.gap_budget --help`. See the
[warm decode budget](../../docker/raon/GAP-BUDGET-20261127.md) for coverage gates,
warmup evidence, profiler overhead and interpretation limits.
The [RDA/mHC hardware-counter report](../../docker/raon/KERNEL-ROOFLINE-20261127.md)
records native roofline results, diagnostic commands and reviewed fusion/tile
candidates. Counter-profile durations are separate from graph replay timings.
The [RDA native-port audit](../../docker/raon/RDA-NATIVE-PORT-AND-BUDGET-20261127.md)
tracks full-model costs, mixed-batch native coverage and pending HIP port/gold
acceptance work separately from the gate-only counter experiment.
The [native HIP AR prototype](../../docker/raon/NATIVE-HIP-RDA-AR-20261127.md)
adds `--decode-backend hip_rda_ar --decode-ar-only`, NULL/padding checks with
`--decode-ar-padding`, optional profiler witnesses with `--decode-kernel-witness`,
and warmed isolated graph timing with `--decode-timing-replays 100`. Its SD and
mixed-batch native support remain pending; defaults are unchanged.

Run inside the pinned Raon ROCm environment with one visible GPU. Source
`docker/raon/runtime-env.sh` before starting Python. Preserve the image's
`PYTHONPATH`: the ROCm SDK places AMD-SMI there. The tools now refuse an
unrecognized ROCm platform. Use a virtual environment with the installed
Torch/vLLM/native extensions available; no model download is performed.

| Command | Independent check |
| --- | --- |
| `draft-native` | Experimental BF16 20Q/5KV D128 DFlash2: native HIP cache bytes versus Torch quantization, untouched slots, FlyDSL paged gather plus HIP attention versus FP32 noncausal sliding-window gold, expired NaNs, strided QKV/cache, padding, and replay with changed lengths, query boundaries, scales and block mappings. Select `--draft-case 0` through `7`; `auto` uses CK for eight-query requests and Opus for mixed lengths. No model-quality or end-to-end speed claim. |
| `flydsl-moe` | Candidate MXFP8 FlyDSL expert rows vs independent Torch gold with both FP32 and BF16 activation boundaries; repeated graph replay with changed routes, probabilities and inputs. Use `--moe-full-shape` for E128/H2560/I1088. This is a finite diagnostic, not a throughput measurement. |
| `moe-route-prepare` | Native route packing versus CPU expert counts and token/slot mapping; exact Torch BF16 probability bits, including ties and subnormals. Checks int32/int64 cache separation, M16/M32 padding, and captured replay with changing counts, duplicate experts and invalid IDs. |
| `aiter-rda` | Select `--decode-backend flydsl` for the three-kernel FlyDSL chain or `aiter_triton` for the older fused derivative. Compare with the existing AMD chain and independent Torch conv/RDA/norm gold; AR and SD lengths 2/4/8, all accepted counts, changing state indices, full conv histories, untouched recurrent rows, and exact graph/direct comparison. Finite MI355X results are linked below. |
| `effort` | Independent mixed-token alpha expansion, real decoder MoE-output boundary, Torch gold, zero-alpha bit preservation and HIP graph replay for release steering. |
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

Native mHC and ordered MoE combine candidates (opt-in):

```bash
.venv/bin/python -m tools.raon_validation.mhc_probe \
  --model /models/Raon-1-mini --candidate hip --packed --splits 16 \
  --output evidence/mhc-probe.json
.venv/bin/python -m tools.raon_validation.mhc_math \
  --model /models/Raon-1-mini --backend hip --output evidence/mhc-math.json
.venv/bin/python -m tools.raon_validation.combine_math \
  --topk 6 --output evidence/combine-math.json
RAON_MOE_COMBINE_UNW_FUSED=flydsl \
  .venv/bin/python -m tools.raon_validation.moe_graph_mxfp8 \
  --output evidence/combine-model-graph.json
```

`RAON_ROCM_MHC_BACKEND=hip` replaces only the mHC projection and mean-square
statistics with AITER's native HIP implementation. Raon's gate remains unchanged;
AITER's different Sinkhorn mHC block is never substituted. `flydsl` selects the
separate skinny-GEMM candidate. Both retain BF16 parameters and score boundaries.
The mHC math tool uses real checkpoint weights, independent Torch gate equations,
changed-input graph replay and in-place refit checks. It reports full-block batch
identity separately: numerical PASS does not mean every batch is bit-identical.

`RAON_MOE_COMBINE_UNW_FUSED=flydsl` selects a native FlyDSL stable expert-ID sort
and sequential FP32 sum of already weighted BF16 slots, with one final BF16 cast.
It applies to the MXFP8 trainer-realized combine caller; the native MXFP4 fused
MoE reduction remains owned by AITER. Value `1` still selects the older Triton
candidate. The combine test checks integer-ID types, duplicate/negative IDs,
cancellation, nonfinite classes, changed-route/input graph replay and batch
identity. Its graph timings are isolated operation costs, not model throughput.
Neither option is enabled by the default runtime profile, and these finite checks
do not establish model accuracy or a speedup for every workload.

`RAON_ROCM_MOE_DIRECT_GRAPH=1` is a separate MXFP8 candidate: it calls the same
native routed/shared chain inside the enclosing graph, bypassing the per-layer
eager-break callback and local graph/output copies. It requires the ordered
FlyDSL combine, breakable graphs, TP1/DP1 without EP and the trainer-realized
MXFP8 path. Unsupported diagnostic/phase modes raise. Warm packing and JIT first;
the native adapters refuse cold packing inside capture. Validate changed-input
replay and one shared execution per replay with:

```bash
.venv/bin/python -m tools.raon_validation.moe_graph_mxfp8 \
  --direct --output evidence/moe-direct-graph.json
```

This does not enable Torch.compile or alter the MXFP4 native runner.

`RAON_ROCM_MLA_SPLIT_GRAPH=1` is another independent candidate, requiring live
MLA metadata. It captures the existing Q/KV normalization/projections and output
gate/projection while retaining the live callback for Kimi's existing native
attention decision and cache owner. No attention kernel is replaced. Its
partition-only fixture checks exact output versus the upstream Kimi wrapper,
fresh phase/slot metadata and one cache update per replay:

```bash
.venv/bin/python -m tools.raon_validation.mla_split_graph \
  --output evidence/mla-split-graph.json
```

The fixture substitutes attention math to isolate the execution boundary; it is
not an additional native MLA kernel gold or model-accuracy test.

`RAON_ROCM_MHC_FINISHER=flydsl`, with the HIP mHC backend, replaces the two
Torch split reductions, score cast and mean normalization with one native
FlyDSL launch. It retains sequential FP32 accumulation and the BF16 score
boundary; its reduction order can differ from Torch's. Check the complete
real-weight block and changed-input/refit replay with `mhc_math --backend hip
--finisher flydsl --model /models/Raon-1-mini --output evidence/mhc-finish.json`.
The default finisher remains Torch. This switch does not change attention,
activation quantization or MoE routing.

Release checkpoint steering tools:

```bash
.venv/bin/python tools/raon_validation/test_effort_steering.py
.venv/bin/python -m tools.raon_validation effort --output evidence/effort.json
.venv/bin/python -m tools.raon_validation.effort_cost --output evidence/effort-cost.json
.venv/bin/python -m tools.raon_validation.release_model \
  --model /models/Raon-1-mini --output evidence/release-smoke.json
```

The release smoke requires the candidate profile in
`docker/raon/runtime-flydsl-env.sh` (choose `RAON_RUNTIME_PROFILE=mxfp4-flydsl`
for the release MXFP4 derivative). It uses breakable PIECEWISE graphs, dtype
auto, FP8 KV and enabled prefix caching; it refuses a resolved non-PIECEWISE
mode and records generation replay counts. Add `--draft-model` with the pinned
release DFlash2 snapshot for SD7. It checks seven mixed-effort requests with
fixed prompt IDs and a 32-token output cap, not full API template behavior,
long generations or model accuracy. `effort_cost --implementation compiled`
measures a separate pointwise fusion candidate, not the deployed plain path.
`aiter-rda --conv-weight-fp32` also checks the loader's losslessly widened
convolution parameters, without changing the checkpoint's BF16 values.

```bash
.venv/bin/python -m tools.raon_validation core --output evidence/core.json
.venv/bin/python -m tools.raon_validation rollback --output evidence/rollback.json
.venv/bin/python -m tools.raon_validation stress --rounds 1024 --output evidence/stress.json
.venv/bin/python -m tools.raon_validation mla --output evidence/mla.json
.venv/bin/python -m tools.raon_validation sampler --samples 65536 --output evidence/sampler.json
.venv/bin/python -m tools.raon_validation sampler-gold --output evidence/sampler-gold.json
```

The opt-in MoE/KDA candidate and its precision-specific activation profile are
documented in [FLYDSL-CANDIDATE.md](../../docker/raon/FLYDSL-CANDIDATE.md).
On an idle gfx950 GPU, its additional checks are:

```bash
export RAON_MXFP8_FLYDSL_FUSED_A2=1
.venv/bin/python -m tools.raon_validation flydsl-moe --moe-full-shape \
  --output evidence/flydsl-mxfp8-gold.json
.venv/bin/python -m tools.raon_validation aiter-rda --decode-backend flydsl \
  --draft-counts 1 3 7 --output evidence/flydsl-rda-gold.json
.venv/bin/python -m tools.raon_validation.moe_graph_mxfp8 \
  --output evidence/mxfp8-model-graph.json
.venv/bin/python -m tools.raon_validation.moe_graph \
  --output evidence/mxfp4-native-graph.json
```

These commands do not start a serving endpoint or measure throughput. The
decode tool selects its fused/direct fixtures explicitly; no global fused
decode option is needed. The three-kernel decode candidate is FlyDSL throughout;
the older single-kernel derivative is AMD Triton. CPU dispatch or offline
gfx950 compile-only checks do not establish GPU arithmetic correctness.

For an opt-in small-M MXFP8 experiment, set
`RAON_MXFP8_FLYDSL_TILE_CONFIG='{"block_m":16,"stage1_tile_n":128}'`.
The adapter uses the same M quantum for sorting, activation-scale layouts and
both native GEMMs; MXFP4 packing is unchanged. M32 remains the default.
Use `--moe-varied-scales --moe-token-counts 1 8 17 32
--moe-kernel-witness` with `flydsl-moe` to check different E8M0 exponents,
partial M tiles, independent Torch gold after changed routes, and the actual
stage1/stage2 GPU kernel names. Missing native M-tile witnesses fail the run.
`--moe-timing-replays 200` optionally measures one warmed graph interval after
correctness passes. This includes sorting, quantization and both GEMMs, but
excludes router, shared expert and combine; it is not model or serving TPS.

The separate `fused_route_prepare` candidate keeps the existing expert GEMMs
and checkpoint bytes. For up to 32 rows, a native FlyDSL kernel packs routes
and rounds probabilities through BF16 in one dispatch. With the native FlyDSL
combine selected, the model also retains router int32 IDs. Larger batches use
the existing native sort and cast sequence. This candidate is off by default
and applies only to the MXFP8 adapter.
The configuration is cached per process; restart the server to change it.
The ordered combine launcher separates compiled signatures by ID dtype and
device. `combine_math` starts with an int64/int32 cancellation regression before
the existing random, nonfinite and changed-input replay cases.

```bash
.venv/bin/python -m tools.raon_validation moe-route-prepare \
  --output evidence/route-prepare-gold.json
RAON_MXFP8_FLYDSL_FUSED_A2=1 \
RAON_MXFP8_FLYDSL_TILE_CONFIG='{"fused_route_prepare":1}' \
.venv/bin/python -m tools.raon_validation flydsl-moe --moe-full-shape \
  --moe-varied-scales --moe-token-counts 1 8 17 32 --moe-kernel-witness \
  --output evidence/route-prepare-expert-gold.json
```

The [2026-11-27 small-batch record](results/moe-smallm-20261127.json) reports
190.9 to 202.1 output tok/s at C1, 859.3 to 895.2 at C8, and 1600.7 to 1649.0
at C32 with route preparation enabled and the default M32 GEMM geometry.
Each condition used one 128-prompt GSM8K diagnostic measurement after separate
warmup, with natural EOS and a 4096-token cap. No compilation was detected
during measurement. These are not official accuracy results or the fixed
ISL/OSL, measured-50%-prefix-hit benchmark. C1 and C8 outputs matched all 128
baseline texts; C32 used dynamic batching and matched 27/128 exactly, so the
record does not claim fixed-batch bitwise generation equivalence.
Matched M32 controls did not support promoting M16. A route-wise GEMV prototype
was slower both in a hot layer and with 24 distinct-address weight sets; neither
experiment changes the default schedule.

The [2026-11-25 MI355X record](results/mi355x-flydsl-20261125.json) contains
executed finite GPU results. FlyDSL RDA passed 45 AR/SD cases across batches
1/3/16, including 128 live SD graph tokens, after correcting physical indexing
of the rank-two state-index matrix. Use `--decode-batch-size` to select a batch.
The fused-A2 MXFP8 path passed Torch gold and legacy comparison; it avoids the
extra BF16 intermediate with `RAON_MXFP8_FLYDSL_FUSED_A2=1`. Its component graph
and actual model shared-stream call sites passed exact replay checks.

`moe_graph` checks the native MXFP4 layout/selector, actual SharedExperts owner,
and outer graph on a synthetic fixture; it does not load a Quark checkpoint or
the full FusedMoERunner. It derives the reference's intermediate dtype from
actual stage1 metadata and restores inputs/routes between stream modes.
MXFP4 numerical gold/replay checks passed, but **exact checks remain FAIL**:
the selected stage2 atomic path is not bitwise-repeatable even in direct eager
repeats. The tool reports `numeric_status` separately and does not relax its
original exact gates. No throughput or full-model accuracy result follows.

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

### FP4 MoE intermediate-precision diagnostic

On one idle gfx950 GPU, with the MXFP4 runtime profile sourced, run:

```bash
.venv/bin/python -m tools.raon_validation.moe_gold \
  --model /models/raon-quark-mxfp4 --output evidence/moe-gold.json \
  --tokens 128 16384
```

This diagnostic uses the installed AITER tuner's synthetic layer inputs and
records its source hash. It compares the existing W4A4 operator and the tuner's
experimental fused port against three Torch references: BF16 intermediate,
BF16 then FP4, and FP32 then FP4. The FP4 intermediate reference uses Torch
E8M0 group-32 scaling and E2M1 packing. Input and weight quantization share the
vendor fixture; this is a conditional arithmetic comparison, not an independent
validation of every quantizer or the complete model. The fixed shapes represent
Raon's 128 experts, top-6 routing, hidden size 2560, intermediate size 1152 and
SiTUv2 beta=4 / linear_beta=25.

The [2026-11-23 diagnostic](results/moe-gold-20261123.json) measured existing
operator NRMSE of 0.395%/0.398% against BF16-then-FP4 Torch, and fused-port NRMSE
of 0.267%/0.267% against FP32-then-FP4 Torch at M=128/16384. Comparing either
W4A4 path against a BF16-only intermediate instead gives about 14.8% NRMSE.
The existing and fused paths differ by 3.57%/3.71% NRMSE, largely explained by
intermediate rounding. These two seeded synthetic cases do not establish model
quality or authorize deploying the port. `MEASURED` means finite measurements
were collected; the tool imposes no numerical acceptance threshold and installs
no tuning CSV. Its `cosine_diff` field follows the vendor-style formula
`1 - 2*dot(x,y)/(sum(x*x)+sum(y*y))`; it is not ordinary cosine distance.
The pinned vendor fixture resets its seed to zero during construction. The
historical record's `seed` is the caller's requested seed, not the effective
fixture seed. The current tool records the observed seed and an input
fingerprint per case; independent-seed claims require distinct fixtures.

### Excluding compilation from SD timing

`--draft-tokens 0` also benchmarks SD off. It passes no speculative
configuration and checks that the engine actually disabled speculation.
Positive draft lengths retain the SD configuration and verification checks.

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

## Release mixed-precision export and independent gold

`quantize --reference-nvfp4-config /path/to/pinned-official-config.json` translates
the official NVFP4 layer policy to MXFP4 W4A4 experts, MXFP4 W4A16 other linears
and an MXFP8 LM head. NVFP4 block-16 scales are not reused as MXFP4 scales.
This remains our quanto/Quark RTN conversion; the reference metadata does not
establish the official conversion algorithm. The original expert-only mode
is retained when that option is omitted.

```bash
.venv/bin/python -m tools.raon_validation.quantized \
  --source /models/raon-original --model /models/raon-mixed \
  --reference-nvfp4-config /path/to/pinned-official-config.json \
  --output evidence/export-layout.json
.venv/bin/python -m tools.raon_validation.quantized_math \
  --source /models/raon-original --model /models/raon-mixed \
  --reference-nvfp4-config /path/to/pinned-official-config.json \
  --output evidence/export-arithmetic.json
RAON_ROCM_DENSE_LINEAR_BACKEND=flydsl .venv/bin/python \
  -m tools.raon_validation.dense_math --output evidence/dense-gold.json
.venv/bin/python -m tools.raon_validation.release_model \
  --model /models/raon-mixed --audit-loaded-linears --torch-gold-mla \
  --max-tokens 128 --output evidence/release-gold.json
```

The arithmetic audit uses only independent Torch E2M1/E4M3/E8M0 math and reads
every converted weight. `dense_math` compares native GEMMs with FP32 Torch,
checks strided inputs and changed-input HIP graph replay, and verifies the
SiTU composition. Its intentionally strided SiTU negative control must fail.
`--audit-loaded-linears` checks real loaded/merged payloads and scales against
checkpoint tensors, then compares each loaded native Linear with Torch GEMM.
`--torch-gold-mla` checks the first seven real AITER decode calls, per sequence,
against dense causal FP32 Torch attention on the same quantized Q/KV inputs.
These tools distinguish conversion, loading, kernel and composition failures.
None is a full model accuracy evaluation or serving-speed measurement.

Source `runtime-flydsl-env.sh` first. The native candidate also requires
`RAON_ROCM_MLA_LIVE_METADATA=1` and
`RAON_ROCM_DENSE_LINEAR_BACKEND=flydsl`. The MLA callback refreshes the current
phase and metadata while retaining the Kimi/AITER kernels; other graph segments
remain active. Dense unpadding materializes contiguous rows because downstream
SiTU addresses rows using the logical width. The fixed-prompt release fixture
uses the same thinking-on token IDs for all efforts to isolate steering;
separate API tests cover `none` turning thinking off.

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

## Raon API parser checks

```bash
python -m tools.raon_validation.parser_api \
  --model /path/to/pinned/Raon-1-mini --output parser-result.json
```

This CPU diagnostic checks both `raon_1` registries, reasoning extraction and
streaming, thinking disabled, plain content, typed tool arguments, and the
current five-argument structural-tag dispatch. It uses the local checkpoint's
tokenizer without downloading remote code. Its PASS is independent of GPU
kernel correctness, generation quality and throughput.

## MLA stack regression artifacts

Keep the JSON and `.L*.Q*.pt` files produced by the `mla` diagnostic for both
stacks. Compare the saved tensor payloads without importing Torch or executing
pickle:

```bash
python -m tools.raon_validation.compare_mla_artifacts \
  --baseline /results/mla-baseline.json \
  --candidate /results/mla-candidate.json \
  --output /results/mla-stack-comparison.json
```

This same-Torch-version check compares inputs, outputs and the FP32 references
byte for byte. It preserves each original gold verdict; identical stacks can
both exceed the mathematical FP32 error threshold. A different payload requires
further inspection and is not automatically a numerical regression.

## Projection and expanded graph checks

The opt-in Raon mHC gate replaces the existing head, Torch chart and mix with
native FlyDSL head/chart/aggregate and mix kernels. It requires
`RAON_ROCM_MHC_BACKEND=hip`, `RAON_ROCM_MHC_FINISHER=flydsl`, and
`RAON_ROCM_MHC_GATE=flydsl`. Its default is `legacy`; this is independent of
attention and MoE precision. The BF16 projection scores and branch boundary,
FP32 chart, zero clamp floor, and original checkpoint parameters are retained.
The chart is Raon's chart equation, not AITER's Sinkhorn mHC algorithm.

```bash
python -m tools.raon_validation.mhc_math \
  --model /path/to/pinned/Raon-1-mini --backend hip --finisher flydsl \
  --gate flydsl --branch affine --layers 0 1 12 24 \
  --rows 1 2 4 5 8 32 64 256 1024 --scales 0.125 1 8 \
  --output mhc-gate.json --profile mhc-gate-kernels.json
```

This compares the branch input, post/residual coefficients, output, changed-input
graph replay and weight-refit replay against independent FP32 Torch math with
the BF16 boundaries preserved. It also compares with the existing native
projection plus legacy gate and the fully legacy path. The `affine` branch
exercises a non-identity BF16 branch. The optional trace is recorded only after
all numerical checks pass, to verify actual native kernel selection. It is
not a serving-throughput or model-accuracy measurement.

`--batch-invariant 0` additionally exercises the existing small-batch aggregation
tree, single-row mix order and shape-dependent chart FMA schedule. The default
is `1`, which fixes each arithmetic schedule across batch sizes.
`--timing-replays 200` optionally records one GPU-event timing interval per
shape/backend, with separate graph warmup; this times only the last selected
layer's MLP mHC block and excludes its real MLP computation. It does not measure
model output TPS. Restore original weights after the refit check before timing.

After sourcing the updated FlyDSL runtime profile:

```bash
python -m tools.raon_validation.projections --output projections.json
python -m tools.raon_validation.moe_graph_mxfp8 --tokens 256 --output moe8.json
python -m tools.raon_validation.moe_graph --tokens 256 --output moe4.json
python -m tools.raon_validation aiter-rda --decode-backend flydsl \
  --decode-batch-size 32 --draft-counts 7 --output rda256.json
```

Projection checks retain BF16 weights and compare against FP32 Torch arithmetic,
including changed-input graph replay, strided activations and bias preservation.
Each case records whether the native skinny kernel was selected; larger shapes
retain existing AITER dispatch. Direct-call and graph timings are distinct and
exclude initial kernel compilation. MXFP4 retains separate numeric and bitwise
statuses because native atomic repeats can differ without graph execution.

### Native fusion stack and AL diagnosis (2026-11-28)

The opt-in attention/mHC, MoE/mHC, RDA projection/ratio and SD logical-boundary
fixtures are registered under [`native_stack_20261128`](native_stack_20261128/README.md).
They include independent Torch checks, changed graph replays, actual-kernel
trace analysis, and processed-probability AL controls. See the
[source, image and validation contract](../../docker/raon/NATIVE-STACK-20261128.md)
for the exact tested scope and per-cell performance comparison.
