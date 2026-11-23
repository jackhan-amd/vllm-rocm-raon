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
whole expired pages and the masked part of a boundary tile. A long full-model
post-fix run is still required; this result does not establish SD quality.

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
default. `model/teacher --mla-heads 64` enables this actual vLLM option and
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
