# ROCm graph/stream candidate: KDA, MXFP8 and MXFP4 MoE

The 2026-11-26 image adds a ROCm-built `torch-c-dlpack-ext` 0.1.5 and
pyarrow 26.0.1 for evaluation. The runtime Python overlay remains commit
`3b4e0972f7cbd83cbef90d24280bc28d9b3373ae`. The published image is
`docker.io/hanjack/vllm-rocm:raon-release-rocm102-3b4e097-dlpack-20261126`,
digest `sha256:457b95f9b571662ab7aa23ed9d019507fed567ba0e9fb9fd9e11ffb7f9f60396`.
Its GPU DLPack C exchange and zero-copy round trips passed for FP32, BF16,
E4M3 FP8 and UINT8 on MI355X in build job 88270. This is not a model-quality
or serving-throughput result. The immutable receipt is
[`image-dlpack-20261126.json`](image-dlpack-20261126.json).

Both Dockerfile recipes now build the addon from the same pinned upstream
source and archive checksum. The published image was assembled and verified
with the recorded Podman build job; the Dockerfile recipes were statically
checked rather than separately rebuilt. Builds without a GPU defer the
round-trip check until allocation.

Status: finite MI355X GPU kernel/component diagnostics completed on 2026-11-25.
Numerical gold checks passed. RDA and MXFP8 replay checks were exact; MXFP4's
native atomic path failed the stricter bitwise gate, including eager repeats.
Release MXFP8 and mixed MXFP4 passed finite SD-off/on native smokes: seven
fixed prompts per profile completed naturally and 49 real native MLA comparisons
per profile passed independent FP32 Torch gold. This is bounded correctness,
not full output-quality acceptance. Full quality and serving speed remain
unvalidated. The
original runtime profiles are unchanged; candidate profiles remain opt-in.

Checkpoint baseline updated to `KRAFTON/Raon-1-mini` on 2026-11-25, pinned at
`3f85cd63ea420100a84378f72ae162af86a441fd`. The official paired draft is
`KRAFTON/Raon-1-mini-dflash2`, pinned at
`74f2835bd683458f6124f48784b078b06770d824`.
See `checkpoint-baseline.json` for the config/index audit. The major tensor
shapes and MXFP8 format match preview, but this is not a weight-identity claim.
The release adds five FP32 reasoning-effort vectors and per-request steering
with default effort `high`. This fork now ports the upstream loader,
request/runner alpha propagation, steering addition and effort-specific prefix
cache salt. Eight CPU tests and 52 MI355X component checks passed. Four bounded
release smokes (MXFP8/MXFP4, SD off/on) processed seven mixed-effort requests each,
with actual generation graph replays. Their original structural PASS is not
an output-quality PASS: subsequent content review found failures described
above. The diagnostic now fails long single-token runs in its fixed math
prompt. Do not silently drop the vectors or
substitute effort `max` (alpha zero). These smokes use at most 32 output tokens;
they are not accuracy evaluation or long-generation certification.

The earlier GPU kernel diagnostics below used synthetic fixtures or preview-era
artifacts. The new release results are separate. A fresh Quark export is public
at `jackhan-amd/Raon-1-mini-MXFP4`, revision
`74fe32713a200a95e00faa30a161e5bbb4b3abc6`: 9,216 packed expert tensors and 1,007
unchanged tensors, including all five FP32 steering vectors. Seven weight shards
total 19,867,538,384 bytes; Hub sizes and LFS hashes were verified. Both exports
below use the final release, never the earlier preview derivative.

The additional `nvfp4-policy` Hub branch at
`62479b0a6c1c9ef2fec51a29bed5c983bc86b7f7` follows the official release NVFP4
layer policy: 9,216 routed matrices are MXFP4 W4A4, 272 other Linear matrices
are MXFP4 W4A16, and the LM head is MXFP8 W8A8. The reference's 322 non-routed
config entries include fused aliases. All 9,489 converted tensors passed
independent Torch scale/code arithmetic and all 734 excluded tensors were
byte-identical. Weight shards total 16,683,563,352 bytes. This uses our Quark RTN;
it does not assert the unpublished official NVFP4 conversion algorithm.
The printer's `mxfp4` selects this mixed variant; `mxfp4-routed` preserves the
earlier expert-only export.

Two runtime composition bugs were isolated using Torch gold. Kimi's MLA
Python phase/metadata dispatch must run in a live breakable callback
(`RAON_ROCM_MLA_LIVE_METADATA=1`), while keeping its native AITER kernels.
The new native FlyDSL A16W4 adapter must return contiguous unpadded rows:
SiTU's kernel addresses each row using the logical width. A deliberately
strided SiTU negative control produced relative RMS errors above 2.6, while
the corrected composition matched Torch exactly in those cases.
`RAON_ROCM_DENSE_LINEAR_BACKEND=flydsl` selects native A16W4 and MXFP8 linears;
Quark's LM head is enabled only by an explicit per-layer quantization entry.
Thirty matrix cases, 90 changing-input graph comparisons, 151 real loaded
Linear payload/scale checks and their GEMMs passed. See the release validation
tools for the boundaries, tolerances and negative controls.

The committed release result and tested overlay hashes live under
`tools/raon_validation/results/release-mixed-mxfp4-*-20261125.json`.
All 89 installed runtime sources matched the local candidate on both GPUs
at GPU validation time. Publication then applied formatting, typing, defensive
assertions, canonical imports and portable accelerator bookkeeping APIs, and
restored previously missing helpers in the inactive NVIDIA-only branch.
The tested raw hashes remain unchanged in the evidence. The accompanying
post-validation source review describes this distinction; it is not a second
GPU run of the publication cleanup.
The 45 changed Python files introduced no new Ruff findings; ten pre-existing
legacy-adapter findings remain in `kda.py`/`rda_base.py` (the parent has eleven).
Local graph capture/replay witnesses include warmup: bounded caches can still
report INERT on uncached request shapes, so graph switches set to one do not
establish universal replay coverage.

Full-model validation also found and corrected two pre-existing integration
gaps: the checkpoint's BF16 convolution weights are widened by the loader, so
FlyDSL now caches an exactly representable BF16 view and rejects lossy casts;
and CacheConfig now tracks explicit Mamba block sizing, as required by the
DFlash/APC alignment adapter. Fifteen AR/SD gold cases covered the FP32 loader
parameter. Core vLLM Python files changed; AITER/FlyDSL package sources did not.

The source comparison is pinned to KRAFTON `raon` at
`b6eecc0ca4c99c0474a0b2171c9432b0eabf5107`, the original import at
`335a45e3f9d8a7f5f76c884734b5797ec9623fb3`, and this fork's committed baseline
`4a874884ab2355775658e4f70cd2ddf5538bc4e0`. The candidate is a delta over that
baseline, with the native wheel ABI still pinned by `overlay-files.json`.
It does not imply that a new container has been built or deployed.

## Actual path

```text
MXFP8: compact checkpoint bytes
  -> one-time per-half padding and AITER weight/scale preshuffle
  -> FlyDSL route sorting
  -> AITER HIP MXFP8 activation quantization
  -> FlyDSL FP8xFP8 GEMM13 + SiTUv2 + original BF16 probability fold
  -> fused FP32-to-MXFP8 activation quantization inside GEMM13
  -> FlyDSL FP8xFP8 GEMM2 returning per-slot BF16 rows
  -> existing Raon combine + existing BF16 shared expert

MXFP4: existing Quark packing and AITER native A4W4 MoE/shared scheduling
  -> assert AITER's selected stage1 and stage2 identify FlyDSL, even graph=0
  -> bounded HIP graph wrapper around the existing native forward

KDA prefill: existing Raon convolution + post-convolution RDA value ratio
  -> same gfx950 support check as Kimi-K3
  -> fused HIP KDA prologue + chunk recurrence/output kernels
  -> explicitly requested fused path fails if native support is unavailable

KDA AR / SD decode: AITER FlyDSL conv -> local FlyDSL vector-decay RDA
  -> FlyDSL sigmoid-gated RMSNorm (three-kernel candidate)
  -> rocm_kda_graph, with refreshed state indices and accepted-token metadata
  -> HIP graph replay; legacy and AITER Triton remain explicit alternatives
```

No HIPIFY or Triton expert-kernel substitution is used by the new MXFP8 adapter.
HIP activation quantization remains a native AITER primitive; this is not a claim
that every operation is written in FlyDSL. MXFP4 is never sent to the MXFP8 packer.
The native MXFP4 callback preserves its arithmetic. Its shared-expert scheduling
can now be selected through the Raon stream switch as described below.
Quark describes/loads the MXFP4 checkpoint; it is not inserted into the original
MXFP8 checkpoint loader. MXFP8 retains its released E4M3/E8M0 expert bytes. The
compact loader no longer requires the old Triton GEMM flag for the FlyDSL backend.

The NVIDIA `(10,3)` / Transformer Engine static-padding path is preserved. A
separate ROCm graph dispatcher captures the already GPU-routed FlyDSL path,
instead of pretending AMD supports the NVIDIA grouped/TE chain. No per-expert
256-row static padding is required. A new local graph can only be captured
outside both breakable capture and replay; existing graphs can be replayed in
eager-break callbacks. Capture errors propagate. Unsupported configurations and
the bounded cache limit emit an INERT witness. The candidate only retains eight
shapes per layer, each with 1..128 tokens, and requires TP1/DP1 without EP.

## Numerical and performance questions to validate

- The early `RAON_MXFP8_FLYDSL_FUSED_A2=0` staged path materializes BF16 and
  measured 1.36..1.65% relative RMS versus the legacy Triton realization.
  The candidate profile now sets this flag to one: stage1 fuses FP32-to-MXFP8
  requantization and uses the original BF16 probability fold. Matched Torch
  gold passed (at most 0.118% relative RMS); legacy Triton comparison passed
  (at most 0.031%). Cross-device bitwise equivalence is not claimed.
- Raon's intermediate width 1088 is not divisible by the chosen 256-wide
  stage1 tile. Gate/up halves are padded separately to 1280, and stage2's K
  is padded consistently. Payload padding is zero; E8M0 padding is 127 (unit
  scale). This increases GEMM work and resident packed-weight memory; graph
  replay may or may not compensate. No speedup is claimed without measurement.
- Refit removes packed caches and graphs. First use, packing, JIT and graph
  warmup must remain outside performance timing.
- MXFP4 validation checks AITER metadata on native forward, before capture and
  independently of the outer-graph switch. AITER registry/tuning
  changes require a new check, and a profiler/runtime witness is still needed
  to confirm actual kernels. The wrapper rejects non-AITER Quark backends,
  non-A4W4 mode and a non-FlyDSL stage selection.

## Review-only activation fragment

### Dense MLP, projection and LM head audit

The default BF16 dense/shared MLP and RDA/MLA linears already use
`UnquantizedLinearMethod` and the ROCm `rocm_unquantized_gemm_impl` dispatcher.
The default BF16 LM head reaches the same dispatcher through
`LogitsProcessor._apply_head` and `UnquantizedEmbeddingMethod.apply` (or an
excluded-layer `UnquantizedLinearMethod`). This is existing AMD optimization,
not a missing CUDA-only linear implementation. The exported MXFP4 checkpoint
changed routed experts only; the other tensors were preserved.

The dispatcher checks batch-invariant mode first, then eligible gfx950
`wvSplitKrc`, a small AITER Triton shape whitelist, HIP skinny GEMMs, and AITER
`tuned_gemm.tgemm.mm` when AITER linear is enabled. Its internal selection can
still choose a library GEMM. AITER selection is not proof of FlyDSL execution.
The tuned result must be examined per real M/N/K, especially prefill-sized M.

`RAON_MLP_PROJ_PERSISTENT`, `RAON_MLA_PROJ_PERSISTENT` and
`RAON_HEAD_GEMM_PERSISTENT` use the parity-oriented Triton persistent matmul.
They do not select AITER/FlyDSL and have no established performance gain here.
`RAON_MLP_SITU_NATIVE` substitutes an FP32 reference expression for the existing
native SiTU op. The FP32-head switch changes precision and bypasses the normal
BF16 head path. These switches are not included in the performance candidate.

Optimization work to evaluate separately: shape-specific AITER tuned GEMM rows,
dense gate/up GEMM + SiTU fusion, projection/epilogue fusion, and LM-head
small-batch versus prefill GEMM selection. No such rewrite or speedup is claimed
by this audit; no serving precision was changed.

### Complete serving command for review

`print-candidate-command.sh` emits a command and never starts it:

```bash
bash docker/raon/print-candidate-command.sh mxfp8 on 0 8000
RAON_MODEL=/models/Raon-1-mini-MXFP4 \
  bash docker/raon/print-candidate-command.sh mxfp4 on 0 8000
# SD off keeps all other settings and omits the three speculation flags.
bash docker/raon/print-candidate-command.sh mxfp8 off 0 8000
```

Run the printer in the intended Linux checkout so its emitted source path is
valid. `RAON_MODEL` can name a local copy of the matching checkpoint.
Before executing a reviewed command, set `RAON_PYTHON` to an environment with
this candidate overlay installed. The previously published image alone does
not contain these uncommitted additions. Preserve its AMD-SMI `PYTHONPATH`.
Target/draft revisions are pinned. The official model card pairs these IDs;
their vocabulary, hidden size, target feature dimensions and block size match
the SD7 configuration. The requested draft ID is preserved, and bounded
release smokes exercised this pair. Full-model accuracy remains unmeasured.

The global scheduling mode is breakable PIECEWISE HIP graph. It requires
`VLLM_USE_BREAKABLE_CUDAGRAPH=1`; this model is not a compiled-submodule model.
`compilation_config.mode=0` disables whole-model torch.compile, not HIP graphs.
The original mode=3 command without the breakable switch was observed falling
back to graph NONE and was rejected. `RAON_EXPECT_CUDAGRAPH_MODE=PIECEWISE` and
the smoke's post-initialization check guard the corrected configuration. The
three local graph/stream flags remain enabled by the candidate profile.
The printer also makes the inherited MLA H40-to-H64 padding explicit and pins
the previously implicit `RAON_MOE_GROUPED_GRAPH=1`,
`RAON_KDA_DECODE_GRAPH_MAX=128` and `RAON_KDA_DECODE_GRAPH_SHAPES=8` defaults.
`RAON_MXFP8_FLYDSL_FUSED_A2=1` is the new numeric correction switch; it affects
MXFP8 only. The old `RAON_ROCM_KDA_FUSED_DECODE=1` compatibility switch is
inherited from the profile, but the explicit `...DECODE_BACKEND=flydsl` selector
takes precedence. MXFP4 inherits `VLLM_ROCM_USE_AITER_MOE_SITUV2=a4w4`.
This combination passed finite release output and structural checks; complete
performance and quality acceptance remain pending. A graph failure must be reported; do not
substitute eager mode.
Local MoE graphs cover 1..128 tokens; local KDA graphs default to 128 tokens
and eight shapes. The graph switches do not remove those bounds, mixed-batch
eligibility rules or native shared-stream scheduling constraints.

Eight GPUs mean eight separate invocations, one GPU visible in each process,
TP1/DP1 each, ports 8000..8007. The printer's GPU index is relative to the
devices assigned by the scheduler. Do not override a partial allocation to use
other devices. This review does not submit Slurm work or start servers.

ISL/OSL 8192/1024, 32768/2048 and 131072/2048, the measured token-based 50%
prefix-hit ratio, per-replica warmup, request distribution and EOS-respecting
sampling belong to the client workload. Server prefix caching alone cannot
guarantee that ratio. No client measurement is launched by this command.

Steering cost was measured separately after warmup on MI355X. The deployed
plain-Torch operation inside a graph took 14.7..16.3 us per steered layer at
1..128 tokens, 33.0 us at 1024 and 264.4 us at 16384. Per-step alpha-buffer
refresh took about 29..32 us. There are five steered layers. These isolated
costs include replay/launch overhead and cannot be converted to a model TPS
regression percentage. A separately torch-compiled pointwise variant measured
5.1 us at 1..128 and 38.9 us at 16384, indicating fusion opportunity; that
variant is not silently enabled in the breakable-graph serving profile.
Prefix cache isolation can also affect mixed-effort workloads, and effort
changes output lengths. Fixed-effort throughput must be measured separately.

```bash
export RAON_RUNTIME_PROFILE=mxfp8-flydsl  # or mxfp4-flydsl for the Quark checkpoint
source docker/raon/runtime-flydsl-env.sh
```

Do not add `--enforce-eager`. These are candidate values for review, not a
validated production configuration. The default profiles remain unchanged.
Both Dockerfiles also install this profile and the entrypoint accepts its two
names. The fragment sets `RAON_MOE_BACKEND=flydsl`, the old MXFP8 Triton GEMM flag
to zero, `RAON_MXFP8_FLYDSL_FUSED_A2=1`, `RAON_KDA_PREFILL=fused`,
`RAON_ROCM_KDA_DECODE_BACKEND=flydsl`, and all three
graph/stream switches to one. It retains the appropriate checkpoint loader and
MXFP4 A4W4 activation selection from its underlying precision profile.

`RAON_KDA_DECODE_GRAPH` now dispatches to a separate `rocm_kda_graph.py` from
the ROCm adapter. Unlike the original CPR-only graph, it supports pure AR and
pure speculative decode batches (TP1/DP1, bounded to 128 actual tokens and
eight shapes by default; existing MAX/SHAPES variables adjust those bounds).
It leaves prefill and mixed batches on the existing AMD implementation with an
INERT witness. The legacy step body after the new opt-in guard is AST-identical
to the published adapter. Exactly
one warm execution advances the live recurrent/conv state; capture records the
next replay without executing another decode, and the first call returns the
warm execution's output. A replay refreshes state indices, accepted counts and
other tensor metadata. A FULL enclosing graph can capture the raw chain; local
capture is never nested inside breakable capture/replay.

CPR means **chunk-pinned recurrence** in KRAFTON's source. It is a numerical
parity mode: resynchronize the recurrent state with BF16 chunk-boundary
snapshots every 64 tokens to reproduce the trainer's computation. The original
graph helper requires it and excludes SD. Generic HIP graph capture does not
need CPR, and this candidate does not enable or port that numerical mode.

The installed AITER contains fused conv/recurrent/gated-RMSNorm AR/SD decode,
but its public KDA call lacks Raon's independent write gate. The new
`rocm_aiter_rda_kernel.py` is an MIT-licensed local derivative with a ratio input
applied to V **after** convolution/SiLU and with the old BF16 boundaries. Beta's
erase term is unchanged. `rocm_aiter_decode.py` binds the actual cache/acceptance
metadata; no AITER package or FlyDSL compiler edit is needed. See
[`ROCM-DECODE-PROVENANCE.md`](../../vllm/models/raon1_mini/ROCM-DECODE-PROVENANCE.md)
for the source hash and license. This is AITER-derived **AMD Triton**, not a
FlyDSL kernel. The installed FlyDSL GDN decode has scalar decay and cannot
replace KDA/RDA's per-channel decay unchanged. The AITER parallel-V
specialization is not included in this first derivative.

`RAON_ROCM_KDA_DECODE_BACKEND=aiter_triton` explicitly selects that older
single-kernel derivative. The old `RAON_ROCM_KDA_FUSED_DECODE=1` boolean also
selects it when no new backend selector is set.

The current `flydsl` candidate instead uses AITER's public FlyDSL causal-conv
update API, a local FlyDSL RDA recurrence adapted from the installed GDN verify
schedule, and a local FlyDSL sigmoid-gated RMSNorm. The recurrence computes
per-K-channel decay, keeps recurrent state FP32, applies the RDA ratio after
conv/SiLU with the old BF16 seam, rolls back by accepted count, and stores every
candidate token's state. Replacing vector decay by the installed GDN's scalar
decay is never used. A separate upstream KDA branch also shows per-channel
FlyDSL decay; its older GTensor API is not assumed compatible with this wheel.

The new chain launches three kernels. It does not claim the fusion level of the
NVIDIA one-kernel decode or the older AITER Triton derivative. Its witness is
`RAON_FLYDSL_RDA_DECODE LAUNCHED ... conv=aiter_flydsl recurrence=flydsl
gated_norm=flydsl ... triton_compute=0`. Only pure AR/SD batches are specialized;
mixed/prefill batches retain the existing chain and emit INERT. Missing kernels,
unsupported architectures/layouts or compilation errors propagate instead of
silently selecting Triton. Serving rejects `COMPILE_ONLY=1`.

The new recurrence compiled and linked for gfx950 at AR=1 / SD=2/4/8; gated norm
compiled with FP32 and BF16 weights (six specializations). This was offline
compile-only on a CPU login node using FlyDSL 0.5.0, Torch 2.15.0+cpu descriptors
and a ROCm 7.2.1 linker. Production ROCm 10.2 GPU execution, numerical correctness,
graph/cache lifetime and throughput were not validated by that offline test.
Subsequent GPU results are recorded below. No speedup is claimed.

Prefill uses the existing optimized Kimi-K3 HIP kernels, not a new FlyDSL
rewrite. The Raon adapter still gathers/scatters recurrent state around that
call. Kimi's newer direct paged-state/checkpoint-export integration is not
ported here, and a 50% prefix-hit workload still requires measured validation.

`RAON_MOE_SHARED_SIDE_STREAM` already controls the custom MXFP8 MoE's shared
launch/join. The opt-in ROCm FlyDSL branch now permits that same event dependency
pattern during capture without needing the NVIDIA experimental graph switch.
For MXFP4, `rocm_shared_stream.py` connects the flag to the native SharedExperts
owner instead of recomputing shared experts externally. Mode 1 gives that owner
a separate HIP stream and native synchronization events; mode 0 disables its
separate stream. Native internal-overlap and token-threshold constraints remain
in force. Conflicting native disable settings are reported. Enqueue/wait
witnesses distinguish configuration from an actual recorded join; a timeline
is still needed to establish useful GPU overlap.

Historically, mode 0 in the Raon variable did not by itself disable MXFP4 native
overlap, which is controlled by `VLLM_DISABLE_SHARED_EXPERTS_STREAM` and the
native runner's policy. The new mapping is opt-in with `RAON_MOE_BACKEND=flydsl`
so it does not silently change existing legacy evaluations.

## Validation tools

CPU control flow only, no Torch/GPU arithmetic proof (28 checks, including AR/SD
state advancement, metadata refresh and native shared-expert scheduling):

```bash
.venv/bin/python tools/raon_validation/test_flydsl_dispatch.py
```

GPU checks for an idle single gfx950 GPU (executed finite cases are listed below):

```bash
source docker/raon/runtime-env.sh
export RAON_MXFP8_FLYDSL_FUSED_A2=1
.venv/bin/python -m tools.raon_validation flydsl-moe \
  --moe-full-shape --output /evidence/flydsl-mxfp8-gold.json
.venv/bin/python -m tools.raon_validation aiter-rda \
  --decode-backend flydsl --draft-counts 1 3 7 \
  --output /evidence/flydsl-rda-gold.json
```

This records the selected native fused-A2 and legacy FP32-intermediate Torch
gold, Triton comparison, and three graph replays with changed inputs/routes/
probabilities. Setting the fused-A2 flag to zero explicitly checks the older
BF16-intermediate candidate instead.
Its BF16 screen remains relative RMS 0.01 / relative peak 0.02 / absolute 1e-5;
graph versus the same eager candidate must be exact. A failed screen blocks
promotion. It does not test a full model, prefix caching, or SD end to end.

The existing `tools.raon_validation.moe_gold` tool remains the independent
MXFP4 baseline/port comparison with BF16- and FP4-intermediate references.
The new MXFP4 graph wrapper additionally needs an actual model run; the existing
tool alone cannot certify its model integration.

`aiter-rda` compares the selected candidate with the old AMD chain and independent
Torch convolution/RDA/norm gold for AR and SD lengths 2/4/8, every accepted
count, changing/reordered state indices and inputs. It checks updated recurrent
rows separately, unchanged rows exactly, full conv histories, and graph versus
the same direct candidate exactly. `--decode-backend aiter_triton` checks the
older fused derivative instead. The same 1% RMS/2% peak screen applies to
non-exact comparisons. Finite synthetic coverage does not establish all batch
sizes, NULL/PAD suffixes, full scheduler lifetimes or model quality.

The [MI355X record](../../tools/raon_validation/results/mi355x-flydsl-20261125.json)
contains the executed diagnostics:

- FlyDSL RDA: 45 AR/SD cases across batches 1/3/16, draft counts 1/3/7, every
  acceptance count, reordered indices and exact graph/cache replay. A physical
  SD index addressing bug was found and fixed. Worst output RMS versus Torch
  gold was 0.307%; recurrent-state RMS was 0.165%.
- MXFP8 fused-A2 expert rows: Torch and legacy comparison passed. Actual model
  combine/shared-stream/outer-graph call sites also passed on a synthetic
  fixture, including exact replay and a device counter checking shared execution.
- Native MXFP4 component: Torch gold and graph replay passed the existing
  1% RMS/2% peak numerical screen. Bitwise gates remain FAIL. Eager repeats of
  the native atomic path are also non-identical; maximum graph/direct RMS was
  0.435%. Shared execution counts passed. This fixture does not load a Quark
  checkpoint or initialize the full FusedMoERunner.
- The older AITER Triton RDA candidate, fused HIP KDA prefill and native rollback
  checks passed their finite diagnostic cases.

These earlier component checks did not run full checkpoint serving. The later
release smokes are recorded above; prefix-cache workloads, model-quality
evaluation and throughput measurement remain pending. Native
MXFP4 bitwise replay remains an explicit limitation; the exact gate was not
relaxed to make it pass.
