# Other kernels and next optimization candidates

Current-image follow-up: see
[the November 27 profile and fusion review](FRESH-PROFILE-AND-FUSION-CANDIDATES-20261127.md).
It updates the candidate ranking with valid GPU traces, direct mHC operator
mapping, and an explicit profiler spawn workaround. The historical evidence
below remains unchanged.

The next candidates are copy/metadata/graph scheduling at C1 and RDA verification
at C32. An arithmetic-only MLA A4W4 BMM rewrite is lower priority. This review
reuses six historical traces and audits current source; it does not measure a
new speedup, allocate GPUs, or change the serving profile.

## Evidence boundaries

- The original November 20 traces reproduce the old Other/unclassified totals
  exactly: MXFP4 **28.8763%**, MXFP8 **38.6222%**. Their workloads and engine-step
  counts differ; these percentages are not a precision throughput comparison.
- The more recent November 26 MXFP8 traces contain 32 profiled engine steps,
  mixing prefill and decode, for SD off/on at C1/C32. They precede the native mHC
  finisher and native DFlash attention. They are **not** profiles of published
  image `sha256:f683e05a88ace2c842316fca4772f0f5fe72c1b5b0f2db69b7bb127a46ac9026`.
- Every percentage below divides by summed GPU kernel durations. It is neither
  end-to-end wall time nor a critical-path speedup estimate.
- TraceLens `c80f7de688c485fd6924da9a4de345a785bac60e` reports
  `GRAPH_UNDER_RECORDED` for all four recent traces. Replay fractions are 73.58%,
  83.26%, 69.58%, and 79.32% for on/C1, off/C1, on/C32, and off/C32. Kernel
  names/times survive, but most graph kernels lack owning operation/layer stacks.
  The prescribed deterministic fallback ran; agentic attribution and roofline
  optimization predictions were skipped. Its heuristic impact estimates are
  not used here.

## What the original Other category contained

The old classifier recognized selected name substrings such as KDA, attention,
GEMM and quantization. Everything else became Other. Reclassification uses
kernel names and, where available, an exact External-ID link to a CPU operation.
It identifies operation types, not necessarily owning layers or redundant work.

| Component inside original Other | MXFP4 | MXFP8 |
| --- | ---: | ---: |
| Copy/cast | 5.862% | 10.842% |
| Named mHC kernels | 4.326% | 3.412% |
| Attention operator identified; backend unresolved | 4.270% | 3.439% |
| Generic elementwise | 3.577% | 5.752% |
| Sampling/SD helpers | 3.184% | 2.500% |
| Normalization, including fused quantization | 2.850% | 2.328% |
| Generic reduction | 2.276% | 1.843% |
| Index/gather/scatter | 0.351% | 5.570% |
| Remaining operation types | 2.180% | 2.936% |
| **Original Other total** | **28.876%** | **38.622%** |

Thus Other did not mean unoptimized kernels. For example, generic `_fwd_kernel`
entries link to `vllm::unified_attention_with_output`, and some small GEMM helpers
link to native AITER operators. The classifier also fails to recognize newer
native kernel names; changes in Other share alone do not establish a regression.

## Recent MXFP8 traces: joint candidate comparison

| Kernel family | SD on C1 | SD on C32 | SD off C1 | SD off C32 |
| --- | ---: | ---: | ---: | ---: |
| Copy/cast | 19.80% | 10.85% | 17.83% | 16.94% |
| Generic elementwise | 11.34% | 6.06% | 14.99% | 10.03% |
| Generic reduction | 4.15% | 2.54% | 5.68% | 4.51% |
| **Above three combined** | **35.29%** | **19.45%** | **38.50%** | **31.48%** |
| RDA/conv recurrence | 3.81% | 19.64% | 1.35% | 6.73% |
| MLA A4W4 batched GEMM | 0.54% | 0.66% | 0.70% | 0.63% |

These are name-based inventories with attribution gaps. The RDA row is not the
complete RDA layer cost. Likewise, the MLA BMM row excludes callback, metadata,
copy, projection and other MLA work. C1 copy/cast alone has 17,524 launches in
the SD-on trace; its cost warrants ownership analysis before another large GEMM
rewrite. Some copies/reductions belong to the old mHC finishing path and have
already been replaced by the native finisher. Do not count them as remaining
opportunities without a current trace.

## Ranked work

### 1. Copy, metadata and graph scheduling

Attribute remaining copies and reductions to their current owners, then remove
redundant conversions/staging or fuse adjacent work where numerical boundaries
allow. Keep live accepted counts, state indices and cache ownership correct.

Current source gives specific scheduling questions:

- DFlash selects its full graph only when `decode_mode() == FULL`; its source
  explicitly does not support PIECEWISE. The reviewed PIECEWISE configuration
  therefore does not establish draft full-graph replay. A startup capture message
  alone is not evidence of activation. This is a graph-mode audit, not permission
  to begin the deferred Torch.compile experiment.
- The draft metadata builder can rebuild metadata eagerly when its backend cannot
  update it in place. Measure this cost on the current named native backend
  before implementing a live-update builder.
- Split MLA still has a Python attention callback and output copy around native
  kernels. Retain this scheduling candidate even though the BMM arithmetic is
  small.
- The runtime already sets KDA graph maximum 512, shape capacity 64, MoE graph
  maximum 512/capacity 64, and warmup batches `1,8,32`. Existing logs include INERT
  cases for new/mixed shapes. Count actual replay hits/misses and warmed shapes;
  blindly increasing environment variables does not prove activation.

### 2. C32 RDA verification: shared preparation and state IO

Current FlyDSL recurrence source repeats the same head/token Q/K normalization,
gate/decay and beta preparation across V-row groups. Test preparing these values
once per head/token, possibly fused with the front end to avoid an extra launch.
Preserve FP32 intermediates and existing BF16 rounding boundaries. Measure saved
computation against extra temporary-buffer traffic before adopting it.

Separately identify the cost of writing speculative full-state checkpoints.
Those writes support rollback; removing them requires a correct state design,
not simply dropping stores. Validate every accepted count 0 through 7, reordered
and padded slots, untouched cache entries, and replay metadata refresh against
Torch gold. Capture/warmup must advance live state exactly once.

### 3. Defer arithmetic-only MLA BMM replacement

The measured 0.54%/0.66% SD-on kernel shares do not support prioritizing this
rewrite as a route to a large end-to-end improvement. Keep its callback/metadata
work in candidate 1. Preserve E2M1/E8M0 formats and scales if revisited. The prior
72-configuration generic BF16 FlyDSL screen also did not justify widening dense
dispatch; do not activate it merely because it is native.

## Next evidence required

Pin the published image above and gather graph capture definitions plus replay
traces, shapes and call stacks for SD off/on C1/C32. Separate prefill and steady
decode, and record graph hit/miss counts. Profile outside timed measurements.
Retain the reviewed precision, KV cache, prefix cache, TP1/DP1 and sampling
settings; never use `--enforce-eager` or silently change graph modes. No such run
was launched by this review.

Machine-readable inputs, percentages, provenance and ranked actions are in
[`next-optimization-candidates-20261126.json`](../../tools/raon_validation/results/next-optimization-candidates-20261126.json).
The complete CPU-only evidence bundle, including per-kernel CSV/JSON inventories,
TraceLens workbooks and deterministic fallback, is retained on Thera at
`/home/jackhan/bench-results/raon-other-kernels-20261126/analysis-evidence.zip`.
