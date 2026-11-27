# Native RDA mixed-batch wiring and shape dispatch

The opt-in HIP RDA family now supports AR with physical conv histories
3/4/6/10, mixed SD/AR, and mixed SD/AR/prefill. Prefill retains the existing
Kimi gfx950 fused HIP prologue/chunk implementation. Decode uses the selected
native HIP RDA kernel. No default serving profile was changed.

## Cache and normalization protocol

An expanded allocation does not change AR's effective history. The existing
`causal_conv1d_update` explicitly uses width minus one when accepted counts are
absent: AR reads/updates the **first three taps**, leaving the expanded tail
intact. The first candidate incorrectly used the last taps and shifted the
whole allocation; it failed the legacy comparison and was corrected. Independent
Torch conv/RDA/norm gold now checks the effective history and exact full pool.

Mixed dispatch gathers the scheduler's token mappings, executes SD separately,
and restores the original order. The non-speculative branch uses native AR when
there is no prefill; with prefill it retains the existing Kimi HIP chunk path.
Each branch produces normalized output, so scatter is not followed by a second
norm. Persistent, unregistered shallow proxies retain staged weight buffers;
registering them as child modules would create a cycle in the shallow-copied
module dictionary. Each kernel variant also retains its own staged buffers.
First mixed warmup must occur outside enclosing graph capture; an unprepared
capture raises explicitly. There is no repeated live state update for warmup.

The existing gather/scatter copies remain. This wiring establishes native
recurrence coverage; it does not fuse the entire mixed operation into one
kernel or remove every Triton operation from prefill's existing conv frontend.

## Explicit choices

Set `RAON_ROCM_KDA_DECODE_BACKEND` to:

| Value | SD implementation |
| --- | --- |
| `hip_rda` | Three-stage native HIP baseline |
| `hip_rda_fused_sd` | Single CTA/head conv + recurrence + norm |
| `hip_rda_vsplit_sd` | Shared prologue, eight V tiles/head recurrence, norm |
| `hip_rda_auto_sd` | Measured shape choices below; otherwise fused HIP |

All these choices use the native AR kernel for AR. `hip_rda_ar` retains its
AR-only contract and explicitly rejects pure SD.

Auto uses V-split only for full uniform `(batch, verify width)` in
`{(1,4), (1,8), (2,4), (2,8), (4,8)}`. Other shapes, including ragged spans,
use fused native HIP. This is an opt-in measured table, not a universal tuning
rule. The close B4/L4 result was not used to switch kernels. Host-known shapes
and token counts select the kernel; accepted counts and slot indices remain
device inputs refreshed for every graph replay. No Triton fallback is selected.

## GPU evidence

TheraC80 job `89711`, pinned image
`sha256:c41349b2bca3335d3a7eacceb6e4d9d21c4fa4602f7d24e454a8ccb9f739c281`.

- Expanded AR: batch1/32 with histories4/6/10, output/state versus independent
  Torch gold and legacy, exact full conv pool, two changed-input graph replays.
- Mixed SD/AR: batch4/32, all eight accepted counts, shuffled token order and
  changing slot indices, output/state versus the existing implementation and
  exact graph/direct output/state/conv comparisons.
- Mixed prefill: batch4/32, lengths3/129/257, alternating fresh/continuation
  initial-state flags and all accepted counts. These integration comparisons
  use the existing mixed implementation; independent per-component Torch gold
  is a separate check, not an independent full-model golden implementation.
- Real mixed-prefill kernel witness includes `sd_conv_recurrence_norm`,
  `kda_chunk_prologue<1024, false>` and `kda_chunk_fused<16,4,true,true,2>`.
  This is direct-call kernel provenance, not a graph latency measurement.
- Ragged SD: three-stage/fused/V-split at batch4/32, spans1..width, every
  accepted count, Torch gold, full conv history and exact graph replay.
- Auto: full batch1/2/4/8, ragged batch32, and mixed prefill129 at batch4.

The [receipt](../../tools/raon_validation/results/native-rda-mixed-20261127.json)
records these results and final NULL-padding checks. Actual checkpoint model
generation, Dynamo/runner integration, MXFP4/MXFP8 model evaluation and full-model
TPS remain unmeasured for this revision. The numerical screen was not relaxed.

## Performance interpretation

Single-layer warm operation measurements, not full-model throughput:

| Condition | FlyDSL | Best candidate | Operation gain |
| --- | ---: | ---: | ---: |
| B1, eight-position SD | 36.579 us | V-split 16.125 us | 2.27x |
| B32, eight-position SD | 320.497 us | Fused 148.161 us | 2.16x |

The [fused](NATIVE-HIP-RDA-SD-FUSED-20261127.md) and
[V-split/crossover](NATIVE-HIP-RDA-SD-VSPLIT-20261127.md) reports preserve the
separate measurement windows and cache-regime limitations. Fused uses 190 VGPRs
with zero spills; V-split recurrence uses 47 with zero spills. B1's original
40-CTA grid cannot occupy more than 40 of the device's 256 CUs at once, whereas
V-split exposes 320 CTAs. This is a scheduling bound, not measured CU busy or
proof that register pressure alone caused the difference.

The parallel mHC experiment found project about4 us and head/mix about2.4..2.8
us in separate warmed stage graphs. A single-CTA project/head prototype passed
gold but was about3x slower, and is explicitly rejected for performance.
Preserving MFMA and parallel work matters more than removing a launch at any
cost. See the [graph evidence](MHC-GRAPH-LATENCY-20261127.md) and
[negative result](MHC-PROJECT-HEAD-SINGLE-CTA-20261127.md).
