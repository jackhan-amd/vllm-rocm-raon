# Native DFlash2 attention candidate

`ROCM_RAON_DFLASH` is an explicit, experimental backend for Raon's DFlash2
draft: BF16, 20 query heads, 5 KV heads, D128, noncausal eight-query blocks and
a 2048-token sliding window. It uses the existing native HIP NHD cache writer,
a FlyDSL paged gather/dequantization kernel, and AITER CK HIP attention.
Unsupported real draft configurations raise; this backend has no Triton fallback.
It is not selected by default.

Keep the reviewed serving command and replace its attention configuration with:

```bash
--attention-config '{"backend_per_kind":{"mla_attention":"ROCM_AITER_MLA","sliding_window":"ROCM_RAON_DFLASH"},"mla_prefill_backend":"ROCM_AITER_FA"}'
```

This changes the draft attention group. Target MLA and RDA retain the previous
native paths. No `RAON_NATIVE_DRAFT` environment variable or registration wrapper
is needed. The target remains `--dtype auto` with its MXFP8 checkpoint metadata,
FP8 E4M3 KV and prefix caching. Previous native mHC/finisher, ordered MoE combine,
direct MoE graph and split MLA graph options remain separate opt-ins.

## Precision and cache contract

Historical FP8 K/V entries are multiplied by their live scalar scales and rounded
to BF16. For multi-query attention, current K/V retain the original BF16 suffix,
matching the prior ROCM_ATTN path. The single-query Opus diagnostic reads the
cached suffix, matching the prior decode contract. Q remains BF16; softmax and
accumulation belong to the native attention implementation. Weight precision is
unchanged. MXFP4-target end-to-end serving has not been tested in this round.

The gather accepts strided NHD cache and strided packed Q/K/V. Each physical page
gets a separate buffer descriptor with a 64-bit base address; flattening the
whole cache would exceed FlyDSL ABI/buffer bounds for production-size KV pools.
Sequence lengths, query boundaries, block mappings and scales are read on every
replay. Gather and attention do not update the live cache; the existing writer
does so once through the normal vLLM cache-update path.

## Validation

The registered component diagnostic checks native cache bytes against Torch
quantization, untouched slots, attention against independent FP32 Torch gold,
expired NaNs, FP8/BF16 cache, nonunit zero-dimensional scales, strided QKV/cache,
partial pages, reordered mappings, padding and changed-input graph replay.
Fixtures include page size 1280 and physical cache addresses beyond 4 GiB.

```bash
for case_id in 0 1 2 3 4 5 6 7; do
  .venv/bin/python -m tools.raon_validation draft-native \
    --draft-case "$case_id" --output "evidence/draft-native-${case_id}.json"
done
```

`auto` uses CK for eight-query fixtures and native Opus for mixed lengths.
Opus is a diagnostic here, not the serving backend. Direct CK single-query
windowed attention produced NaNs at K=2048 in this pinned AITER build; no such
configuration is admitted by the serving candidate. Query-specific overlapping
Opus spans passed gold but scaled poorly, so they were not adopted for serving.
The eight final component fixtures passed. The named backend also passed three
short generation smokes and a 3022-token input, preserving the existing native
target-path graph witnesses. This is not a full accuracy evaluation.

## Serving screen

One MI355X per independent TP1/DP1 server, MXFP8 target plus DFlash2 with seven
draft tokens; the immutable parent image is
`sha256:21ae335dd3d6aa57d0d75b5ee044c43336019d3ee726e5966c9eba54c729a329`.
Both profiles retain graph execution and the previously validated native options.
No eager fallback or Torch.compile candidate was introduced.

Each condition used 128 GSM8K prompts, C=1/8/32, thinking enabled, temperature 1,
top-p .95, top-k 20, EOS respected, at most 4096 new tokens and separate warmup.
These variable-length speed prompts do not reproduce the official prompt IDs or
the separate fixed ISL/OSL workload with 50% prefix hits. Output TPS includes
natural-length and acceptance effects; modest ratios are not a variance study.

| C | Prior native SD on | Native draft SD on | Ratio | Status |
| --- | ---: | ---: | ---: | --- |
| 1 | 284.7 | 296.8 | 1.043x | Valid single measurement |
| 8 | 1257.1 | 1439.9 | 1.145x | Valid single measurement |
| 32 | 2222.5 | 2168.2 | 0.976x | Valid C32-only retry after complete warm cache |

The initial C=32 trial is retained as invalid because a FlyDSL launch cache
artifact was created during timing. The explicitly announced retry measured
only C32 after warmup with that cache populated; C1/C8 were not repeated.
The retry recorded no cache changes or JIT warnings. Each point contains 128
unique raw request IDs and validated output-token totals.
The isolated attention screen showed approximately 1.8–2.5x for K=4096 and query
length 8, including gather; that is not the end-to-end speedup above.

Aggregate acceptance length (1 + total accepted drafts / total draft steps):

| C | Prior native | Native draft |
| --- | ---: | ---: |
| 1 | 3.480 | 3.497 |
| 8 | 3.443 | 3.511 |
| 32 | 3.420 | 3.568 |

This is not another twofold end-to-end improvement. The candidate stays opt-in;
the C32 trial provides no evidence to enable it universally. Scratch is retained
per visited batch/token shape. Serving with a wider concurrency/shape sweep needs
a scratch-memory audit beyond this measured C1/8/32 workload.

Six additional GPUs screened 72 generic full-tile BF16 FlyDSL projection
configurations over six observed shapes and M=32/64/128/256. All numerical and
replay checks passed, but almost all were slower than the current runtime. A
roughly 4% LM-head result at M32 is insufficient to justify another default.
No wide-projection dispatch change was applied. Native alone is not proof of
better performance; future work needs suitable split-K/skinny schedules or the
remaining precision-preserving MLA A4W4 batched-GEMM replacement.

Raw evidence is under
`/home/jackhan/bench-results/raon-next-native-20261126` (Slurm 89383, TheraC80).
The first three failed serving starts are preserved separately from the passing
candidate and named-backend smoke. They exposed an overstrict metadata guard,
scalar-scale ABI handling, and the large-cache flattening bug, respectively.

Source commit: `fcee6105d3cc5e3d2814a10b87e205610caa9e7a`. The published image is
`docker.io/hanjack/vllm-rocm:raon-draft-native-fcee6105d3-20261126`, digest
`sha256:f683e05a88ace2c842316fca4772f0f5fe72c1b5b0f2db69b7bb127a46ac9026`.
The image's source-file hashes matched the committed build context, all eight
component fixtures passed in the new image, and its three generation smokes and
C32 serving point passed. See the [image receipt](image-draft-native-20261126.json)
and [validated measurements](../../tools/raon_validation/results/native-draft-20261126.json).
Slurm 89383 completed normally in 1:25:20; its owned containers were removed.
