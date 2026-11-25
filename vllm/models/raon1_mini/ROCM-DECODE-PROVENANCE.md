# AITER-derived RDA decode candidate

`rocm_aiter_rda_kernel.py` derives `_valid_spec_prefix` and
`fused_conv_recurrent_norm_kernel` from the installed AITER file
`aiter/ops/triton/_triton_kernels/gated_delta_net/decode/fused_conv_recurrent_norm.py`.
The original file SHA256 is
`fe15c27f37fb515a17592baeaca36504f92191f6638a934265be77a780c33b07`.
It was read from the existing pinned Raon image, without modifying AITER.
The copyright and MIT SPDX header are retained.

The derivative adds a per-token/head RDA ratio pointer and token stride, then
multiplies V after conv/SiLU. Erase strength beta is unchanged. It also rounds
conv Q/K/V through BF16 to preserve the existing Raon adapter's seams. The rest
of the original single-kernel AR/speculative cache and normalization logic is
retained. The higher-throughput parallel-V AITER specialization is not included
in this initial candidate. This is an AMD Triton kernel, not FlyDSL or HIPIFY.

An independent RDA write gate cannot be implemented by replacing KDA's beta:
that would also change the erase term. Scaling raw V before convolution would
also change the model because convolution/SiLU is nonlinear. Hence the local
post-convolution patch rather than a call to the unmodified KDA API.

`rocm_aiter_decode.py` binds pure AR and pure SD batches to this kernel and leaves
prefill/mixed batches on the existing AMD chain, with a witness. Unsupported
architecture, state layout, bias or conv width raises. GPU gold and serving
validation are required before enabling this candidate for reported results.

The 2026-11-25 finite GPU diagnostic passed 15 AR/SD cases for this older
single-kernel Triton derivative. It remains an explicit alternative.

## FlyDSL decode candidate

`rocm_flydsl_rda.py` adapts the K/V-lane recurrence and vector-copy helpers from
installed AITER's `aiter/ops/flydsl/kernels/gdn_decode_verify.py`, SHA256
`f93500381e5b9edaf297dd337c3dedbbc7551868c826d72f00153115a37d27e6`.
It calls that unchanged module's `_load_vec` / `_store_vec` helpers. The local
derivative retains the MIT license in its source so the installed overlay also
carries the license. The AITER package and FlyDSL compiler remain unchanged.

The changes are vector decay along K, FP32 V-major recurrent caches, post-conv
BF16 RDA value scaling, vLLM's accepted-count rollback and per-token candidate
slots, NULL/PAD suppression, and a separate FlyDSL sigmoid-gated RMSNorm.
The public AITER FlyDSL causal-conv update handles the existing conv history.
The selected pure AR/SD chain launches three FlyDSL kernels and never calls the
Triton recurrence or norm. Imports of AITER's device utilities do not imply a
Triton compute launch.

The separate KDA branch inspected at
`https://github.com/ROCm/aiter/tree/waqahmed/KIMI-K3_KDA-Decode` also demonstrates
per-channel decay in FlyDSL GDR; it is not assumed merged or installed. This
implementation uses the installed 0.5-compatible AITER helper API instead of
copying that branch's older GTensor/compiler API or replacing the installed
GDN's scalar decay unchanged.

`RAON_ROCM_KDA_DECODE_BACKEND=flydsl` selects this candidate. `aiter_triton`
selects the older single-kernel derivative above; `legacy` selects the original
AMD chain. The old boolean is honored only when the new selector is absent.
Graph stamps include the selector. Prefill/mixed batches retain the existing
implementation; this is not complete mixed-batch fusion coverage.

Offline compile-only checks on Thera's CPU login node generated and linked
gfx950 kernels for AR=1 and SD=2/4/8 plus gated norm with BF16/FP32 weight. They
used FlyDSL 0.5.0, Torch 2.15.0+cpu descriptors and the ROCm 7.2.1 host linker;
no GPU pointers were executed. That compile-only check did not establish
production ROCm 10.2 execution, Torch gold, graph lifetime or speed.

Subsequent MI355X/ROCm 10.2 GPU diagnostics found and fixed a rank-two SD
state-index bug: a scalar Tensor coordinate remapped an already physical
offset. Explicit scalar buffer loads now apply the declared row/column strides.
The corrected kernel passed 45 cases at batches 1/3/16 and draft counts 1/3/7
against independent Torch gold, existing kernels and exact graph/cache replay.
This finite coverage does not establish full checkpoint serving, NULL/PAD
suffix coverage, model quality, scheduler lifetimes or speed.

## MIT license

Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
