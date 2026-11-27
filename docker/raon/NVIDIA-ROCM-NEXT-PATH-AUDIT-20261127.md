# Remaining NVIDIA/ROCm optimization boundaries

Read-only comparison against KRAFTON's Raon revision
`7dcdc8a35f1d76a03fa74af6c022ff6be0ef3bf0` and the ROCm fork after native
mixed RDA support. The RDA improvement used Kimi HIP plus CUDA RDA semantics
and workload partitioning; it was not a PTX translation. CUDA, CuTE and
NVIDIA-targeted Triton paths all deserve comparison, with their implementation
language distinguished from their actual device requirements.

## Priorities

| Priority | Source-proven boundary | ROCm work to investigate |
| --- | --- | --- |
| 1 | NVIDIA TRTLLM GEMM2 finalize includes unpermute/fold/sum; current FlyDSL stage2 returns per-slot rows | Preserve MFMA expert GEMMs while reducing the finalize/combine/shared-join boundary |
| 2 | NVIDIA RDA helpers fuse stable log-sigmoid ratio and post-conv V scale; current native HIP still consumes an external ratio | Pass beta/delta into the native prologue and eliminate the external seven-operation Torch chain |
| 3 | NVIDIA trainer mHC connects aggregate and branch RMSNorm | Preserve AITER MFMA project and investigate aggregate/branch-norm fusion |
| 4 | NVIDIA MLA overlaps gate projection and attention, then uses a compiled gate | Investigate gate projection overlap and native sigmoid/multiply at the existing Kimi/AITER boundary |
| 5 | Reasoning-effort steering precedes mHC mix | Consider a mix epilogue, but this is a common fusion opportunity, not a demonstrated missing NVIDIA optimization |

These are identified boundaries, not implemented optimizations or measured
speedups. Prior block-profile fractions do not establish the cost of each
boundary. MoE occupied about 34/49% of the earlier SD-off instrumented target
graph at C1/C32; RDA ratio alone had 126 recorded calls per graph, about 0.600 ms
there. Observer overhead and changed kernels prevent treating these as exact
removable time or predicting another 2x model gain. A current full-model trace
is needed after the newly connected paths are enabled.

## Exact consumers and constraints

1. **MoE finalize:** NVIDIA `mxfp8_w8a8_trtllm.py:718-724` enables finalize.
   ROCm [stage2](../../vllm/models/raon1_mini/mxfp8_flydsl.py) uses
   `return_per_slot=True`, followed by
   [ordered combine](../../vllm/models/raon1_mini/rocm_flydsl_combine.py)
   and the shared join in `model.py`. Preserve stable expert-ID FP32 addition,
   routed BF16 rounding, and the subsequent shared addition. NVIDIA's finalize
   may fold probabilities or sum in a different order; substituting it is not
   a numerically neutral change. Measure the boundary before changing GEMMs.
2. **RDA ratio:** upstream `kda.py:134-166` implements `_rda_scale_v_kernel`;
   its CUDA SD kernel also supports raw-delta ratio modes. Current
   [kda.py](../../vllm/models/raon1_mini/kda.py) computes ratio at lines 48-52
   and passes it at the input boundary. Native HIP multiplies V after conv/SiLU.
   Keep stable log-sigmoid math, independent erase/write gates and the
   BF16 conv-V then scaled-V seams. Test extreme finite logits and accepted
   rollback/replay; do not move the scale before convolution.
3. **mHC branch norm:** upstream `mhc.py:1219-1250,1609-1615` contains the
   aggregate/norm trainer path. Native
   [gate](../../vllm/models/raon1_mini/rocm_mhc_gate.py) writes BF16 aggregate;
   `model.py:1884-1890` applies the branch norm separately. It does not currently
   support `branch_norm_weight` just by enabling a trainer option. First prove
   the BF16 aggregate and normalization boundaries match. This is different
   from the [rejected scalar project/head fusion](MHC-PROJECT-HEAD-SINGLE-CTA-20261127.md).
4. **MLA gate:** upstream `mla_attention.py:111-114,524-544` contains the
   compiled gate and parallel gate projection. Current Kimi AMD `mla.py:304-307`
   and [split wrapper](../../vllm/models/raon1_mini/rocm_mla_split.py) project
   the gate after attention, then sigmoid/multiply. Kimi/AITER attention itself
   is already native. Keep the BF16 sigmoid seam, graph stream joins and single
   KV update; prove the overlap does not duplicate work.
5. **Steering:** `effort_steering.py:406-417` performs cast/mul/add/cast/where
   before mHC mix. Keep the BF16 result and alpha-zero bit preservation. No
   separate NVIDIA-only optimized kernel was established for this boundary.

## Already present or previously rejected

NVIDIA CuTE GEMM13 + SiTU + probability fold + A2 quantization has an existing
ROCm counterpart in `mxfp8_flydsl.py:218-250`, under
`RAON_MXFP8_FLYDSL_FUSED_A2`. Confirm actual activation, rather than listing it
as absent or separating SiTU back into a standalone kernel. Native MXFP4 and
compact MXFP8 packing remain different contracts.

Dense/projection already selects native AITER/FlyDSL in `rocm_dense.py`.
Repeating an undirected tile-N sweep has less evidence than the boundaries
above. The previous M16/direct-route GEMV losses and the new scalar single-CTA
mHC loss remain negative evidence. Any new schedule should preserve the
successful MFMA paths and answer a specific measured bottleneck.

The [mixed RDA report](NATIVE-RDA-MIXED-AND-DISPATCH-20261127.md) records the
completed implementation and its limits; the [whole-model budget](LAYER-BUDGET-20261127.md)
records earlier instrumented fractions. No new node allocation or GPU trial
was made for this source audit.
