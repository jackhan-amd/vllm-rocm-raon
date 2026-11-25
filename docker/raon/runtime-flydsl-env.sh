# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Review-only optimized-path candidate. GPU correctness/performance not accepted.
case "${RAON_RUNTIME_PROFILE:-mxfp8-flydsl}" in
  mxfp8|mxfp8-flydsl)
    source "$(dirname "${BASH_SOURCE[0]}")/runtime-env.sh"
    ;;
  mxfp4|mxfp4-flydsl)
    source "$(dirname "${BASH_SOURCE[0]}")/runtime-mxfp4-env.sh"
    ;;
  *) echo "Unsupported FlyDSL Raon profile: ${RAON_RUNTIME_PROFILE}" >&2; return 2 ;;
esac
export RAON_MOE_BACKEND=flydsl
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export RAON_EXPECT_CUDAGRAPH_MODE=PIECEWISE
# The FlyDSL backend consumes checkpoint bytes without executing the old GEMMs.
export RAON_MOE_MXFP8_W8A8_TRITON=0
export RAON_MXFP8_FLYDSL_FUSED_A2=1
export RAON_KDA_PREFILL=fused
export RAON_ROCM_KDA_FUSED_DECODE=1
export RAON_ROCM_KDA_DECODE_BACKEND=flydsl
export RAON_KDA_DECODE_GRAPH=1
export RAON_MOE_OUTER_GRAPH=1
export RAON_MOE_SHARED_SIDE_STREAM=1
# Refresh MLA's phase/metadata in the breakable callback; keep native kernels.
export RAON_ROCM_MLA_LIVE_METADATA=1
# Mixed-policy MXFP4 uses native FlyDSL A16W4 linears and the MXFP8 LM head.
export RAON_ROCM_DENSE_LINEAR_BACKEND=flydsl
