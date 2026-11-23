# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Source this experimental profile only for the separate Quark MXFP4 checkpoint.
source "$(dirname "${BASH_SOURCE[0]}")/runtime-env.sh"
export RAON_MXFP8_COMPACT_LOAD=0
export RAON_MOE_TRAINER_REALIZED=0
export RAON_MOE_EXPERT_MXFP8_REALIZED=0
export RAON_MOE_MXFP8_W8A8_TRITON=0
export VLLM_ROCM_USE_AITER_MOE_SITUV2=a4w4
