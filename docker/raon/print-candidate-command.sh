#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Print a reviewable command only. Never allocate resources or start a server.
set -euo pipefail

precision=${1:-mxfp8}
sd=${2:-on}
gpu=${3:-0}
port=${4:-8000}
case "$precision" in
  mxfp8) model=KRAFTON/Raon-1-mini ;;
  mxfp4|mxfp4-routed) model=jackhan-amd/Raon-1-mini-MXFP4 ;;
  *) echo 'precision must be mxfp8, mxfp4 or mxfp4-routed' >&2; exit 2 ;;
esac
case "$sd" in on|off) ;; *) echo 'SD must be on or off' >&2; exit 2 ;; esac
[[ "$gpu" =~ ^[0-7]$ ]] || { echo 'GPU must be 0..7' >&2; exit 2; }
[[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1024 && port <= 65535 )) || exit 2
profile_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

printf '# REVIEW ONLY: patched candidate vLLM must be installed in RAON_PYTHON.\n'
printf '# Release steering is implemented; see checkpoint-baseline.json for validation status.\n'
printf '# Finite native smokes are separate from full accuracy/TPS acceptance.\n'
printf '# Target source revision: 3f85cd63ea420100a84378f72ae162af86a441fd\n'
printf 'export RAON_RUNTIME_PROFILE=%q\n' "${precision%-routed}-flydsl"
printf 'source %q\n' "$profile_dir/runtime-flydsl-env.sh"
cat <<'ENV'
export RAON_MOE_BACKEND=flydsl
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export RAON_EXPECT_CUDAGRAPH_MODE=PIECEWISE
export RAON_MXFP8_FLYDSL_FUSED_A2=1
export RAON_KDA_PREFILL=fused
export RAON_ROCM_KDA_DECODE_BACKEND=flydsl
export RAON_KDA_DECODE_GRAPH=1
export RAON_KDA_DECODE_GRAPH_MAX=128
export RAON_KDA_DECODE_GRAPH_SHAPES=8
export RAON_MOE_OUTER_GRAPH=1
export RAON_MOE_GROUPED_GRAPH=1
export RAON_MOE_SHARED_SIDE_STREAM=1
export RAON_ROCM_MLA_LIVE_METADATA=1
export RAON_ROCM_DENSE_LINEAR_BACKEND=flydsl
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_LINEAR=1
export VLLM_ROCM_USE_AITER_MLA=1
export VLLM_ROCM_AITER_MLA_PAD_H40_TO_64=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_ROCM_USE_SKINNY_GEMM=1
# Preserve existing shape dispatch; this permits some AITER Triton GEMMs.
export VLLM_ROCM_USE_AITER_TRITON_GEMM=1
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=0
export VLLM_BATCH_INVARIANT=0
# Persistent flags select parity-oriented Triton, not native speed guarantees.
export RAON_MLP_PROJ_PERSISTENT=0
export RAON_MLA_PROJ_PERSISTENT=0
export RAON_KDA_PROJ_PERSISTENT=0
export RAON_MLP_SITU_NATIVE=0
export VLLM_KPAMO_FP32_LM_HEAD=0
export RAON_HEAD_GEMM_PERSISTENT=0
ENV
printf 'export ROCR_VISIBLE_DEVICES=%q\n' "$gpu"
# Emit variable references for evaluation by the eventual serving shell.
# shellcheck disable=SC2016
printf ': "${RAON_PYTHON:?Set the Python executable with the patched candidate installed}"\n'
# shellcheck disable=SC2016
printf '"$RAON_PYTHON"'
args=(
  -m vllm.entrypoints.cli.main serve "${RAON_MODEL:-$model}"
  --host 0.0.0.0 --port "$port"
  --tensor-parallel-size 1 --data-parallel-size 1
  --dtype auto
  --max-model-len 262144
  --max-num-batched-tokens 16384
  --max-num-seqs 128
  --gpu-memory-utilization 0.85
  --kv-cache-dtype fp8_e4m3
  --enable-prefix-caching
  --mamba-cache-mode align
  --attention-config '{"backend_per_kind":{"mla_attention":"ROCM_AITER_MLA"},"mla_prefill_backend":"ROCM_AITER_FA"}'
  --compilation-config '{"mode":0,"cudagraph_mode":"PIECEWISE"}'
  --reasoning-parser raon_1
  --enable-auto-tool-choice
  --tool-call-parser raon_1
  --default-chat-template-kwargs '{"enable_thinking":true,"truncate_history_thinking":false}'
)
if [[ "$precision" == mxfp8 ]]; then
  args+=(--revision 3f85cd63ea420100a84378f72ae162af86a441fd)
elif [[ "$precision" == mxfp4 ]]; then
  args+=(--revision 62479b0a6c1c9ef2fec51a29bed5c983bc86b7f7)
else
  args+=(--revision 74fe32713a200a95e00faa30a161e5bbb4b3abc6)
fi
if [[ "$sd" == on ]]; then
  args+=(--spec-method dflash --spec-model KRAFTON/Raon-1-mini-dflash2 --spec-tokens 7)
  args+=(--speculative-config '{"revision":"74f2835bd683458f6124f48784b078b06770d824"}')
fi
for arg in "${args[@]}"; do printf ' \\\n  %q' "$arg"; done
printf '\n'
