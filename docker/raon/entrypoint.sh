#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
case "${RAON_RUNTIME_PROFILE:-mxfp8}" in
  mxfp8) source /opt/raon/runtime-env.sh ;;
  mxfp4) source /opt/raon/runtime-mxfp4-env.sh ;;
  mxfp8-flydsl|mxfp4-flydsl) source /opt/raon/runtime-flydsl-env.sh ;;
  *) echo "Unsupported RAON_RUNTIME_PROFILE: ${RAON_RUNTIME_PROFILE}" >&2; exit 2 ;;
esac
exec /opt/raon/.venv/bin/python -m vllm.entrypoints.cli.main serve "$@"
