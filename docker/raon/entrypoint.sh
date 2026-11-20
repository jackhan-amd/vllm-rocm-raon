#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
source /opt/raon/runtime-env.sh
exec /opt/raon/.venv/bin/python -m vllm.entrypoints.cli.main serve "$@"
