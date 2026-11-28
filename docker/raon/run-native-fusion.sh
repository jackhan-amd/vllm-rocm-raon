#!/usr/bin/env bash
set -euo pipefail
source /opt/raon/runtime-native-fusion-env.sh
exec /usr/local/bin/uv run --no-project --python /opt/raon/.venv/bin/python python \
  -m vllm.entrypoints.cli.main "$@" \
  --worker-extension-cls geometry_speed_extension.GeometryExtension
