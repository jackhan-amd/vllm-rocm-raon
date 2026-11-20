# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture installed Kimi KDA implementations without assuming a newer AITER API."""

import ast
import hashlib
import importlib.metadata as md
import importlib.util
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
root = Path(importlib.util.find_spec("vllm").origin).parent
records = []
for path in root.rglob("*.py"):
    rel = path.relative_to(root)
    if not (
        "kda" in path.name.lower()
        or ("kimi" in path.name.lower() and "gdn" in str(path))
    ):
        continue
    data = path.read_bytes()
    tree = ast.parse(data.decode())
    functions = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(
                {
                    "name": node.name,
                    "signature": ast.unparse(node.args),
                    "line": node.lineno,
                }
            )
    target = out / "sources" / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    records.append(
        {
            "module": "vllm."
            + str(rel.with_suffix("")).replace("/", ".").replace("\\", "."),
            "path": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "functions": functions,
        }
    )
versions = {}
for name in ("torch", "triton", "vllm", "amd-aiter", "flydsl", "pybind11", "ninja"):
    try:
        versions[name] = md.version(name)
    except md.PackageNotFoundError:
        versions[name] = None
(out / "runtime.json").write_text(
    json.dumps(
        {
            "python": sys.version,
            "versions": versions,
            "vllm_root": str(root),
            "kda_sources": records,
        },
        indent=2,
    )
)
print(json.dumps({"versions": versions, "kda_source_files": len(records)}, indent=2))
