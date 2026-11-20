# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install this fork's Python integration over its matching ROCm native wheel."""

import ast
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import shutil
import sys
from pathlib import Path


def main():
    source = Path(sys.argv[1]).resolve()
    manifest = json.loads((source / "docker/raon/overlay-files.json").read_text())
    version = metadata.version("vllm")
    expected = manifest["base_commit"][:9]
    if f"g{expected}" not in version:
        raise RuntimeError(
            f"This Python overlay requires the native wheel at {expected}; "
            f"found {version}. Rebase and rebuild for another native ABI."
        )
    package = Path(importlib.util.find_spec("vllm").origin).parent
    native = {
        str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in package.rglob("*.so")
    }
    receipt = {}
    for name in manifest["files"]:
        file = source / name
        if not file.resolve().is_relative_to(source) or file.suffix != ".py":
            raise ValueError(name)
        ast.parse(file.read_text())
        target = package.parent / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
        receipt[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    for name, expected_hash in native.items():
        assert (
            hashlib.sha256((package / name).read_bytes()).hexdigest() == expected_hash
        )
    out = Path("/opt/raon/build-evidence")
    out.mkdir(parents=True, exist_ok=True)
    (out / "vllm-overlay.json").write_text(
        json.dumps(
            {
                "native_wheel_version": version,
                "base_commit": manifest["base_commit"],
                "python_files": receipt,
                "native_extensions_preserved": native,
                "gpu_validation": "NOT_RUN_DURING_BUILD",
            },
            indent=2,
        )
    )
    print(f"Installed {len(receipt)} Raon Python files over {version}")


if __name__ == "__main__":
    main()
