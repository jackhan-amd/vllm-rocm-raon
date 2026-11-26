# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install an immutable candidate stack without replacing ROCm Torch/Triton."""

import hashlib
import importlib.metadata as md
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    pins = json.loads((root / "pins-stack-20261126.json").read_text())
    out = Path(os.environ.get("RAON_STACK_EVIDENCE", str(root / "stack-evidence")))
    out.mkdir(parents=True, exist_ok=False)
    packages = ("torch", "triton", "vllm", "amd-aiter", "flydsl", "apache-tvm-ffi")
    before = {n: md.version(n) for n in packages}
    (out / "versions-before.json").write_text(json.dumps(before, indent=2))

    def run(name, argv, cwd=None, timeout=1800):
        print("STACK_STEP", name, flush=True)
        (out / (name + "-argv.json")).write_text(json.dumps(argv))
        with (out / (name + ".log")).open("w") as stream:
            subprocess.run(
                argv,
                cwd=cwd,
                check=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )

    wheel = pins["flydsl_wheel"]
    path = out / wheel["filename"]
    urllib.request.urlretrieve(wheel["url"], path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == wheel["sha256"]
    run(
        "flydsl",
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            "--reinstall",
            str(path),
        ],
    )
    src = Path("/opt/raon/aiter-source-" + pins["aiter_commit"][:8])
    assert not src.exists(), src
    src.mkdir()
    run("git-init", ["git", "init", str(src)])
    run("git-remote", ["git", "remote", "add", "origin", pins["aiter_repository"]], src)
    run("git-fetch", ["git", "fetch", "--depth=1", "origin", pins["aiter_commit"]], src)
    run("git-checkout", ["git", "checkout", "--detach", "FETCH_HEAD"], src)
    observed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=src, text=True
    ).strip()
    assert observed == pins["aiter_commit"]
    run(
        "submodules",
        ["git", "submodule", "update", "--init", "--recursive", "--depth=1"],
        src,
    )
    run("submodule-provenance", ["git", "submodule", "status", "--recursive"], src)
    os.environ.update(
        AITER_USE_SYSTEM_TRITON="1",
        BUILD_TARGET="rocm",
        PREBUILD_KERNELS="0",
        AITER_PY3_WHEEL="1",
        GPU_ARCHS="gfx950",
    )
    run(
        "aiter-install",
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            "--no-build-isolation",
            "--reinstall",
            str(src),
        ],
    )
    # A fresh interpreter is required after package replacement.
    code = (
        "import json,importlib.metadata as m;print(json.dumps({n:m.version(n) for n in "
        + repr(packages)
        + "}))"
    )
    after = json.loads(subprocess.check_output([sys.executable, "-c", code], text=True))
    for n in ("torch", "triton", "vllm", "apache-tvm-ffi"):
        assert before[n] == after[n], (n, before, after)
    assert after["flydsl"] == pins["flydsl_version"]
    receipt = {
        "status": "INSTALLED_NOT_GPU_VALIDATED",
        "aiter_commit": observed,
        "source": str(src),
        "versions": after,
        "pins": pins,
    }
    (out / "installed.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
