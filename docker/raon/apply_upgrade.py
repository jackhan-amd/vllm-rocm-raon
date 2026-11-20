# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Version-pinned AITER/FlyDSL overlay on the exact requested vLLM image."""

import hashlib
import importlib.metadata as md
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "build-evidence"
OUT.mkdir(exist_ok=False)
PINS = json.loads((ROOT / "pins.json").read_text())
STEPS = []


def run(label, argv, cwd=None, env=None, timeout=900):
    (OUT / (label + "-argv.json")).write_text(json.dumps(argv, indent=2))
    start = time.monotonic()
    with (OUT / (label + ".log")).open("w") as stream:
        p = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    STEPS.append(
        {"stage": label, "exit_code": p.returncode, "seconds": time.monotonic() - start}
    )
    (OUT / "steps.json").write_text(json.dumps(STEPS, indent=2))
    print(label, p.returncode, flush=True)
    if p.returncode:
        raise RuntimeError(label + " failed; see " + str(OUT / (label + ".log")))


def versions():
    return {
        name: md.version(name)
        for name in ("torch", "triton", "vllm", "amd-aiter", "flydsl")
    }


before = versions()
(OUT / "versions-before.json").write_text(json.dumps(before, indent=2))
run(
    "capture-before",
    [sys.executable, str(ROOT / "capture_runtime.py"), str(OUT / "runtime-before")],
)
work = Path(tempfile.mkdtemp(prefix="raon-library-build-", dir="/tmp"))
try:
    wheels = PINS["flydsl_cp314_wheels"]
    assert len(wheels) == 1
    fly = wheels[0]
    target = work / fly["filename"]
    with (
        urllib.request.urlopen(fly["url"], timeout=120) as response,
        target.open("wb") as stream,
    ):
        shutil.copyfileobj(response, stream, 4 * 1024 * 1024)
    with target.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == fly["sha256"]
    build_deps = [name + "==" + ver for name, ver in PINS["build_packages"].items()]
    run(
        "build-tools",
        [sys.executable, "-m", "pip", "install", "--no-deps", *build_deps],
    )
    run(
        "flydsl-upgrade",
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--upgrade",
            "--report",
            str(OUT / "flydsl-pip-report.json"),
            str(target),
        ],
    )
    src = work / "aiter"
    run(
        "aiter-clone",
        [
            "git",
            "clone",
            "--depth=1",
            "--branch",
            PINS["aiter_release"],
            "https://github.com/ROCm/aiter.git",
            str(src),
        ],
    )
    observed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=src, text=True
    ).strip()
    assert observed == PINS["aiter_commit"], observed
    run(
        "aiter-submodules",
        ["git", "submodule", "update", "--init", "--recursive", "--depth=1"],
        src,
    )
    run(
        "aiter-submodule-provenance", ["git", "submodule", "status", "--recursive"], src
    )
    # Preserve the image's ROCm Torch/Triton. Native modules JIT against CPython
    # 3.14 on first use. Keep CK/HIP/ASM available; this is not Triton-only mode.
    env = dict(
        os.environ,
        AITER_USE_SYSTEM_TRITON="1",
        BUILD_TARGET="rocm",
        PREBUILD_KERNELS="0",
        AITER_PY3_WHEEL="1",
        GPU_ARCHS="gfx950",
    )
    wheel_dir = work / "wheels"
    run(
        "aiter-wheel",
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            str(src),
        ],
        env=env,
        timeout=1500,
    )
    candidates = list(wheel_dir.glob("amd_aiter-*.whl"))
    assert len(candidates) == 1, candidates
    wheel = candidates[0]
    with wheel.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    (OUT / "built-wheel.json").write_text(
        json.dumps(
            {
                "filename": wheel.name,
                "sha256": digest,
                "source_commit": observed,
                "bytes": wheel.stat().st_size,
            },
            indent=2,
        )
    )
    run(
        "aiter-upgrade",
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--upgrade",
            "--report",
            str(OUT / "aiter-pip-report.json"),
            str(wheel),
        ],
    )
    after = versions()
    for name in ("torch", "triton", "vllm"):
        assert after[name] == before[name], (name, before, after)
    assert after["flydsl"] == PINS["flydsl_version"]
    assert after["amd-aiter"].split("+")[0] == PINS["aiter_release"].removeprefix("v")
    (OUT / "versions-after.json").write_text(json.dumps(after, indent=2))
    run(
        "capture-after",
        [sys.executable, str(ROOT / "capture_runtime.py"), str(OUT / "runtime-after")],
    )
    run("pip-freeze", [sys.executable, "-m", "pip", "freeze", "--all"])
    check = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    )
    (OUT / "pip-check.log").write_text(check.stdout + check.stderr)
    (OUT / "result.json").write_text(
        json.dumps(
            {
                "status": "LIBRARIES_UPGRADED",
                "before": before,
                "after": after,
                "source_commit": observed,
                "gpu_validation": "NOT_RUN_IN_IMAGE_BUILD",
                "pip_check_exit_code": check.returncode,
            },
            indent=2,
        )
    )
finally:
    # Only this tempfile-created build tree is removed, never image/user caches.
    assert work.parent == Path("/tmp") and work.name.startswith("raon-library-build-")
    shutil.rmtree(work)
