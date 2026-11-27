# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental single-CTA mHC project/head; direct hipcc build."""

import functools
import hashlib
import importlib.util
import json
import os
import subprocess
import sysconfig
from pathlib import Path

import torch
from filelock import FileLock


@functools.cache
def load():
    if not torch.version.hip:
        raise RuntimeError("Native mHC requires ROCm")
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        raise RuntimeError("Experimental native mHC requires gfx950")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Compile native mHC before graph capture")
    from torch.utils.cpp_extension import include_paths, library_paths

    source = Path(__file__).with_name("rocm_mhc_project_head.hip")
    identity = hashlib.sha256(
        source.read_bytes() + torch.__version__.encode()
    ).hexdigest()[:20]
    root = Path(os.environ.get("RAON_MHC_HIP_BUILD_DIR", "/tmp/raon-mhc-hip"))
    root = root / identity
    root.mkdir(parents=True, exist_ok=True)
    name = "raon_mhc_project_head_" + identity
    binary = root / (name + sysconfig.get_config_var("EXT_SUFFIX"))
    with FileLock(str(root / "build.lock")):
        if not binary.exists():
            import shutil

            hipcc = shutil.which("hipcc")
            if hipcc is None:
                raise RuntimeError("Native mHC requires hipcc on PATH")
            libs = library_paths(device_type="cuda")
            command = [
                hipcc,
                "-x",
                "hip",
                "--offload-arch=gfx950",
                "-std=c++20",
                "-O3",
                "-ffp-contract=off",
                "-shared",
                "-fPIC",
                f"-DTORCH_EXTENSION_NAME={name}",
                f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
                "-I" + sysconfig.get_path("include"),
            ]
            command += ["-I" + p for p in include_paths(device_type="cuda")]
            temporary = binary.with_suffix(".building.so")
            command += [str(source), "-o", str(temporary)]
            for p in libs:
                command += ["-L" + p, "-Wl,-rpath," + p]
            command += [
                "-ltorch_python",
                "-ltorch",
                "-ltorch_cpu",
                "-ltorch_hip",
                "-lc10",
                "-lc10_hip",
                "-lamdhip64",
            ]
            (root / "build-command.json").write_text(json.dumps(command, indent=2))
            with (root / "build.log").open("w") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            temporary.replace(binary)
    spec = importlib.util.spec_from_file_location(name, binary)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load native mHC module at {binary}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def forward(site, s):
    module = load()
    aggregate = torch.empty((s.shape[0], 2560), device=s.device, dtype=s.dtype)
    coeff = torch.empty((s.shape[0], 20), device=s.device, dtype=torch.float32)
    module.forward(
        s,
        site.phi,
        site.A_row,
        site.B_row,
        site.Q,
        site.H0,
        site.negH0,
        site.g_eye,
        site.g_gate,
        aggregate,
        coeff,
    )
    return aggregate, coeff
