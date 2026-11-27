# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental pure AR/SD native RDA; compile directly without hipify."""

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
def load(kind="ar"):
    if not torch.version.hip:
        raise RuntimeError("Native RDA requires ROCm")
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        raise RuntimeError("Experimental native RDA requires gfx950")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Compile native RDA before graph capture")
    from torch.utils.cpp_extension import include_paths, library_paths

    source = Path(__file__).with_name(f"rocm_hip_rda_{kind}.hip")
    identity = hashlib.sha256(
        source.read_bytes()
        + Path(__file__).with_name("rocm_hip_rda_ar.hip").read_bytes()
        + torch.__version__.encode()
    ).hexdigest()[:20]
    root = Path(os.environ.get("RAON_HIP_RDA_BUILD_DIR", "/tmp/raon-hip-rda"))
    root = root / identity
    root.mkdir(parents=True, exist_ok=True)
    name = "raon_hip_rda_" + identity
    binary = root / (name + sysconfig.get_config_var("EXT_SUFFIX"))
    with FileLock(str(root / "build.lock")):
        if not binary.exists():
            import shutil

            hipcc = shutil.which("hipcc")
            if hipcc is None:
                raise RuntimeError("Native RDA requires hipcc on PATH")
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
        raise RuntimeError(f"Cannot load native RDA module at {binary}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def forward(layer, inputs, output, conv, state, weight, indices, metadata=None):
    from .mxfp8_flydsl import _identity

    if layer.local_num_heads != 40:
        raise ValueError("Experimental HIP RDA requires 40 heads")
    spec = metadata is not None
    key = (_identity(layer.conv1d.weight), _identity(layer.o_norm.weight), spec)
    cache_name = "_raon_hip_rda_prepared_sd" if spec else "_raon_hip_rda_prepared_ar"
    prepared = getattr(layer, cache_name, None)
    if prepared is None or prepared[0] != key:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare native RDA weights before capture")
        packed = weight.view(3, 5120, 4).transpose(1, 2).float().contiguous()
        norm = layer.o_norm.weight.float().contiguous()
        prepared = (key, packed, norm, load("sd" if spec else "ar"))
        setattr(layer, cache_name, prepared)
    raw, gate, out_gate, beta, ratio = inputs
    if spec:
        prepared[3].forward_sd(
            raw,
            prepared[1],
            conv,
            gate,
            beta,
            ratio,
            layer.A_log,
            layer.dt_bias,
            indices.contiguous(),
            metadata.num_accepted_tokens[: indices.shape[0]],
            metadata.spec_query_start_loc[: indices.shape[0] + 1],
            state,
            out_gate,
            prepared[2],
            output,
            float(layer.gate_lower_bound),
            float(layer.o_norm.eps),
        )
        return
    prepared[3].forward(
        raw,
        prepared[1],
        conv,
        gate,
        beta,
        ratio,
        layer.A_log,
        layer.dt_bias,
        indices,
        state,
        out_gate,
        prepared[2],
        output.view(1, -1, 40, 128),
        float(layer.gate_lower_bound),
        float(layer.o_norm.eps),
    )
