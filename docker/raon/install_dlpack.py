# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the pinned TVM FFI Torch DLPack addon for this image's ROCm ABI."""

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

SOURCE_REF = "05e3a7c5568b5e21c92f50aa7b916fe25e674b61"
SOURCE_SHA256 = "986481a38c9c45dffbb30fb9502304e6c0383399291ffa809eef1f4b5dafe10a"


def main():
    import torch

    if not torch.version.hip:
        raise RuntimeError("Build this addon against the image's ROCm PyTorch")
    evidence = Path("/opt/raon/build-evidence")
    evidence.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TVM_FFI_DISABLE_TORCH_C_DLPACK="1")
    with tempfile.TemporaryDirectory(prefix="raon-dlpack-") as directory:
        work = Path(directory)
        archive = work / "tvm-ffi.tar.gz"
        url = f"https://codeload.github.com/apache/tvm-ffi/tar.gz/{SOURCE_REF}"
        with urllib.request.urlopen(url, timeout=120) as response:
            archive.write_bytes(response.read())
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual != SOURCE_SHA256:
            raise RuntimeError("Pinned TVM FFI archive checksum mismatch")
        with tarfile.open(archive) as source:
            source.extractall(work, filter="data")
        checkout = work / ("tvm-ffi-" + SOURCE_REF)
        package = checkout / "addons/torch_c_dlpack_ext"
        builder = checkout / "python/tvm_ffi/utils/_build_optional_torch_c_dlpack.py"
        subprocess.run(
            [
                sys.executable,
                str(builder),
                "--build-with-rocm",
                "--output-dir",
                str(package / "torch_c_dlpack_ext"),
            ],
            env=env,
            check=True,
            timeout=900,
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-deps",
                "--no-build-isolation",
                str(package),
            ],
            env=env,
            check=True,
            timeout=300,
        )
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        verify_env = dict(os.environ, TVM_FFI_DISABLE_TORCH_C_DLPACK="0")
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_c_dlpack_ext, tvm_ffi; "
                "assert hasattr(torch.Tensor, '__dlpack_c_exchange_api__'); "
                "x=torch.arange(16,device='cuda',dtype=torch.float32); "
                "y=torch.from_dlpack(tvm_ffi.from_dlpack(x)); "
                "assert x.data_ptr()==y.data_ptr() and torch.equal(x,y)",
            ],
            env=verify_env,
            check=True,
            timeout=120,
        )
    (evidence / "torch-c-dlpack.json").write_text(
        json.dumps(
            {
                "source_ref": SOURCE_REF,
                "source_sha256": actual,
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "gpu_roundtrip": "PASS" if gpu_available else "REQUIRES_GPU_ALLOCATION",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
