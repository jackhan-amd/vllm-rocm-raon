# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage the ROCm AOT addon for TVM FFI's forced ROCm loader before imports."""

import hashlib
import importlib.metadata
import importlib.util
import os
import shutil
from pathlib import Path


def main():
    if os.environ.get("TVM_FFI_DISABLE_TORCH_C_DLPACK", "0") != "0":
        return
    spec = importlib.util.find_spec("torch_c_dlpack_ext")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "The Raon FlyDSL profile requires its ROCm torch-c-dlpack-ext"
        )
    libraries = list(Path(spec.origin).parent.glob("libtorch_c_dlpack_addon*.so"))
    if len(libraries) != 1:
        raise RuntimeError(f"Expected one ROCm AOT DLPack library, found {libraries}")
    major, minor = importlib.metadata.version("torch").split(".")[:2]
    cache = Path(os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi")).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    source = libraries[0]
    destination = cache / f"libtorch_c_dlpack_addon_torch{major}{minor}-rocm.so"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
        not destination.exists()
        or hashlib.sha256(destination.read_bytes()).hexdigest() != digest
    ):
        temporary = destination.with_suffix(f".{os.getpid()}.tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    print(
        f"RAON_TORCH_C_DLPACK AOT_STAGED sha256={digest} cache={destination}",
        flush=True,
    )


if __name__ == "__main__":
    main()
