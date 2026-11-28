# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record real linear call sites without extra forwards or device operations.

Import ``install(path)`` before GPUModelRunner.load_model in a diagnostic
entrypoint. This is host-side shape/implementation evidence, not kernel timing
or proof of the final device kernel. Pair it with a GPU activity trace.
"""

import functools
import json
import threading
from pathlib import Path

import torch

_INSTALLED = False


def _tensor(value):
    if not isinstance(value, torch.Tensor):
        return None
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def install(output: Path):
    """Instrument diagnostic processes only; never read device tensor values."""
    global _INSTALLED
    if _INSTALLED:
        raise RuntimeError("GEMM ownership diagnostics already installed")
    _INSTALLED = True
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    names = {}
    seen = set()
    patched = set()
    original = GPUModelRunner.load_model

    def write(row):
        key = json.dumps(row, sort_keys=True)
        with lock:
            if key in seen:
                return
            seen.add(key)
            with output.open("a", encoding="utf-8") as stream:
                stream.write(key + "\n")

    def observe_method(cls):
        if cls in patched or not callable(getattr(cls, "apply", None)):
            return
        patched.add(cls)
        apply = cls.apply

        @functools.wraps(apply)
        def observed(method, layer, x, *args, **kwargs):
            result = apply(method, layer, x, *args, **kwargs)
            impl = getattr(method, "_gemm_impl", apply)
            write(
                {
                    "kind": "linear_apply",
                    "owner": names.get(id(layer), "unmapped"),
                    "prefix": getattr(layer, "prefix", None),
                    "method": f"{cls.__module__}.{cls.__qualname__}",
                    "implementation": (
                        f"{getattr(impl, '__module__', '?')}."
                        f"{getattr(impl, '__qualname__', type(impl).__name__)}"
                    ),
                    "capturing": torch.cuda.is_current_stream_capturing(),
                    "input": _tensor(x),
                    "parameters": {
                        key: _tensor(value)
                        for key, value in layer.named_parameters(recurse=False)
                    },
                    "output": _tensor(result),
                }
            )
            return result

        cls.apply = observed

    def register(model, root):
        if not isinstance(model, torch.nn.Module):
            return
        for name, module in model.named_modules():
            names.setdefault(id(module), root + ("." + name if name else ""))
            method = getattr(module, "quant_method", None)
            if isinstance(module, (LinearBase, ParallelLMHead)) and method is not None:
                observe_method(type(method))

    @functools.wraps(original)
    def load(runner, *args, **kwargs):
        result = original(runner, *args, **kwargs)
        register(runner.model, "target")
        speculator = getattr(runner, "speculator", None)
        register(getattr(speculator, "model", None), "draft")
        for key in ("drafter", "proposer"):
            nested = getattr(speculator, key, None)
            register(getattr(nested, "model", None), "draft." + key)
        write(
            {
                "kind": "installed",
                "modules": len(names),
                "quant_methods": sorted(c.__qualname__ for c in patched),
                "speculator": type(speculator).__qualname__,
                "device_operations_added": 0,
                "extra_forwards": 0,
                "scope": "host shape and dispatcher identity; not device timing",
            }
        )
        return result

    GPUModelRunner.load_model = load
