# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import torch


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def provenance(function):
    path = Path(inspect.getsourcefile(inspect.unwrap(function)))
    return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def environment():
    from vllm.platforms import current_platform

    assert torch.cuda.is_available(), "A HIP/CUDA device is required"
    if torch.version.hip:
        assert current_platform.is_rocm(), (
            "ROCm platform detection failed; preserve image PYTHONPATH"
        )
    p = torch.cuda.get_device_properties(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(8)
    return {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": p.name,
        "architecture": getattr(p, "gcnArchName", None),
        "visible_devices": os.getenv("ROCR_VISIBLE_DEVICES"),
        "job": os.getenv("SLURM_JOB_ID"),
        "python": sys.executable,
        "pythonpath": os.getenv("PYTHONPATH"),
        "platform": type(current_platform).__name__,
    }


def error(actual, expected, rms_limit=0.01, peak_limit=0.02, atol=1e-5):
    """Predeclared BF16 screen; never ignore nonfinite values or near-zero tensors."""
    a, b = actual.float(), expected.float()
    finite = bool(torch.isfinite(a).all() & torch.isfinite(b).all())
    if not finite:
        return {"pass": False, "finite": False}
    diff = a - b
    rms = float(diff.square().mean().sqrt())
    ref_rms = float(b.square().mean().sqrt())
    peak = float(diff.abs().max())
    ref_peak = float(b.abs().max())
    return {
        "pass": rms <= atol + rms_limit * ref_rms
        and peak <= atol + peak_limit * ref_peak,
        "finite": True,
        "rms_error": rms,
        "reference_rms": ref_rms,
        "relative_rms": rms / max(ref_rms, 1e-30),
        "max_abs": peak,
        "reference_peak": ref_peak,
        "limits": {
            "relative_rms": rms_limit,
            "relative_peak": peak_limit,
            "atol": atol,
        },
    }
