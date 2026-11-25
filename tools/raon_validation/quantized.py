# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit a Raon Quark export: packed expert layout and unchanged other tensors."""

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

from .quantize import expert_weight, nvfp4_reference_policy


def check(source, model, reference=None):
    import torch
    from safetensors import safe_open

    source_map = json.loads((source / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    output_map = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    experts = [n for n in source_map if expert_weight(n)]
    policy = nvfp4_reference_policy(reference, source_map)[0] if reference else {}
    converted = (
        {
            n: policy[n[:-7]]
            for n in source_map
            if n.endswith(".weight") and n[:-7] in policy
        }
        if policy
        else {n: "NVFP4" for n in experts}
    )
    source_scales = {n + "_scale_inv" for n in experts}
    expected = (source_map.keys() - source_scales) | {n + "_scale" for n in converted}
    if set(output_map) != expected:
        raise ValueError("Exported tensor names do not match weights/scales contract")
    result = {
        "status": "EXPORT_VALID",
        "packed_expert_weights": 0,
        "unchanged_tensors": 0,
        "converted_nonexpert_weights": 0,
    }
    with ExitStack() as stack:
        original = {
            f: stack.enter_context(safe_open(source / f, framework="pt", device="cpu"))
            for f in set(source_map.values())
        }
        exported = {
            f: stack.enter_context(safe_open(model / f, framework="pt", device="cpu"))
            for f in set(output_map.values())
        }
        for name in sorted(source_map):
            if name in source_scales:
                continue
            src = original[source_map[name]]
            dst = exported[output_map[name]]
            if name in converted:
                shape = src.get_slice(name).get_shape()
                weight = dst.get_slice(name)
                scale_name = name + "_scale"
                scale = exported[output_map[scale_name]].get_tensor(scale_name)
                fp8 = converted[name] == "MXFP8"
                if weight.get_dtype() != (
                    "F8_E4M3" if fp8 else "U8"
                ) or weight.get_shape() != [
                    shape[0],
                    shape[1] if fp8 else shape[1] // 2,
                ]:
                    raise ValueError(f"Invalid packed MXFP4 weight: {name}")
                if list(scale.shape) != [shape[0], shape[1] // 32]:
                    raise ValueError(f"Invalid E8M0 scale shape: {name}")
                if scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
                    raise ValueError(f"Invalid E8M0 scale dtype: {name}")
                if bool((scale.view(torch.uint8) == 255).any()):
                    raise ValueError(f"Nonfinite E8M0 scale: {name}")
                result[
                    "packed_expert_weights"
                    if expert_weight(name)
                    else "converted_nonexpert_weights"
                ] += 1
            else:
                a, b = src.get_tensor(name), dst.get_tensor(name)
                if a.shape != b.shape or a.dtype != b.dtype:
                    raise ValueError(f"Excluded tensor layout changed: {name}")
                if not torch.equal(
                    a.reshape(-1).contiguous().view(torch.uint8),
                    b.reshape(-1).contiguous().view(torch.uint8),
                ):
                    raise ValueError(f"Excluded tensor bytes changed: {name}")
                result["unchanged_tensors"] += 1
    result["source_safetensors_bytes"] = sum(
        (source / f).stat().st_size for f in set(source_map.values())
    )
    result["output_safetensors_bytes"] = sum(
        (model / f).stat().st_size for f in set(output_map.values())
    )
    result["scope"] = (
        "Packed layout, all scale codes and exact excluded tensors; "
        "not native GEMM correctness or model quality"
    )
    result["source_index_sha256"] = hashlib.sha256(
        (source / "model.safetensors.index.json").read_bytes()
    ).hexdigest()
    result["output_index_sha256"] = hashlib.sha256(
        (model / "model.safetensors.index.json").read_bytes()
    ).hexdigest()
    result["tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-nvfp4-config", type=Path)
    a = p.parse_args()
    result = check(a.source, a.model, a.reference_nvfp4_config)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
