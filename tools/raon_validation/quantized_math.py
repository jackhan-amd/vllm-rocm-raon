# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent arithmetic audit of the released MXFP8 -> MXFP4 expert export.

Uses only Torch and safetensors: no Quark, AITER or FlyDSL quantizer/decoder.
Checks every selected scale and decoded code against E2M1 half-even RTN with
the declared E8M0 even scale rule. Quantization error is reported separately
from conversion mismatches. This does not certify model-level accuracy.
"""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

from .common import write_json
from .quantize import expert_weight, nvfp4_reference_policy


def even_exponents(amax, mbits=1, emax=2):
    import torch

    # E2M1's shared exponent increments at 1.75 * 2**floor(log2(amax)).
    # frexp gives an independent arithmetic implementation of Quark's FP32
    # bit-mask rounding; it does not invoke the conversion implementation.
    fraction, exponent = torch.frexp(amax.float())
    result = exponent - (emax + 1) + (fraction >= 1 - 2.0 ** -(mbits + 2)).int()
    return torch.where(amax == 0, -127, result).clamp(-127, 127)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--limit", type=int, default=0, help="0 audits all expert weights"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-nvfp4-config", type=Path)
    args = parser.parse_args()
    import torch
    from safetensors import safe_open

    torch.set_num_threads(8)
    source_map = json.loads((args.source / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    target_map = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    config = json.loads((args.model / "config.json").read_text())
    spec = config["quantization_config"]["global_quant_config"]["weight"]
    assert all(
        spec[k] == v
        for k, v in {
            "dtype": "fp4",
            "group_size": 32,
            "scale_format": "e8m0",
            "round_method": "half_even",
            "scale_calculation_mode": "even",
        }.items()
    ), spec
    policy = (
        nvfp4_reference_policy(args.reference_nvfp4_config, source_map)[0]
        if args.reference_nvfp4_config
        else {}
    )
    names = sorted(
        n
        for n in source_map
        if n.endswith(".weight") and (n[:-7] in policy if policy else expert_weight(n))
    )
    if args.limit:
        names = names[: args.limit]
    result = {
        "status": "RUNNING",
        "scope": __doc__,
        "weights_expected": len(names),
        "weights_checked": 0,
        "elements": 0,
        "scale_mismatches": 0,
        "decoded_rtn_mismatches": 0,
        "nonfinite_tensors": 0,
        "quantization_error_is_not_conversion_mismatch": True,
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mismatches": [],
    }
    codebook = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=args.device)
    midpoints = (codebook[:-1] + codebook[1:]) * 0.5
    sums = defaultdict(
        lambda: {"sse": 0.0, "source_square": 0.0, "elements": 0, "weights": 0}
    )
    started = time.monotonic()
    write_json(args.output, result)
    with ExitStack() as stack:
        sources = {
            f: stack.enter_context(
                safe_open(args.source / f, framework="pt", device="cpu")
            )
            for f in set(source_map.values())
        }
        targets = {
            f: stack.enter_context(
                safe_open(args.model / f, framework="pt", device="cpu")
            )
            for f in set(target_map.values())
        }
        for number, name in enumerate(names, 1):
            payload = sources[source_map[name]].get_tensor(name).to(args.device)
            shape = (payload.shape[0], payload.shape[1] // 32)
            # Every E4M3 value times an E8M0 scale is exactly representable in
            # FP64; verify that the converter's BF16 restoration loses no bits.
            exact = payload.double()
            if expert_weight(name):
                scale_name = name + "_scale_inv"
                source_codes = (
                    sources[source_map[scale_name]]
                    .get_tensor(scale_name)
                    .to(args.device)
                    .view(torch.uint8)
                )
                assert not bool((source_codes == 255).any()), name
                exact = (
                    exact.reshape(*shape, 32)
                    * torch.exp2(source_codes.double() - 127)[..., None]
                ).reshape(payload.shape)
            restored = exact.bfloat16()
            if not torch.equal(exact, restored.double()):
                raise ValueError("Source restoration is not lossless BF16: " + name)
            source = restored.float().reshape(*shape, 32)
            packed = targets[target_map[name]].get_tensor(name).to(args.device)
            target_scale_name = name + "_scale"
            codes = (
                targets[target_map[target_scale_name]]
                .get_tensor(target_scale_name)
                .to(args.device)
                .view(torch.uint8)
            )
            fp8 = policy.get(name[:-7]) == "MXFP8"
            exponent = (
                even_exponents(source.abs().amax(-1), 3, 8)
                if fp8
                else even_exponents(source.abs().amax(-1))
            )
            scale = torch.exp2(exponent.float())
            scale_bad = int((codes.int() != exponent + 127).sum())
            if fp8:
                expected = (source / scale[..., None]).clamp(-448, 448).to(
                    torch.float8_e4m3fn
                ).float() * scale[..., None]
                actual = packed.float().reshape_as(source)
            else:
                magnitude = (source / scale[..., None]).abs()
                index = torch.bucketize(magnitude.contiguous(), midpoints)
                # bucketize chooses the lower code at ties; select the even code.
                tie = (index < 7) & (magnitude == midpoints[index.clamp_max(6)])
                index = index + (tie & ((index % 2) == 1)).int()
                expected = codebook[index] * source.sign() * scale[..., None]
                nibbles = torch.stack((packed & 15, packed >> 4), -1).reshape(
                    payload.shape
                )
                actual = (
                    codebook[(nibbles & 7).long()]
                    * torch.where(nibbles >= 8, -1.0, 1.0)
                ).reshape_as(source)
            actual = actual * torch.exp2(codes.float() - 127)[..., None]
            data_bad = int((actual != expected).sum())
            finite = bool(torch.isfinite(actual).all())
            result["scale_mismatches"] += scale_bad
            result["decoded_rtn_mismatches"] += data_bad
            result["nonfinite_tensors"] += int(not finite)
            if (scale_bad or data_bad or not finite) and len(result["mismatches"]) < 12:
                result["mismatches"].append(
                    {
                        "name": name,
                        "scales": scale_bad,
                        "values": data_bad,
                        "finite": finite,
                    }
                )
            key = (
                ".".join((name.split(".")[2], name.split(".")[6]))
                if expert_weight(name)
                else name.removesuffix(".weight")
            )
            bucket = sums[key]
            bucket["sse"] += float((actual - source).square().sum(dtype=torch.float64))
            bucket["source_square"] += float(source.square().sum(dtype=torch.float64))
            bucket["elements"] += source.numel()
            bucket["weights"] += 1
            result["weights_checked"] = number
            result["elements"] += source.numel()
            if number % 128 == 0 or number == len(names):
                result["seconds"] = time.monotonic() - started
                write_json(args.output, result)
                print(
                    json.dumps(
                        {
                            k: result[k]
                            for k in (
                                "weights_checked",
                                "scale_mismatches",
                                "decoded_rtn_mismatches",
                                "seconds",
                            )
                        }
                    ),
                    flush=True,
                )
    result["by_layer_projection"] = {
        k: {**v, "relative_rms": (v["sse"] / max(v["source_square"], 1e-100)) ** 0.5}
        for k, v in sums.items()
    }
    result["weight_relative_rms"] = (
        sum(v["sse"] for v in sums.values())
        / sum(v["source_square"] for v in sums.values())
    ) ** 0.5
    result["status"] = (
        "PASS_ARITHMETIC"
        if not any(
            result[k]
            for k in ("scale_mismatches", "decoded_rtn_mismatches", "nonfinite_tensors")
        )
        else "FAIL_CONVERSION"
    )
    result["seconds"] = time.monotonic() - started
    write_json(args.output, result)
    assert result["status"] == "PASS_ARITHMETIC", result["mismatches"]


if __name__ == "__main__":
    main()
