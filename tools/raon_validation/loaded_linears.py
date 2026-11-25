# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit real loaded dense weights and GEMMs against checkpoint-derived Torch gold."""

import json
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import error


def check(model, directory):
    from aiter.ops.shuffle import shuffle_weight

    from vllm.models.raon1_mini.model import Raon1MiniForCausalLM

    directory = Path(directory)
    index = json.loads((directory / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
    result = []
    with ExitStack() as stack, torch.inference_mode():
        from safetensors import safe_open

        files = {
            f: stack.enter_context(
                safe_open(directory / f, framework="pt", device="cpu")
            )
            for f in set(index.values())
        }

        def tensor(name):
            return files[index[name]].get_tensor(name).to("cuda")

        for name, layer in model.named_modules():
            kernel = getattr(getattr(layer, "scheme", None), "ocp_mx_linear", None)
            if type(kernel).__name__ not in (
                "FlydslMxfp4A16LinearKernel",
                "FlydslMxfp8LinearKernel",
            ):
                continue
            prefix = layer.prefix.replace(".block_sparse_moe.", ".moe.")
            fp8 = type(kernel).__name__ == "FlydslMxfp8LinearKernel"
            if prefix + ".weight" in index:
                keys = [prefix]
            else:
                parent, leaf = prefix.rsplit(".", 1)
                keys = [
                    parent + "." + c
                    for c in Raon1MiniForCausalLM.packed_modules_mapping[leaf]
                ]
            payload = torch.cat([tensor(k + ".weight") for k in keys])
            scale = torch.cat(
                [tensor(k + ".weight_scale").view(torch.uint8) for k in keys]
            )
            # RDA's final alignment partition has no checkpoint tensor.
            n = layer.weight.shape[0]
            padding = n - payload.shape[0]
            assert padding >= 0
            if padding:
                payload = F.pad(payload, (0, 0, 0, padding))
                scale = F.pad(scale, (0, 0, 0, padding))
            expected_payload = (
                shuffle_weight(payload, layout=(16, 16)) if fp8 else payload
            )
            payload_equal = torch.equal(
                layer.weight.view(torch.uint8), expected_payload.view(torch.uint8)
            )
            scale_equal = torch.equal(layer.weight_scale.view(torch.uint8), scale)
            if fp8:
                weight = payload.float()
            else:
                codes = torch.stack((payload & 15, payload >> 4), -1).reshape(n, -1)
                weight = table[(codes & 7).long()] * torch.where(codes >= 8, -1.0, 1.0)
            weight = (
                weight.reshape(n, -1, 32) * torch.exp2(scale.float() - 127)[..., None]
            ).reshape(n, -1)
            metrics = []
            for m in (1, 7):
                x = torch.randn(
                    (m, weight.shape[1]), device="cuda", dtype=torch.bfloat16
                )
                a = x.float()
                if fp8:
                    b = a.reshape(m, -1, 32)
                    s = torch.exp2(
                        torch.ceil(
                            torch.log2((b.abs().amax(-1) / 448).clamp_min(2.0**-126))
                        )
                    )
                    a = (
                        (b / s[..., None]).to(torch.float8_e4m3fn).float()
                        * s[..., None]
                    ).reshape(a.shape)
                gold = F.linear(a, weight).bfloat16()
                actual = layer.scheme.apply_weights(layer, x)
                metrics.append(error(actual, gold, 0.01, 0.02))
            result.append(
                {
                    "layer": name,
                    "checkpoint_prefix": prefix,
                    "payload_equal": payload_equal,
                    "scale_equal": scale_equal,
                    "torch_gold": metrics,
                }
            )
            if (
                not payload_equal
                or not scale_equal
                or not all(v["pass"] for v in metrics)
            ):
                break
    return result
