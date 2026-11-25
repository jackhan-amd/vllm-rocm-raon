# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raon routed-expert MXFP8 -> Quark MXFP4 using the pinned haanjack/quanto.

Uses quanto's file-to-file RTN path with a registered Raon template. Restores
the release's row-wise E8M0 MXFP8 before quantization, keeping other weights
excluded. No calibration or evaluation data is used by RTN conversion.
"""

import argparse
import copy
import hashlib
import importlib.metadata as metadata
import json
import time
from pathlib import Path


def expert_weight(name):
    parts = name.split(".")
    return (
        len(parts) == 8
        and parts[:2] == ["model", "layers"]
        and parts[2].isdigit()
        and parts[3:5] == ["moe", "experts"]
        and parts[5].isdigit()
        and parts[6] in ("gate_proj", "up_proj", "down_proj")
        and parts[7] == "weight"
    )


def nvfp4_reference_policy(reference, names):
    """Map the published mixed-precision layer policy onto source tensor names."""
    config = json.loads(reference.read_text())
    quant = config["quantization_config"]
    if quant.get("quant_algo") != "MIXED_PRECISION":
        raise ValueError("Expected the official mixed-precision NVFP4 reference")
    packed = {
        "in_proj_qkvgfab": [
            "q_proj",
            "k_proj",
            "v_proj",
            "g_proj",
            "f_a_proj",
            "b_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }
    policy, aliases = {}, {}
    for module, settings in quant["quantized_layers"].items():
        algo = settings["quant_algo"]
        if algo not in ("NVFP4", "W4A16_NVFP4", "MXFP8"):
            raise ValueError(f"Unsupported reference role: {module}: {algo}")
        aliases[module] = algo
        if module + ".weight" in names:
            targets = [module]
        else:
            parent, leaf = module.rsplit(".", 1)
            targets = [parent + "." + child for child in packed.get(leaf, [])]
            if not targets or any(t + ".weight" not in names for t in targets):
                raise ValueError("Cannot map reference module: " + module)
        for target in targets:
            if target in policy and policy[target] != algo:
                raise ValueError("Conflicting reference policy: " + target)
            policy[target] = algo
    return policy, {**aliases, **policy}


def restore_mxfp8(weight, scales):
    """Independent row-wise FP32 decode, exactly representable as BF16 weights."""
    import torch

    if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
        raise ValueError("Expected released 2D E4M3 weights")
    if scales.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise ValueError("Expected raw E8M0 scale bytes")
    if weight.shape[-1] % 32 or scales.shape != (
        weight.shape[0],
        weight.shape[1] // 32,
    ):
        raise ValueError("Expected row-wise 1x32 scale blocks")
    codes = scales.view(torch.uint8)
    if bool((codes == 255).any()):
        raise ValueError("Nonfinite E8M0 scale")
    restored = (
        (
            weight.float().reshape(weight.shape[0], -1, 32)
            * torch.exp2(codes.float() - 127)[..., None]
        )
        .reshape(weight.shape)
        .to(torch.bfloat16)
    )
    if not bool(torch.isfinite(restored).all()):
        raise ValueError("Nonfinite restored MXFP8 weight")
    return restored


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference-nvfp4-config", type=Path)
    a = p.parse_args()
    source = a.model.resolve()
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "raon1_mini":
        raise ValueError("This adapter supports only the released Raon model")
    quant = config.get("quantization_config", {})
    if quant.get("quant_method") != "mxfp8" or quant.get("weight_block_size") != [
        1,
        32,
    ]:
        raise ValueError("Expected the released expert-only MXFP8 checkpoint")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    names = index["weight_map"]
    experts = [n for n in names if expert_weight(n)]
    if not experts or any(n + "_scale_inv" not in names for n in experts):
        raise ValueError("Missing routed-expert weights or MXFP8 scales")
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError("Use a fresh output directory; failed attempts are retained")
    a.output.mkdir(parents=True, exist_ok=True)
    policy, policy_aliases = ({}, {})
    if a.reference_nvfp4_config:
        policy, policy_aliases = nvfp4_reference_policy(a.reference_nvfp4_config, names)
    excluded = sorted(
        n[:-7]
        for n in names
        if n.endswith(".weight")
        and (n[:-7] not in policy if policy else not expert_weight(n))
    )
    receipt = {
        "status": "QUANTIZING",
        "source_config_sha256": hashlib.sha256(
            (source / "config.json").read_bytes()
        ).hexdigest(),
        "source_index_sha256": hashlib.sha256(
            (source / "model.safetensors.index.json").read_bytes()
        ).hexdigest(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "routed_weight_tensors": len(experts),
        "excluded_modules": excluded,
        "source_precision": "Released MXFP8 E4M3 + row-wise E8M0 (1x32)",
        "target_precision": "Quark MXFP4 weights and dynamic MXFP4 activations",
        "algorithm": "quanto file-to-file RTN; no calibration data",
        "scope": (
            "Requantization of the released MXFP8 checkpoint; "
            "original BF16 training weights are unavailable in this input"
        ),
    }
    path = a.output / "raon-quantization-receipt.json"
    if policy:
        from collections import Counter

        receipt["reference_policy"] = {
            "config_sha256": hashlib.sha256(
                a.reference_nvfp4_config.read_bytes()
            ).hexdigest(),
            "source_module_roles": dict(Counter(policy.values())),
            "translation": {
                "NVFP4": "MXFP4 W4A4",
                "W4A16_NVFP4": "MXFP4 W4A16",
                "MXFP8": "MXFP8 W8A8",
            },
            "algorithm": (
                "Our Quark RTN; reference metadata does not establish "
                "its conversion algorithm"
            ),
        }

    def save():
        path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    save()
    started = time.monotonic()
    try:
        # Import quanto first: its own compatibility hooks must be installed
        # before adapting Quark's two source-format helpers for this process.
        from quanto.core.unified_quantizer import UnifiedConfig, UnifiedQuantizer
        from quark.torch import LLMTemplate
        from quark.torch.quantization import file2file_quantization as f2f

        receipt["packages"] = {
            n: metadata.version(n)
            for n in ("quanto", "amd-quark", "torch", "safetensors")
        }
        if "raon1_mini" not in LLMTemplate.list_available():
            LLMTemplate.register_template(
                LLMTemplate(
                    model_type="raon1_mini",
                    kv_layers_name=["*self_attn.kv*"],
                    q_layer_name=["*self_attn.q*"],
                    exclude_layers_name=[],
                )
            )
        from safetensors import safe_open

        config_name = (
            "get_quantization_config"
            if hasattr(f2f, "get_quantization_config")
            else "_get_quantization_config"
        )
        original_config = getattr(f2f, config_name)
        original_recover = f2f._recover_fp8_weights
        fp8_dispatch = original_fp8_real = None
        if policy:
            # Quark 0.12 has a block-MX observer and FP8 fake-quant support,
            # but its real FP8 exporter only accepts tensor/channel schemes.
            # Each block is one independent channel: use Quark's own real
            # quantizer on [blocks, 32] without changing scales or rounding.
            from quark.torch.kernel.hw_emulation import hw_emulation_interface as hw

            fp8_dispatch = hw.DTYPE_TO_REAL_QUANTIZE_FUNCS
            original_fp8_real = fp8_dispatch["fp8_e4m3"]

            def real_fp8_groups(
                inputs, scale=None, axis=None, qscheme=None, group_size=None, **kwargs
            ):
                if qscheme != "per_group":
                    return original_fp8_real(
                        inputs,
                        scale=scale,
                        axis=axis,
                        qscheme=qscheme,
                        group_size=group_size,
                        **kwargs,
                    )
                if axis not in (-1, inputs.ndim - 1) or group_size != 32:
                    raise ValueError("Only row-wise block-32 MXFP8 export is adapted")
                shape = inputs.shape
                return original_fp8_real(
                    inputs.reshape(-1, 32),
                    scale=scale.reshape(-1),
                    axis=0,
                    qscheme="per_channel",
                    **kwargs,
                ).reshape(shape)

            fp8_dispatch["fp8_e4m3"] = real_fp8_groups
            receipt["fp8_export_adapter"] = (
                "Quark block32 -> per-channel real quant; same observer scales"
            )

        def source_format(hf_config):
            value = original_config(hf_config)
            if value is not None and value.get("quant_method") == "mxfp8":
                return {**value, "quant_method": "fp8"}
            return value

        def recover_raon(safetensor_path, device, **_kwargs):
            recovered = {}
            with safe_open(safetensor_path, framework="pt", device=str(device)) as f:
                keys = set(f.keys())
                for name in sorted(keys):
                    if name.endswith("_scale_inv"):
                        continue
                    value = f.get_tensor(name)
                    if expert_weight(name):
                        scale_name = name + "_scale_inv"
                        if scale_name in keys:
                            scales = f.get_tensor(scale_name)
                        else:
                            with safe_open(
                                source / names[scale_name],
                                framework="pt",
                                device=str(device),
                            ) as scale_file:
                                scales = scale_file.get_tensor(scale_name)
                        value = restore_mxfp8(value, scales)
                    recovered[name] = value
            return recovered

        setattr(f2f, config_name, source_format)
        f2f._recover_fp8_weights = recover_raon
        try:

            class PolicyQuantizer(UnifiedQuantizer):
                def _create_quant_config(self, exclude_layers):
                    value = super()._create_quant_config(exclude_layers)
                    for module, role in policy.items():
                        if role == "NVFP4":
                            continue
                        layer = copy.deepcopy(value.global_quant_config)
                        if role == "W4A16_NVFP4":
                            layer.input_tensors = None
                        else:
                            from quark.torch.quantization.config.type import Dtype

                            layer.weight.dtype = Dtype.fp8_e4m3
                            layer.input_tensors.dtype = Dtype.fp8_e4m3
                        value.layer_quant_config[module] = layer
                    return value

                def _patch_output_config(self, exclude_layers):
                    super()._patch_output_config(exclude_layers)
                    if not policy:
                        return
                    output_config_path = Path(self.config.output_dir) / "config.json"
                    output_config = json.loads(output_config_path.read_text())
                    qc = output_config["quantization_config"]
                    for module, role in policy_aliases.items():
                        if role == "NVFP4":
                            continue
                        layer = copy.deepcopy(qc["global_quant_config"])
                        if role == "W4A16_NVFP4":
                            layer["input_tensors"] = None
                        else:
                            layer["weight"]["dtype"] = "fp8_e4m3"
                            layer["input_tensors"]["dtype"] = "fp8_e4m3"
                        qc["layer_quant_config"][module] = layer
                    output_config_path.write_text(
                        json.dumps(output_config, indent=2), encoding="utf-8"
                    )

            runner = PolicyQuantizer(
                UnifiedConfig(
                    model_path=str(source),
                    output_dir=str(a.output),
                    precision="mxfp4",
                    algorithm="rtn",
                    device=a.device,
                    exclude_layers=excluded,
                    trust_remote_code=False,
                    skip_evaluation=True,
                )
            )
            result = runner.run()
            if not result.success:
                raise RuntimeError(result.error_message)
        finally:
            setattr(f2f, config_name, original_config)
            f2f._recover_fp8_weights = original_recover
            if fp8_dispatch is not None:
                fp8_dispatch["fp8_e4m3"] = original_fp8_real
        receipt["status"] = "EXPORTED_REQUIRES_VALIDATION"
    except BaseException as error:
        receipt.update(status="FAILED", error=str(error))
        raise
    finally:
        receipt["seconds"] = time.monotonic() - started
        save()


if __name__ == "__main__":
    main()
