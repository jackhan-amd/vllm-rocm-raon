# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raon routed-expert MXFP8 -> Quark MXFP4 using the pinned haanjack/quanto.

Uses quanto's file-to-file RTN path with a registered Raon template. Restores
the release's row-wise E8M0 MXFP8 before quantization, keeping other weights
excluded. No calibration or evaluation data is used by RTN conversion.
"""

import argparse
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
    excluded = sorted(
        n[:-7] for n in names if n.endswith(".weight") and not expert_weight(n)
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
            runner = UnifiedQuantizer(
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
        receipt["status"] = "EXPORTED_REQUIRES_VALIDATION"
    except BaseException as error:
        receipt.update(status="FAILED", error=str(error))
        raise
    finally:
        receipt["seconds"] = time.monotonic() - started
        save()


if __name__ == "__main__":
    main()
