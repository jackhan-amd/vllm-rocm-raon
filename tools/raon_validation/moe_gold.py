# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare existing Raon FP4 MoE and a port using matched Torch intermediates.

Uses synthetic layer inputs from the pinned installed AITER tuner. Reports both
BF16-intermediate and FP4-intermediate references; no threshold is changed and
no tuned configuration is installed. This is not an all-input or quality proof.
"""

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 16384])
    args = parser.parse_args()
    config = json.loads((args.model / "config.json").read_text())
    config = config.get("text_config", config)
    if (
        config.get("activation_situ_beta"),
        config.get("activation_situ_linear_beta"),
    ) != (4.0, 25.0):
        parser.error("This pinned diagnostic represents SiTUv2 beta=4, linear_beta=25")
    if min(args.tokens) < 1:
        parser.error("Token counts must be positive")

    import aiter
    import torch
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import fused_moe
    from aiter.jit.core import AITER_CSRC_DIR
    from aiter.utility import fp4_utils as fp4

    from .common import environment

    observed = environment()
    free, total = torch.cuda.mem_get_info()
    if free / total <= 0.99 or not observed["architecture"].startswith("gfx950"):
        raise RuntimeError("Select an idle gfx950 GPU for this diagnostic")
    tuner = Path(AITER_CSRC_DIR) / "ck_gemm_moe_2stages_codegen/gemm_moe_tune.py"
    namespace = runpy.run_path(str(tuner), run_name="vendor_reference_diagnostic")
    port, base = namespace["Mxfp4FlydslTuner"], namespace["FmoeTuner"]
    torch.manual_seed(20261123)
    result = {
        "status": "RUNNING",
        "scope": __doc__,
        "environment": observed,
        "seed": 20261123,
        "cases": [],
        "legacy_operator_support": aiter.is_mxfp4_moe_shape_supported(
            128, 2560, 1152, 6
        ),
        "vendor_sha256": hashlib.sha256(tuner.read_bytes()).hexdigest(),
        "fp4_torch_helper_sha256": hashlib.sha256(
            Path(fp4.__file__).read_bytes()
        ).hexdigest(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def error(actual, reference):
        x, y = actual.double().flatten(), reference.double().flatten()
        finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        if not finite:
            return {
                "finite": False,
                "cosine_diff": None,
                "normalized_rmse": None,
                "max_abs": None,
            }
        delta = x - y
        return {
            "cosine_diff": (
                1
                - 2
                * (x * y).sum()
                / (x.square().sum() + y.square().sum()).clamp_min(1e-30)
            ).item(),
            "normalized_rmse": (
                delta.square().mean().sqrt() / y.square().mean().sqrt().clamp_min(1e-30)
            ).item(),
            "max_abs": delta.abs().max().item(),
            "finite": finite,
        }

    def quantize_dequantize(value):
        groups = value.float().reshape(-1, 32)
        scales = fp4.e8m0_to_f32(fp4.fp4_f32_to_e8m0_scale(groups.abs().amax(dim=-1)))
        packed = fp4.f32_to_mxfp4(groups / scales[:, None])
        return (
            (fp4.mxfp4_to_f32(packed) * scales[:, None]).reshape(value.shape).bfloat16()
        )

    try:
        with torch.inference_mode():
            for tokens in args.tokens:
                data = port._prepare_case(tokens, 2560, 1152, 128, 6, dtypes.bf16)
                data["w1_a16"].is_shuffled = data["w2_a16"].is_shuffled = True
                references = {
                    "bf16_intermediate": port._torch_ref(
                        data, 6, dtypes.bf16, ActivationType.Situv2
                    )
                }
                for dtype, label in [
                    (dtypes.bf16, "bf16_then_fp4"),
                    (dtypes.fp32, "fp32_then_fp4"),
                ]:
                    stage1 = base.run_torch_moe_stage1(
                        data["a1_qt"],
                        data["w1_qt"],
                        data["w2_qt"],
                        data["topk_weights"],
                        data["topk_ids"],
                        data["a1_scale"],
                        data["w1_scale"],
                        dtype=dtype,
                        activation=ActivationType.Situv2,
                        quant_type=QuantType.per_1x32,
                        topk=6,
                    )
                    references[label] = base.run_torch_moe_stage2(
                        quantize_dequantize(stage1),
                        data["w1_qt"],
                        data["w2_qt"],
                        data["topk_weights"],
                        data["topk_ids"],
                        None,
                        data["w2_scale"],
                        dtype=dtypes.bf16,
                        quant_type=QuantType.per_1x32,
                    )
                baseline = fused_moe(
                    data["input"],
                    data["w1_a16"],
                    data["w2_a16"],
                    data["topk_weights"],
                    data["topk_ids"],
                    activation=ActivationType.Situv2,
                    quant_type=QuantType.per_1x32,
                    w1_scale=data["w1s_a16"],
                    w2_scale=data["w2s_a16"],
                    dtype=dtypes.bf16,
                    gate_mode="separated",
                    quant_dtype_a=dtypes.fp4x2,
                    quant_dtype_a2=dtypes.fp4x2,
                    beta=4.0,
                    linear_beta=25.0,
                )
                candidate = port._port_e2e(
                    data,
                    "flydsl_mxmoe_g1_a4w4_128x256x256_situv2_xcd4",
                    "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_reduce_sbm128",
                    6,
                    128,
                    2560,
                    dtypes.bf16,
                )
                result["cases"].append(
                    {
                        "tokens": tokens,
                        "baseline_vs_torch": {
                            k: error(baseline, v) for k, v in references.items()
                        },
                        "port_vs_torch": {
                            k: error(candidate, v) for k, v in references.items()
                        },
                        "port_vs_existing_fp4": error(candidate, baseline),
                    }
                )
                save()
        finite = all(
            metric["finite"]
            for case in result["cases"]
            for metric in [
                *case["baseline_vs_torch"].values(),
                *case["port_vs_torch"].values(),
                case["port_vs_existing_fp4"],
            ]
        )
        result["status"] = "MEASURED" if finite else "NONFINITE"
    except BaseException:
        import traceback

        result.update(status="ERROR", traceback=traceback.format_exc())
        raise
    finally:
        save()
    print(result["status"], str(args.output), flush=True)


if __name__ == "__main__":
    main()
