# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic MXFP4 native layout + actual SharedExperts + Raon outer graph.

This checks the adapter with real AITER GPU math and the real shared-expert
owner. It does not load a checkpoint or initialize the full FusedMoERunner.
"""

import argparse
import importlib
import os
import runpy
from types import SimpleNamespace

import torch

from .common import environment, error, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ.update(
        RAON_MOE_BACKEND="flydsl",
        RAON_MOE_OUTER_GRAPH="1",
        VLLM_ROCM_USE_AITER_MOE_SITUV2="a4w4",
    )
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import fused_moe
    from aiter.jit.core import AITER_CSRC_DIR
    from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params
    from aiter.utility import fp4_utils as fp4

    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
        SharedExpertsOrder,
    )
    from vllm.model_executor.layers.quantization.quark.quark_moe import (
        QuarkOCP_MX_MoEMethod,
    )
    from vllm.models.raon1_mini import rocm_mxfp4_graph, rocm_shared_stream

    result = {
        "status": "RUNNING",
        "scope": __doc__,
        "environment": environment(),
        "cases": [],
    }
    free, total = torch.accelerator.get_memory_info()
    if free / total < 0.95:
        raise RuntimeError("Select an idle GPU")
    from pathlib import Path

    namespace = runpy.run_path(
        str(Path(AITER_CSRC_DIR) / "ck_gemm_moe_2stages_codegen/gemm_moe_tune.py")
    )
    port, base = namespace["Mxfp4FlydslTuner"], namespace["FmoeTuner"]
    dispatch_module = importlib.import_module("aiter.fused_moe")
    original_selector = dispatch_module.get_2stage_cfgs
    observed = {}

    def observe_selector(*values, **keywords):
        metadata = original_selector(*values, **keywords)
        observed["metadata"] = metadata
        return metadata

    dispatch_module.get_2stage_cfgs = observe_selector

    def selected_boundary():
        stage = observed["metadata"].stage1
        keywords = {}
        while hasattr(stage, "func"):
            keywords = {**getattr(stage, "keywords", {}), **keywords}
            stage = stage.func
        name = keywords.get("kernelName", "")
        parsed = get_flydsl_kernel_params(name)
        if parsed is None:
            raise RuntimeError(f"Need an explicit Torch boundary for stage1 {name}")
        output = keywords.get("out_dtype") or parsed["out_dtype"]
        if output not in ("bf16", "fp4"):
            raise RuntimeError(f"Unsupported stage1 reference boundary: {output}")
        return ("bf16_then_fp4" if output == "bf16" else "fp32_then_fp4"), name

    cfg = SimpleNamespace(
        hidden_dim_unpadded=2560,
        intermediate_size_per_partition=1152,
        intermediate_size_per_partition_unpadded=1152,
        intermediate_pad=0,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        moe_parallel_config=SimpleNamespace(
            enable_eplb=False,
            all2all_backend="naive",
            use_fi_nvl_two_sided_kernels=False,
        ),
    )

    class SharedLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(
                2560, 2560, bias=False, device="cuda", dtype=torch.bfloat16
            )
            self.calls = torch.zeros((), device="cuda", dtype=torch.int64)

        def forward(self, x):
            self.calls.add_(1)
            return self.linear(x)

    class Fixture(torch.nn.Module):
        def __init__(self, data):
            super().__init__()
            self._raon_local_graph_world = True
            self.experts = torch.nn.Module()
            routed = self.experts.routed_experts = torch.nn.Module()
            routed.register_buffer("w13_weight", data["w1_a16"])
            routed.register_buffer("w2_weight", data["w2_a16"])
            routed.moe_config = cfg
            # Only the selector metadata is needed; no mocked GPU operation.
            method = object.__new__(QuarkOCP_MX_MoEMethod)
            method.mxfp4_backend = Mxfp4MoeBackend.AITER_MXFP4_MXFP4
            method.ocp_mx_scheme = "w_mxfp4_a_mxfp4"
            routed.quant_method = method
            self.experts.router = SimpleNamespace(top_k=6)
            self.experts.shared_experts = SharedExperts(
                SharedLayer(), cfg, False, lambda: False
            )
            self.data = data

        def _native_forward(self, x):
            rocm_shared_stream.configure_native(self, x)
            rocm_mxfp4_graph.verify_native_backend(self, x)
            shared = self.experts.shared_experts
            queued = shared.maybe_forward_async(x)
            data = self.data
            routed = fused_moe(
                x,
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
            if queued:
                shared.wait()
            else:
                shared(x, SharedExpertsOrder.NO_OVERLAP)
            return routed + shared.output

    try:
        with torch.inference_mode():
            for tokens in (1, 8, 32):
                data = port._prepare_case(tokens, 2560, 1152, 128, 6, dtypes.bf16)
                data["w1_a16"].is_shuffled = data["w2_a16"].is_shuffled = True
                references = {}
                for reference_dtype, label in (
                    (dtypes.fp32, "fp32_then_fp4"),
                    (dtypes.bf16, "bf16_then_fp4"),
                ):
                    ref1 = base.run_torch_moe_stage1(
                        data["a1_qt"],
                        data["w1_qt"],
                        data["w2_qt"],
                        data["topk_weights"],
                        data["topk_ids"],
                        data["a1_scale"],
                        data["w1_scale"],
                        dtype=reference_dtype,
                        activation=ActivationType.Situv2,
                        quant_type=QuantType.per_1x32,
                        topk=6,
                    )
                    groups = ref1.float().reshape(-1, 32)
                    scales = fp4.e8m0_to_f32(
                        fp4.fp4_f32_to_e8m0_scale(groups.abs().amax(-1))
                    )
                    packed = fp4.f32_to_mxfp4(groups / scales[:, None])
                    inter = (
                        (fp4.mxfp4_to_f32(packed) * scales[:, None])
                        .reshape(ref1.shape)
                        .bfloat16()
                    )
                    references[label] = base.run_torch_moe_stage2(
                        inter,
                        data["w1_qt"],
                        data["w2_qt"],
                        data["topk_weights"],
                        data["topk_ids"],
                        None,
                        data["w2_scale"],
                        dtype=dtypes.bf16,
                        quant_type=QuantType.per_1x32,
                    )
                original_inputs = {
                    name: data[name].clone()
                    for name in ("input", "topk_ids", "topk_weights")
                }
                for stream in ("0", "1"):
                    for name, value in original_inputs.items():
                        data[name].copy_(value)
                    os.environ["RAON_MOE_SHARED_SIDE_STREAM"] = stream
                    module = Fixture(data)
                    x = data["input"]
                    eager = module._native_forward(x)
                    boundary, kernel_name = selected_boundary()
                    gold = references[boundary]
                    shared_gold = module.experts.shared_experts._layer.linear(x)
                    case = dict(
                        tokens=tokens,
                        stream=stream,
                        reference_boundary=boundary,
                        stage1_kernel=kernel_name,
                        eager_vs_gold=error(eager, gold + shared_gold),
                    )
                    for repeat in range(3):
                        x.add_(0.015625)
                        data["topk_ids"].copy_((data["topk_ids"] + 1) % 128)
                        expected = module._native_forward(x).clone()
                        repeated = module._native_forward(x).clone()
                        case[f"eager_repeat_exact_{repeat}"] = error(
                            repeated, expected, 0, 0, 0
                        )
                        counter = module.experts.shared_experts._layer.calls
                        before = int(counter)
                        captured = rocm_mxfp4_graph.try_forward(module, x)
                        assert captured is not None
                        delta = int(counter) - before
                        case[f"graph_{repeat}"] = error(captured, expected, 0, 0, 0)
                        case[f"graph_numeric_{repeat}"] = error(captured, expected)
                        case[f"shared_once_{repeat}"] = {
                            "pass": delta == (3 if repeat == 0 else 1),
                            "executions": delta,
                        }
                    result["cases"].append(case)
                    write_json(args.output, result)
        result["numeric_status"] = (
            "PASS"
            if all(
                value["pass"]
                for case in result["cases"]
                for key, value in case.items()
                if isinstance(value, dict)
                and (
                    key == "eager_vs_gold"
                    or key.startswith(("graph_numeric_", "shared_once_"))
                )
            )
            else "FAIL"
        )
        result["status"] = (
            "PASS"
            if all(
                v["pass"]
                for c in result["cases"]
                for v in c.values()
                if isinstance(v, dict) and "pass" in v
            )
            else "FAIL"
        )
    except Exception:
        import traceback

        result.update(status="ERROR", traceback=traceback.format_exc())
    write_json(args.output, result)
    print(result["status"], flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
