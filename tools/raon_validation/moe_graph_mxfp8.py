# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual Raon MXFP8 combine/shared-stream/outer-graph call sites on a fixture.

Uses synthetic checkpoint-native bytes and routing weights. No model loading
or throughput measurement. Metadata-only BF16 placeholders reside on meta.
"""

import argparse
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .common import environment, error, write_json
from .flydsl_moe import gold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32])
    args = parser.parse_args()
    os.environ.update(RAON_MOE_BACKEND="flydsl", RAON_MOE_OUTER_GRAPH="1")
    from vllm.models.raon1_mini import model
    from vllm.models.raon1_mini.mxfp8_w8a8_triton import _WQ_CACHE

    result = {
        "status": "RUNNING",
        "scope": __doc__,
        "environment": environment(),
        "cases": [],
    }
    free, total = torch.accelerator.get_memory_info()
    if free / total < 0.95:
        raise RuntimeError("Select an idle GPU")
    torch.manual_seed(20261125)
    e, h, intermediate, topk = 128, 2560, 1088, 6
    q13 = torch.randn(e, 2 * intermediate, h, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    q2 = torch.randn(e, h, intermediate, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    s13 = torch.full(
        (e, 2 * intermediate, h // 32), 122, device="cuda", dtype=torch.uint8
    )
    s2 = torch.full((e, h, intermediate // 32), 122, device="cuda", dtype=torch.uint8)

    class Router:
        top_k = topk

        def select_experts(self, hidden_states, router_logits, topk_indices_dtype):
            probs, ids = router_logits.sigmoid().topk(topk, -1)
            return probs / probs.sum(-1, keepdim=True), ids.to(topk_indices_dtype)

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(e, h, device="cuda") * 0.01)

        def forward(self, x):
            return F.linear(x.float(), self.weight), None

    class Shared(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(
                h, h, bias=False, dtype=torch.bfloat16, device="cuda"
            )
            self.calls = torch.zeros((), device="cuda", dtype=torch.int64)

        def forward(self, x):
            self.calls.add_(1)
            return self.linear(x)

    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 0
            self.gate = Gate()
            self.shared_experts = Shared()
            self.experts = torch.nn.Module()
            self.experts.router = Router()
            routed = self.experts.routed_experts = torch.nn.Module()
            routed.w13_weight = torch.nn.Parameter(
                torch.empty(
                    e, 2 * intermediate, h, device="meta", dtype=torch.bfloat16
                ),
                requires_grad=False,
            )
            routed.w2_weight = torch.nn.Parameter(
                torch.empty(e, h, intermediate, device="meta", dtype=torch.bfloat16),
                requires_grad=False,
            )
            routed.moe_config = SimpleNamespace(
                activation_situ_beta=4.0, activation_situ_linear_beta=25.0
            )
            for name, value in zip(("q13", "s13", "q2", "s2"), (q13, s13, q2, s2)):
                routed.register_buffer(name, value)
            _WQ_CACHE[id(routed.w13_weight)] = (
                q13,
                s13,
                q2,
                s2,
                routed.w13_weight,
                routed.w2_weight,
            )

    try:
        with torch.inference_mode():
            for stream in ("0", "1"):
                os.environ["RAON_MOE_SHARED_SIDE_STREAM"] = stream
                module = Fixture()
                for tokens in args.tokens:
                    x = torch.randn(tokens, h, dtype=torch.bfloat16, device="cuda")
                    eager = model._raon_fp32_combine_moe_impl_eager(module, x)
                    probs, ids = module.experts.router.select_experts(
                        x, module.gate(x)[0], torch.int32
                    )
                    rows = gold(
                        x,
                        probs,
                        ids,
                        q13,
                        s13,
                        q2,
                        s2,
                        os.environ.get("RAON_MXFP8_FLYDSL_FUSED_A2", "0") != "1",
                    )
                    ordered = rows.gather(
                        1, ids.argsort(-1)[..., None].expand(-1, -1, h)
                    )
                    total_rows = torch.zeros(tokens, h, device="cuda")
                    for slot in range(topk):
                        total_rows = total_rows + ordered[:, slot].float()
                    expected_gold = (
                        total_rows.bfloat16() + module.shared_experts.linear(x)
                    )
                    case = dict(
                        tokens=tokens,
                        stream=stream,
                        eager_vs_torch_gold=error(eager, expected_gold),
                    )
                    for repeat in range(3):
                        x.add_(0.125)
                        expected = model._raon_fp32_combine_moe_impl_eager(
                            module, x
                        ).clone()
                        counter = module.shared_experts.calls
                        before = int(counter)
                        actual = model._raon_fp32_combine_moe_impl(module, x)
                        delta = int(counter) - before
                        case[f"graph_{repeat}"] = error(actual, expected, 0, 0, 0)
                        case[f"shared_once_{repeat}"] = {
                            "pass": delta == (3 if repeat == 0 else 1),
                            "executions": delta,
                        }
                    result["cases"].append(case)
                    write_json(args.output, result)
                _WQ_CACHE.pop(id(module.experts.routed_experts.w13_weight), None)
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
    finally:
        for key in list(_WQ_CACHE):
            if _WQ_CACHE[key][0] is q13:
                _WQ_CACHE.pop(key, None)
    write_json(args.output, result)
    print(result["status"], flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
