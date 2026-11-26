# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight Raon mHC block vs independent Torch math, including refit/replay."""

import argparse
import json
import os
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from tools.raon_validation.common import environment, error, provenance, write_json


def reference(site, s, branch=lambda value: value, *, intermediates=False):
    rows = s.shape[0]
    x = s.float()
    # Keep the checkpoint BF16 parameter and GEMM-output rounding boundaries.
    scores = F.linear(x.reshape(rows, -1), site.phi.float()).to(s.dtype).float()
    inverse = torch.rsqrt(x.square().mean((1, 2)) + 1e-5)
    logits = scores * inverse[:, None]
    gates = site.A_row * logits + site.B_row
    hpre = torch.sigmoid(gates[:, :4])
    hpost = 2 * torch.sigmoid(gates[:, 4:8])
    z = gates[:, 8:].reshape(rows, 3, 3)
    chart = site.Q @ z @ site.Q.t()
    rho = (chart / (-site.H0)).amax((-2, -1)).clamp_min(0)
    manifold = site.H0 + chart / (1 + rho)[:, None, None]
    residual = site.g_eye + site.g_gate * manifold
    aggregate = (hpre[:, :, None] * x).sum(1).to(s.dtype)
    output = residual @ x + hpost[:, :, None] * branch(aggregate)[:, None, :].float()
    if intermediates:
        return aggregate, torch.cat((hpost, residual.flatten(1)), dim=1)
    return output.to(s.dtype), aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=["hip", "flydsl"], default="hip")
    parser.add_argument("--finisher", choices=["torch", "flydsl"], default="torch")
    parser.add_argument("--gate", choices=["legacy", "flydsl"], default="legacy")
    parser.add_argument("--branch", choices=["identity", "affine"], default="identity")
    parser.add_argument("--batch-invariant", choices=["0", "1"], default="1")
    parser.add_argument("--profile", help="Optional GPU trace path, after gold passes")
    parser.add_argument(
        "--timing-replays",
        type=int,
        default=0,
        help="Optional graph-replay block timing, after gold passes",
    )
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 12, 24])
    parser.add_argument(
        "--rows", nargs="+", type=int, default=[1, 8, 32, 64, 256, 1024, 2048]
    )
    parser.add_argument("--scales", nargs="+", type=float, default=[0.125, 1.0, 8.0])
    args = parser.parse_args()
    if args.finisher == "flydsl" and args.backend != "hip":
        parser.error("The FlyDSL finisher requires the HIP mHC backend")
    if args.gate == "flydsl" and (args.backend, args.finisher) != ("hip", "flydsl"):
        parser.error("The native gate requires HIP projection and FlyDSL finisher")
    os.environ["RAON_ROCM_MHC_BACKEND"] = args.backend
    os.environ["RAON_ROCM_MHC_FINISHER"] = args.finisher
    os.environ["RAON_ROCM_MHC_GATE"] = args.gate
    os.environ["RAON_BATCH_INVARIANT"] = args.batch_invariant

    def branch_fn(value):
        return value if args.branch == "identity" else value * 0.75 + 0.125

    result = dict(status="RUNNING", scope=__doc__, cases=[], refit=[])
    try:
        result["environment"] = environment()
        from vllm.models.raon1_mini import mhc, rocm_mhc, rocm_mhc_gate

        result["native_source"] = provenance(rocm_mhc.project)
        result["mhc_source"] = provenance(mhc.RaonMHC.forward)
        result["backend"] = args.backend
        result["finisher"] = args.finisher
        result["gate"] = args.gate
        result["branch"] = args.branch
        result["batch_invariant"] = args.batch_invariant
        result["gate_source"] = provenance(rocm_mhc_gate.forward)
        if args.finisher == "flydsl":
            from vllm.models.raon1_mini import rocm_mhc_finish

            result["finisher_source"] = provenance(rocm_mhc_finish.reduce)
        mapping = json.loads(
            next(args.model.glob("*.safetensors.index.json")).read_text()
        )["weight_map"]
        suffixes = [
            "phi_pre",
            "phi_post",
            "res.phi_z",
            "b_pre",
            "b_post",
            "res.b_z",
            "alpha_pre",
            "alpha_post",
            "res.alpha_z",
            "res.eta",
        ]
        with torch.inference_mode():
            for layer in args.layers:
                for kind in ("self_attn", "mlp"):
                    old_dtype = torch.get_default_dtype()
                    try:
                        torch.set_default_dtype(torch.bfloat16)
                        site = mhc.RaonMHC(4, 2560, 2 * layer + int(kind == "mlp"))
                    finally:
                        torch.set_default_dtype(old_dtype)
                    site = site.to("cuda")
                    prefix = f"model.layers.{layer}.{kind}_raon_mhc."
                    for suffix in suffixes:
                        key = prefix + suffix
                        with safe_open(
                            str(args.model / mapping[key]), framework="pt", device="cpu"
                        ) as f:
                            site.load_weight(suffix, f.get_tensor(key).to("cuda"))
                    site.finalize()
                    torch.manual_seed(1401 + 2 * layer + int(kind == "mlp"))
                    inputs = torch.randn(
                        (max(args.rows), 4, 2560), device="cuda", dtype=torch.bfloat16
                    )
                    for scale in args.scales:
                        for rows in args.rows:
                            s = inputs[:rows].clone().mul_(scale)
                            captured_branch = []

                            def branch(value, captured_branch=captured_branch):
                                captured_branch.append(value)
                                return branch_fn(value)

                            actual = site(s, branch)
                            gold, gold_aggregate = reference(site, s, branch_fn)
                            # Same native HIP projection/finisher, old head/chart/mix.
                            rocm_mhc_gate._BACKEND = "legacy"
                            production = site(s, branch_fn)
                            try:
                                rocm_mhc._BACKEND = "legacy"
                                baseline = site(s, branch_fn)
                            finally:
                                rocm_mhc._BACKEND = args.backend
                                rocm_mhc_gate._BACKEND = args.gate
                            intermediate = {}
                            if args.gate == "flydsl":
                                agg, coeff, _ = rocm_mhc_gate.prepare_branch(site, s)
                                ga, gc = reference(site, s, intermediates=True)
                                intermediate = dict(
                                    coefficient_vs_gold=error(coeff, gc),
                                    head_aggregate_vs_gold=error(agg, ga),
                                )
                            repeat = site(s, branch_fn)
                            one = site(s[:1], branch_fn)
                            site(s, branch_fn)
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                replayed = site(s, branch_fn)
                            s.add_(0.125)
                            graph.replay()
                            replay_gold, _ = reference(site, s, branch_fn)
                            case = dict(
                                layer=layer,
                                site=kind,
                                rows=rows,
                                scale=scale,
                                native_vs_gold=error(actual, gold),
                                aggregate_vs_gold=error(
                                    captured_branch[0], gold_aggregate
                                ),
                                baseline_vs_gold=error(baseline, gold),
                                native_vs_baseline=error(actual, baseline),
                                native_vs_production=error(actual, production),
                                changed_input_replay=error(replayed, replay_gold),
                                exact_repeat=torch.equal(actual, repeat),
                                full_block_batch_invariant=torch.equal(actual[:1], one),
                                **intermediate,
                            )
                            result["cases"].append(case)
                            write_json(args.output, result)
                    s = inputs[: min(8, max(args.rows))].clone()
                    site(s, branch_fn)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        refit_output = site(s, branch_fn)
                    before = site._rocm_mhc_phi.data_ptr()
                    packed_before = getattr(
                        site, "_rocm_mhc_packed", site._rocm_mhc_phi
                    ).data_ptr()
                    gate_pointers = {
                        name: getattr(site, name).data_ptr()
                        for name in (
                            "A_row",
                            "B_row",
                            "g_eye",
                            "g_gate",
                            "negH0",
                        )
                    }
                    refit_names = (
                        "phi",
                        "alpha_pre",
                        "alpha_post",
                        "alpha_z",
                        "b_pre",
                        "b_post",
                        "b_z",
                        "eta",
                    )
                    original_params = {
                        name: getattr(site, name).clone() for name in refit_names
                    }
                    site.phi.add_(0.001)
                    for name in refit_names[1:]:
                        getattr(site, name).add_(0.01)
                    site.finalize()
                    graph.replay()
                    refit_gold, _ = reference(site, s, branch_fn)
                    result["refit"].append(
                        dict(
                            layer=layer,
                            site=kind,
                            pointer_stable=before == site._rocm_mhc_phi.data_ptr(),
                            packed_pointer_stable=packed_before
                            == getattr(
                                site, "_rocm_mhc_packed", site._rocm_mhc_phi
                            ).data_ptr(),
                            gate_pointer_stable=all(
                                getattr(site, name).data_ptr() == ptr
                                for name, ptr in gate_pointers.items()
                            ),
                            replay_vs_gold=error(refit_output, refit_gold),
                        )
                    )
                    for name, original in original_params.items():
                        getattr(site, name).copy_(original)
                    site.finalize()
                    print(
                        f"layer={layer} site={kind} cases={len(result['cases'])}",
                        flush=True,
                    )
        checks = [
            v["pass"]
            for c in result["cases"] + result["refit"]
            for v in c.values()
            if isinstance(v, dict) and "pass" in v
        ]
        stable = all(c["exact_repeat"] for c in result["cases"]) and all(
            c["pointer_stable"]
            and c["packed_pointer_stable"]
            and c["gate_pointer_stable"]
            for c in result["refit"]
        )
        result["status"] = "PASS" if checks and all(checks) and stable else "FAIL"
        if result["status"] == "PASS" and args.timing_replays:
            if args.gate != "flydsl" or args.timing_replays < 1:
                raise ValueError("Timing requires the FlyDSL gate and positive replays")
            result["timing_scope"] = (
                "One captured mHC block, last selected layer's MLP site; "
                "warmup/compilation excluded, GPU event time, not serving TPS"
            )
            result["timing"] = []
            with torch.inference_mode():
                for rows in args.rows:
                    value = inputs[:rows].clone()
                    record = dict(rows=rows, replays=args.timing_replays)
                    try:
                        for backend in ("legacy", "flydsl"):
                            rocm_mhc_gate._BACKEND = backend
                            for _ in range(3):
                                site(value, branch_fn)
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                site(value, branch_fn)
                            for _ in range(10):
                                graph.replay()
                            torch.accelerator.synchronize()
                            start, end = (
                                torch.cuda.Event(enable_timing=True),
                                torch.cuda.Event(enable_timing=True),
                            )
                            start.record()
                            for _ in range(args.timing_replays):
                                graph.replay()
                            end.record()
                            end.synchronize()
                            record[backend + "_graph_us"] = (
                                start.elapsed_time(end) * 1000 / args.timing_replays
                            )
                    finally:
                        rocm_mhc_gate._BACKEND = args.gate
                    record["speedup"] = (
                        record["legacy_graph_us"] / record["flydsl_graph_us"]
                    )
                    result["timing"].append(record)
        if result["status"] == "PASS" and args.profile:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as p:
                site(s, branch_fn)
                torch.accelerator.synchronize()
            p.export_chrome_trace(args.profile)
    except Exception:
        result.update(status="ERROR", traceback=traceback.format_exc())
        raise
    finally:
        write_json(args.output, result)
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
