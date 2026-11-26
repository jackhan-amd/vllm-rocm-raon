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


def reference(site, s):
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
    # The test branch is identity, with the real BF16 branch-input boundary.
    aggregate = (hpre[:, :, None] * x).sum(1).to(s.dtype)
    output = residual @ x + hpost[:, :, None] * aggregate[:, None, :].float()
    return output.to(s.dtype), aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=["hip", "flydsl"], default="hip")
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 12, 24])
    parser.add_argument(
        "--rows", nargs="+", type=int, default=[1, 8, 32, 64, 256, 1024, 2048]
    )
    parser.add_argument("--scales", nargs="+", type=float, default=[0.125, 1.0, 8.0])
    args = parser.parse_args()
    os.environ["RAON_ROCM_MHC_BACKEND"] = args.backend
    result = dict(status="RUNNING", scope=__doc__, cases=[], refit=[])
    try:
        result["environment"] = environment()
        from vllm.models.raon1_mini import mhc, rocm_mhc

        result["native_source"] = provenance(rocm_mhc.project)
        result["mhc_source"] = provenance(mhc.RaonMHC.forward)
        result["backend"] = args.backend
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
                                return value

                            actual = site(s, branch)
                            gold, gold_aggregate = reference(site, s)
                            rocm_mhc._BACKEND = "legacy"
                            baseline = site(s, lambda value: value)
                            rocm_mhc._BACKEND = args.backend
                            repeat = site(s, lambda value: value)
                            one = site(s[:1], lambda value: value)
                            site(s, lambda value: value)
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                replayed = site(s, lambda value: value)
                            s.add_(0.125)
                            graph.replay()
                            replay_gold, _ = reference(site, s)
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
                                changed_input_replay=error(replayed, replay_gold),
                                exact_repeat=torch.equal(actual, repeat),
                                full_block_batch_invariant=torch.equal(actual[:1], one),
                            )
                            result["cases"].append(case)
                            write_json(args.output, result)
                    s = inputs[:8].clone()
                    site(s, lambda value: value)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        refit_output = site(s, lambda value: value)
                    before = site._rocm_mhc_phi.data_ptr()
                    packed_before = getattr(
                        site, "_rocm_mhc_packed", site._rocm_mhc_phi
                    ).data_ptr()
                    site.phi.add_(0.001)
                    site.finalize()
                    graph.replay()
                    refit_gold, _ = reference(site, s)
                    result["refit"].append(
                        dict(
                            layer=layer,
                            site=kind,
                            pointer_stable=before == site._rocm_mhc_phi.data_ptr(),
                            packed_pointer_stable=packed_before
                            == getattr(
                                site, "_rocm_mhc_packed", site._rocm_mhc_phi
                            ).data_ptr(),
                            replay_vs_gold=error(refit_output, refit_gold),
                        )
                    )
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
            c["pointer_stable"] and c["packed_pointer_stable"] for c in result["refit"]
        )
        result["status"] = "PASS" if checks and all(checks) and stable else "FAIL"
    except Exception:
        result.update(status="ERROR", traceback=traceback.format_exc())
        raise
    finally:
        write_json(args.output, result)
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
