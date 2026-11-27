# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import hashlib
import time
import traceback
from pathlib import Path

import torch

from .common import environment, write_json


def main():
    p = argparse.ArgumentParser(description="Raon ROCm correctness diagnostics")
    p.add_argument(
        "command",
        choices=[
            "core",
            "rollback",
            "stress",
            "sampler",
            "sampler-gold",
            "model",
            "mla",
            "teacher",
            "replay",
            "draft-replay",
            "draft-native",
            "flydsl-moe",
            "moe-route-prepare",
            "aiter-rda",
            "effort",
        ],
    )
    p.add_argument("--output", required=True)
    p.add_argument("--dump")
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--draft-case", type=int, choices=range(8), default=0)
    p.add_argument("--draft-provider", choices=["auto", "opus", "ck"], default="auto")
    p.add_argument("--torch-gold", action="store_true")
    p.add_argument("--heads", type=int, default=40)
    p.add_argument("--moe-full-shape", action="store_true")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=1509)
    p.add_argument("--rounds", type=int, default=1024)
    p.add_argument("--draft-counts", nargs="+", type=int, default=[1, 3, 7])
    p.add_argument("--decode-batch-size", type=int, default=3)
    p.add_argument("--conv-weight-fp32", action="store_true")
    p.add_argument(
        "--decode-backend",
        choices=[
            "aiter_triton",
            "flydsl",
            "hip_rda_ar",
            "hip_rda",
            "hip_rda_fused_sd",
            "hip_rda_vsplit_sd",
            "hip_rda_auto_sd",
        ],
        default="aiter_triton",
        help="Candidate backend for the aiter-rda operation/graph diagnostic",
    )
    p.add_argument("--decode-ar-only", action="store_true")
    p.add_argument("--decode-ar-padding", action="store_true")
    p.add_argument("--decode-ar-history", type=int, default=3)
    p.add_argument("--decode-mixed", action="store_true")
    p.add_argument("--decode-mixed-prefill", type=int, default=0)
    p.add_argument("--decode-spec-ragged", action="store_true")
    p.add_argument("--decode-spec-null", action="store_true")
    p.add_argument("--decode-timing-replays", type=int, default=0)
    p.add_argument("--decode-kernel-witness", action="store_true")
    p.add_argument("--samples", type=int, default=65536)
    p.add_argument("--scratch-poison", action="store_true")
    p.add_argument(
        "--kv-padding", choices=["exact", "zero", "finite", "nan"], default="exact"
    )
    p.add_argument("--mla-heads", type=int, default=48)
    p.add_argument(
        "--mla-backend",
        choices=["ROCM_AITER_MLA", "TRITON_MLA"],
        default="ROCM_AITER_MLA",
    )
    p.add_argument(
        "--common-rda",
        action="store_true",
        help="Diagnostic: same recurrent implementation and launch shape for AR/SD",
    )
    p.add_argument("--model")
    p.add_argument("--draft-model")
    p.add_argument("--draft-tokens", type=int, default=7)
    p.add_argument("--prompts-json")
    p.add_argument("--reference-json")
    p.add_argument("--force-token", type=int)
    p.add_argument(
        "--kv-cache-dtype", choices=["fp8_e4m3", "bfloat16"], default="fp8_e4m3"
    )
    p.add_argument("--accepted-drafts", type=int)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--light",
        action="store_true",
        help="Check decoder/logits/sampler without cloning RDA cache rows",
    )
    p.add_argument("--generation-seconds", type=int, default=900)
    p.add_argument("--moe-timing-replays", type=int, default=0)
    p.add_argument("--moe-token-counts", nargs="+", type=int, default=[1, 8, 32])
    p.add_argument("--moe-varied-scales", action="store_true")
    p.add_argument("--moe-kernel-witness", action="store_true")
    p.add_argument("--max-tokens", type=int, default=262144)
    args = p.parse_args()
    result = {
        "status": "RUNNING",
        "command": args.command,
        "arguments": vars(args),
        "cases": [],
        "scope": "Finite diagnostic; not full model quality acceptance",
        "tool_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    started = time.monotonic()
    try:
        result["environment"] = environment()
        torch.manual_seed(args.seed)
        # A corrupt tensor must fail the checker; prevents a vacuous PASS.
        from .common import error

        assert not error(torch.ones(4, device="cuda"), torch.zeros(4, device="cuda"))[
            "pass"
        ]
        if args.command in ["core", "rollback", "stress"]:
            from .rda import run
        elif args.command == "sampler":
            from .sampler import run
        elif args.command == "sampler-gold":
            from .sampler_gold import run
        elif args.command == "mla":
            from .mla import run
        elif args.command == "replay":
            from .replay import run
        elif args.command == "draft-replay":
            from .draft_replay import run
        elif args.command == "draft-native":
            from .draft_native import run
        elif args.command == "flydsl-moe":
            from .flydsl_moe import run
        elif args.command == "moe-route-prepare":
            from .route_prepare import run
        elif args.command == "aiter-rda":
            from .aiter_rda import run
        elif args.command == "effort":
            from .effort import run
        else:
            from .model import run
        with torch.inference_mode():
            run(args, result)
        checks = [
            v["pass"]
            for c in result["cases"]
            for v in c.values()
            if isinstance(v, dict) and "pass" in v
        ]
        assert checks, "No checks executed"
        result["status"] = "PASS" if all(checks) else "FAIL"
        if result.get("incomplete") and all(checks):
            result["status"] = "INCOMPLETE"
    except Exception:
        result.update(status="ERROR", traceback=traceback.format_exc())
    result["seconds"] = time.monotonic() - started
    write_json(args.output, result)
    print(result["status"], args.command, args.output, flush=True)
    if "traceback" in result:
        print(result["traceback"], flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
