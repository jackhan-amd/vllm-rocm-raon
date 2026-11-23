# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect warmed Raon prefill and short-generation traces outside performance runs."""

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=["mxfp4", "mxfp8"], required=True)
    parser.add_argument("--python-only", action="store_true")
    args, remaining = parser.parse_known_args()
    args.audit_dir.mkdir(parents=True, exist_ok=True)

    import aiter.jit.core as ac
    import torch
    from triton import knobs

    import vllm

    from . import benchmark as bm

    real_llm, real_measure = vllm.LLM, bm.measure
    audit = {
        "status": "RUNNING",
        "captures": [],
        "compile_attempts": [],
        "python_only": args.python_only,
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }

    def save():
        (args.audit_dir / "profile-audit.json").write_text(json.dumps(audit, indent=2))

    def forbid(*_args, **_kwargs):
        audit["compile_attempts"].append("Compilation/cache miss during profiling")
        raise RuntimeError(audit["compile_attempts"][-1])

    def factory(**kwargs):
        free, total = torch.cuda.mem_get_info()
        if free / total <= 0.99:
            raise RuntimeError("Selected profiling GPU must be idle")
        if args.precision == "mxfp4":
            kwargs["kernel_config"] = {"moe_backend": "aiter"}
        (args.audit_dir / "provider-config.json").write_text(
            json.dumps(kwargs, indent=2)
        )
        return real_llm(**kwargs)

    def measure(llm, prompts, cap, repetition, warmup, concurrency=None):
        if warmup:
            warm = []
            for rep in [-3, -2, -1]:
                warm.append(real_measure(llm, prompts, 512, rep, True, concurrency))
            (args.audit_dir / "warmup.json").write_text(json.dumps(warm))
            return warm[-1]

        cap = 1 if repetition == 0 else 64
        step = llm.llm_engine.step
        step_number = [0]

        def recorded_step():
            with torch.profiler.record_function(f"engine_step/{step_number[0]}"):
                step_number[0] += 1
                return step()

        llm.llm_engine.step = recorded_step
        old_hook, old_build = knobs.runtime.jit_cache_hook, ac.build_module
        old_run = os.environ.get("FLYDSL_RUNTIME_RUN_ONLY")
        knobs.runtime.jit_cache_hook, ac.build_module = forbid, forbid
        os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = "1"
        try:
            if args.python_only:
                import cProfile

                prof = cProfile.Profile()
                point = prof.runcall(
                    real_measure, llm, prompts, cap, repetition, False, concurrency
                )
            else:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                    profile_memory=False,
                    with_stack=False,
                ) as prof:
                    point = real_measure(
                        llm, prompts, cap, repetition, False, concurrency
                    )
        finally:
            llm.llm_engine.step = step
            knobs.runtime.jit_cache_hook, ac.build_module = old_hook, old_build
            if old_run is None:
                os.environ.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
            else:
                os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = old_run
        if args.python_only:
            import io
            import pstats

            name = f"python-c{concurrency or len(prompts)}-cap{cap}.pstats"
            prof.dump_stats(str(args.audit_dir / name))
            for sort in ["cumulative", "tottime"]:
                output = io.StringIO()
                pstats.Stats(prof, stream=output).sort_stats(sort).print_stats(80)
                (args.audit_dir / f"{name}-{sort}.txt").write_text(output.getvalue())
        else:
            name = f"trace-c{concurrency or len(prompts)}-cap{cap}.json"
            prof.export_chrome_trace(str(args.audit_dir / name))
        if audit["compile_attempts"]:
            raise RuntimeError("Compilation was attempted during profiling")
        point["scope"] = (
            "Instrumented trace capture; throughput is not performance evidence"
        )
        audit["captures"].append({"trace": name, "cap": cap, "steps": step_number[0]})
        save()
        return point

    saved_argv = sys.argv
    bm.measure, vllm.LLM = measure, factory
    sys.argv = [
        saved_argv[0],
        *remaining,
        "--output",
        str(args.output),
        "--repetitions",
        "2",
    ]
    try:
        bm.main()
        record = json.loads(args.output.read_text())
        record.update(
            status="PROFILE_CAPTURED",
            scope="Instrumented traces only; throughput is not performance evidence",
            performance_acceptance="NOT_APPLICABLE",
        )
        args.output.write_text(json.dumps(record, indent=2))
        audit["status"] = "CAPTURED"
    except BaseException:
        audit.update(status="FAILED", traceback=traceback.format_exc())
        raise
    finally:
        bm.measure, vllm.LLM = real_measure, real_llm
        sys.argv = saved_argv
        save()


if __name__ == "__main__":
    main()
