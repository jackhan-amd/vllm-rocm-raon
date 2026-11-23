# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the offline SD benchmark with full warmup and compiler/cache-miss guards.

Use isolated AITER_JIT_DIR, TRITON_CACHE_DIR, TORCH_EXTENSIONS_DIR and
FLYDSL_RUNTIME_CACHE_DIR. Extra arguments are forwarded to the benchmark tool.
The first two repetitions are priming by default; three subsequent repetitions
must complete without new compiler/cache activity. EOS remains honored.
"""

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path


def cache_snapshot(roots):
    """Observe executable/cache artifacts, excluding logs and lock files."""
    result = {}
    extensions = {".so", ".hsaco", ".co", ".o", ".bin", ".pkl", ".cubin", ".ptx"}
    for root in roots:
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file() and path.suffix in extensions:
                    result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def cache_changes(before, after):
    """Include created, modified and removed artifacts in the audit."""
    return sorted(
        name
        for name in before.keys() | after.keys()
        if before.get(name) != after.get(name)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-dir", type=Path, required=True)
    p.add_argument("--precision", choices=["mxfp8", "mxfp4"], required=True)
    p.add_argument("--priming-repetitions", type=int, default=2)
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--warmup-tokens", type=int, default=512)
    a, remaining = p.parse_known_args()
    if a.priming_repetitions < 0 or a.repetitions < a.priming_repetitions + 2:
        p.error("At least two official repetitions after priming are required")
    cache_vars = (
        "TRITON_CACHE_DIR",
        "AITER_JIT_DIR",
        "FLYDSL_RUNTIME_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
    )
    if any(not os.getenv(name) for name in cache_vars):
        p.error("Configure all four isolated cache directories before starting Python")
    roots = [Path(os.environ[name]).resolve() for name in cache_vars]
    out = a.audit_dir
    out.mkdir(parents=True, exist_ok=True)

    import aiter.jit.core as ac
    from triton import knobs

    import vllm

    from . import benchmark as bm

    events, attempts = [], []
    original_measure, original_llm = bm.measure, vllm.LLM
    audit = {
        "status": "RUNNING",
        "arguments": vars(a) | {"audit_dir": str(out)},
        "events": events,
        "blocked_attempts": attempts,
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": (
            "Forbid Triton cache misses, AITER builds and FlyDSL compilation "
            "during official timings; reject changed code-cache artifacts. "
            "Not a universal proof covering uninstrumented compiler APIs."
        ),
    }

    def save():
        (out / "compile-exclusion.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )

    def forbid(*_args, **kwargs):
        attempts.append({"key": str(kwargs.get("key", ""))[:160]})
        raise RuntimeError("Compilation/cache miss blocked during official timing")

    def measured(llm, prompts, cap, repetition, warmup, concurrency=None):
        official = not warmup and repetition >= a.priming_repetitions
        before = cache_snapshot(roots)
        saved_hook, saved_build = knobs.runtime.jit_cache_hook, ac.build_module
        saved_run = os.environ.get("FLYDSL_RUNTIME_RUN_ONLY")
        if official:
            knobs.runtime.jit_cache_hook = forbid
            ac.build_module = forbid
            os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = "1"
        try:
            result = original_measure(
                llm,
                prompts,
                a.warmup_tokens if warmup else cap,
                repetition,
                warmup,
                concurrency,
            )
        finally:
            knobs.runtime.jit_cache_hook = saved_hook
            ac.build_module = saved_build
            if saved_run is None:
                os.environ.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
            else:
                os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = saved_run
        after = cache_snapshot(roots)
        changed = cache_changes(before, after)
        events.append(
            {
                "repetition": repetition,
                "warmup": warmup,
                "official": official,
                "cache_artifacts_before": len(before),
                "cache_artifacts_after": len(after),
                "created_modified_or_removed_artifacts": changed,
                "compile_attempts": len(attempts),
                "output_tps": result["output_tokens_per_second"],
                "output_tokens": result["output_tokens"],
            }
        )
        save()
        if official and (changed or attempts):
            raise RuntimeError("Compile/cache activity invalidated official timing")
        result["official_compile_free"] = official
        result["compile_cache_changes"] = changed
        return result

    def factory(**kwargs):
        if a.precision == "mxfp4":
            kwargs["kernel_config"] = {"moe_backend": "aiter"}
        (out / "provider-config.json").write_text(json.dumps(kwargs, indent=2))
        return original_llm(**kwargs)

    saved_argv = sys.argv
    bm.measure, vllm.LLM = measured, factory
    sys.argv = [saved_argv[0], *remaining, "--repetitions", str(a.repetitions)]
    try:
        bm.main()
        audit["status"] = "PASS"
    except BaseException:
        audit.update(status="FAILED", traceback=traceback.format_exc())
        raise
    finally:
        bm.measure, vllm.LLM = original_measure, original_llm
        sys.argv = saved_argv
        save()


if __name__ == "__main__":
    main()
