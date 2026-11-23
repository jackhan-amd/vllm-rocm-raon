# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, uninstrumented offline-engine SD timing with accepted-token evidence."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from .common import environment, write_json


def metrics(llm):
    snapshot = {}
    for m in llm.get_metrics():
        if m.name.startswith("vllm:spec_decode"):
            if hasattr(m, "value"):
                snapshot[m.name] = m.value
            elif hasattr(m, "values"):
                snapshot[m.name] = m.values
    return snapshot


def delta(after, before):
    return {
        k: (
            [v[i] - before.get(k, [0] * len(v))[i] for i in range(len(v))]
            if isinstance(v, list)
            else v - before.get(k, 0)
        )
        for k, v in after.items()
    }


def measure(llm, prompts, cap, repetition, warmup, concurrency=None):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    before = metrics(llm)
    torch.accelerator.synchronize()
    start = time.perf_counter()
    records = {}
    limit = concurrency or len(prompts)
    pending = iter(enumerate(prompts))
    active = 0

    def admit():
        nonlocal active
        while active < limit:
            item = next(pending, None)
            if item is None:
                break
            i, row = item
            rid = f"{'warm' if warmup else 'measure'}-{limit}-{repetition}-{i}"
            records[rid] = {
                "id": row["id"],
                "token_ids": [],
                "events": [],
                "enqueued_seconds": time.perf_counter() - start,
                "finished": False,
            }
            llm.llm_engine.add_request(
                rid,
                {"prompt_token_ids": row["prompt_token_ids"]},
                SamplingParams(
                    temperature=1.0,
                    top_p=0.95,
                    top_k=20,
                    presence_penalty=1.5,
                    max_tokens=cap,
                    min_tokens=0,
                    ignore_eos=False,
                    stop_token_ids=[259202, 259200],
                    seed=42 + int(row["id"]),
                    output_kind=RequestOutputKind.DELTA,
                ),
            )
            active += 1

    admit()
    while llm.llm_engine.has_unfinished_requests():
        assert time.perf_counter() - start < 600, "Bounded timing exceeded ten minutes"
        for output in llm.llm_engine.step():
            now = time.perf_counter() - start
            record = records[output.request_id]
            if output.outputs:
                choice = output.outputs[0]
                if choice.token_ids:
                    record["token_ids"].extend(choice.token_ids)
                    record["events"].append(
                        {"seconds": now, "tokens": len(choice.token_ids)}
                    )
                    assert not (
                        len(record["token_ids"]) >= 256
                        and all(t == 0 for t in record["token_ids"][-256:])
                    ), "Repeated token zero"
                record["finish_reason"] = choice.finish_reason
            if output.finished and not record["finished"]:
                active -= 1
            record["finished"] = output.finished
        admit()
    torch.accelerator.synchronize()
    seconds = time.perf_counter() - start
    tokenizer = llm.get_tokenizer()
    for record in records.values():
        n = len(record["token_ids"])
        assert record["finished"] and n > 0
        first, last = record["events"][0]["seconds"], record["events"][-1]["seconds"]
        record["ttft_seconds"] = first - record["enqueued_seconds"]
        record["tpot_seconds"] = (last - first) / (n - 1) if n > 1 else None
        record["text"] = tokenizer.decode(record["token_ids"])
    spec = delta(metrics(llm), before)
    drafts = spec.get("vllm:spec_decode_num_drafts", 0)
    proposed = spec.get("vllm:spec_decode_num_draft_tokens", 0)
    accepted = spec.get("vllm:spec_decode_num_accepted_tokens", 0)
    return {
        "concurrency": limit,
        "requests": len(prompts),
        "repetition": repetition,
        "warmup": warmup,
        "wall_seconds": seconds,
        "output_tokens": sum(len(r["token_ids"]) for r in records.values()),
        "output_tokens_per_second": sum(len(r["token_ids"]) for r in records.values())
        / seconds,
        "spec_metrics": spec,
        "accepted_fraction": accepted / proposed if proposed else None,
        "accepted_per_draft": accepted / drafts if drafts else None,
        "records": records,
    }


def main():
    from vllm import LLM
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--draft-model", required=True)
    p.add_argument("--prompts-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--draft-tokens", type=int, required=True)
    p.add_argument(
        "--draft-method", choices=["greedy", "probabilistic"], default="probabilistic"
    )
    p.add_argument("--verify-method", choices=["standard", "block"], default="standard")
    p.add_argument("--concurrencies", type=int, nargs="+", default=[1, 16])
    p.add_argument("--repetitions", type=int, default=2)
    p.add_argument("--output-tokens", type=int, default=512)
    p.add_argument(
        "--requests",
        type=int,
        help=(
            "Use this same request set at every concurrency, with closed-loop admission"
        ),
    )
    a = p.parse_args()
    assert AiterMLAHelper.get_actual_mla_num_heads(40) == 64, (
        "Enable the tested H40->64 decode workaround"
    )
    result = {
        "status": "RUNNING",
        "scope": (
            "Offline-engine timings; bounded output cap with EOS honored; "
            "not HTTP latency or model-quality evaluation"
        ),
        "arguments": vars(a),
        "environment": environment(),
        "warmup": [],
        "measurements": [],
    }
    rows = json.loads(Path(a.prompts_json).read_text())
    if a.requests is not None:
        assert max(a.concurrencies) <= a.requests <= len(rows)
    result["runtime_overrides"] = {
        k: os.getenv(k)
        for k in ["ROCR_VISIBLE_DEVICES", "VLLM_ROCM_AITER_MLA_PAD_H40_TO_64"]
    }
    assert len(rows) >= max(a.concurrencies) and a.repetitions >= 2
    result["prompt_sha256"] = {
        str(r["id"]): hashlib.sha256(
            json.dumps(r["prompt_token_ids"], separators=(",", ":")).encode()
        ).hexdigest()
        for r in rows
    }
    config = dict(
        model=a.model,
        tokenizer=a.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        data_parallel_size=1,
        max_model_len=262144,
        max_num_batched_tokens=16384,
        max_num_seqs=max(a.concurrencies),
        gpu_memory_utilization=0.80,
        enforce_eager=True,
        enable_prefix_caching=False,
        kv_cache_dtype="fp8_e4m3",
        mamba_cache_mode="none",
        seed=42,
        disable_log_stats=False,
        attention_config={
            "backend_per_kind": {"mla_attention": "ROCM_AITER_MLA"},
            "mla_prefill_backend": "ROCM_AITER_FA",
        },
        additional_config={"kda_prefill_backend": "fused"},
        speculative_config={
            "method": "dflash",
            "model": a.draft_model,
            "num_speculative_tokens": a.draft_tokens,
            "draft_sample_method": a.draft_method,
            "rejection_sample_method": a.verify_method,
        },
    )
    result["requested_config"] = config
    llm = LLM(**config)
    result["observed_config"] = repr(llm.llm_engine.vllm_config)
    # Warm every requested batch shape before timing it.
    for c in a.concurrencies:
        result["warmup"].append(measure(llm, rows[:c], 32, -1, True))
    write_json(a.output, result)
    for rep in range(a.repetitions):
        for c in a.concurrencies:
            result["measurements"].append(
                measure(
                    llm,
                    rows[: a.requests] if a.requests else rows[:c],
                    a.output_tokens,
                    rep,
                    False,
                    c,
                )
            )
            write_json(a.output, result)
    result["status"] = "MEASURED"
    result["performance_acceptance"] = (
        "NOT_DECIDED: no user-specified gain/latency thresholds"
    )
    write_json(a.output, result)


if __name__ == "__main__":
    main()
