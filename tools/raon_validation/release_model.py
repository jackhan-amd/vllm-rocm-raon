# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded release checkpoint smoke with graph, prefix cache and mixed efforts."""

import argparse
import json
from itertools import groupby
from types import SimpleNamespace

from .common import write_json


def install_mla_gold(result, output):
    """Check the first seven real native decode calls against dense FP32 Torch."""
    import inspect

    import aiter.mla as mla
    import torch

    from .common import error

    original = mla.mla_decode_fwd
    signature = inspect.signature(original)
    calls = [0]
    result["mla_torch_gold"] = []

    def checked(*args, **kwargs):
        out = original(*args, **kwargs)
        if torch.cuda.is_current_stream_capturing() or calls[0] >= 7:
            return out
        calls[0] += 1
        b = signature.bind(*args, **kwargs).arguments
        q, cache, actual = b["q"], b["kv_buffer"], b["o"]
        qo, kp, ki = b["qo_indptr"], b["kv_indptr"], b["kv_indices"]
        for seq in range(qo.numel() - 1):
            qa, qb = int(qo[seq]), int(qo[seq + 1])
            ka, kb = int(kp[seq]), int(kp[seq + 1])
            if qb <= qa:
                continue
            kv = cache.reshape(-1, cache.shape[-1])[ki[ka:kb].long()].float()
            kv = kv * (b.get("kv_scale") if b.get("kv_scale") is not None else 1.0)
            query = q[qa:qb, :40].float()
            query = query * (b.get("q_scale") if b.get("q_scale") is not None else 1.0)
            scores = torch.einsum("qhd,kd->qhk", query, kv) * b.get("sm_scale", 1.0)
            if b.get("causal", True):
                mask = torch.arange(kb - ka, device=q.device)[None, :] > (
                    kb
                    - ka
                    - (qb - qa)
                    + torch.arange(qb - qa, device=q.device)[:, None]
                )
                scores.masked_fill_(mask[:, None, :], -torch.inf)
            gold = torch.einsum("qhk,kd->qhd", scores.softmax(-1), kv[:, :512])
            metric = error(actual[qa:qb, :40], gold, 0.02, 0.05)
            result["mla_torch_gold"].append(
                {"call": calls[0], "seq": seq, "tokens": qb - qa, **metric}
            )
            if not metric["pass"]:
                torch.save(
                    {
                        "q": query.cpu(),
                        "kv": kv.cpu(),
                        "gold": gold.cpu(),
                        "actual": actual[qa:qb, :40].cpu(),
                    },
                    output + ".mla-failure.pt",
                )
                result.update(
                    status="FAIL_MLA_TORCH_GOLD", failure_stage="native_mla_decode"
                )
                write_json(output, result)
                raise RuntimeError("Native MLA differs from independent Torch gold")
        return out

    mla.mla_decode_fwd = checked


def install_finite_trace(result, output):
    """Add explicit graph breaks for a diagnostic first-nonfinite search.

    This changes graph segmentation and synchronizes at each boundary. It is
    unsuitable for throughput measurements and can mask a timing-sensitive bug.
    """
    import torch

    from vllm.compilation.breakable_cudagraph import eager_break_during_capture
    from vllm.forward_context import get_forward_context
    from vllm.models.raon1_mini.model import RaonDecoderLayer

    trace = {"active": False, "checks": 0}

    @eager_break_during_capture
    def check(tensor, stage):
        if not trace["active"]:
            return
        metadata = get_forward_context().attn_metadata
        n = next(
            m.num_actual_tokens
            for m in metadata.values()
            if hasattr(m, "num_actual_tokens")
        )
        value = tensor[:n]
        trace["checks"] += 1
        if not bool(torch.isfinite(value).all()):
            result.update(
                status="FAIL_NONFINITE_BOUNDARY",
                failure_stage=stage,
                finite_boundary_checks=trace["checks"],
                failure_tokens=n,
                failure_metadata=[
                    {
                        "layer": key,
                        **{
                            attr: getattr(m, attr)
                            for attr in (
                                "num_prefills",
                                "num_decodes",
                                "num_decode_tokens",
                            )
                            if hasattr(m, attr)
                        },
                    }
                    for key, m in metadata.items()
                ],
            )
            write_json(output, result)
            torch.save(value.detach().cpu(), output + ".boundary.pt")
            raise RuntimeError("First nonfinite boundary: " + stage)

    original = RaonDecoderLayer.__init__

    def init(layer, config, vllm_config, prefix, aux_stream):
        original(layer, config, vllm_config, prefix, aux_stream)
        for attr in ("self_attn", "mlp", "self_attn_raon_mhc", "mlp_raon_mhc"):
            stage = prefix + "." + attr

            def hook(module, inputs, out, stage=stage):
                check(out, stage)

            getattr(layer, attr).register_forward_hook(hook)

    RaonDecoderLayer.__init__ = init
    return trace


def probe_moe_rows(model, config):
    """Compare identical rows through real loaded native MoE runners.

    This stateless diagnostic bypasses only the local MoE graph wrapper. It
    does not change the engine's graph configuration or touch attention caches.
    The comparison is a batching invariant, not a Torch mathematical gold.
    """
    import torch

    from vllm.forward_context import set_forward_context
    from vllm.models.raon1_mini.mlp import RaonMoE

    checks = []
    modules = [(n, m) for n, m in model.named_modules() if isinstance(m, RaonMoE)]
    for name, module in (modules[0], modules[len(modules) // 2], modules[-1]):
        generator = torch.Generator(device="cuda").manual_seed(42)
        row = torch.randn(
            (1, 2560), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        with torch.inference_mode(), set_forward_context(None, config, num_tokens=1):
            singleton = module._native_forward(row).float().clone()
        for count in (7, 8, 56, 161):
            x = row.repeat(count, 1)
            with (
                torch.inference_mode(),
                set_forward_context(None, config, num_tokens=count),
            ):
                y = module._native_forward(x).float().clone()
            scale = y[0].square().mean().sqrt().clamp_min(1e-8)
            checks.append(
                {
                    "layer": name,
                    "tokens": count,
                    "finite": bool(torch.isfinite(y).all()),
                    "row_rms_relative_to_row0": (
                        (y - y[:1]).square().mean(-1).sqrt() / scale
                    ).tolist(),
                    "singleton_relative_rms": float(
                        (y - singleton).square().mean().sqrt()
                        / singleton.square().mean().sqrt().clamp_min(1e-8)
                    ),
                }
            )
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument(
        "--mla-backend",
        choices=["ROCM_AITER_MLA", "TRITON_MLA"],
        default="ROCM_AITER_MLA",
    )
    parser.add_argument("--probe-moe-rows", action="store_true")
    parser.add_argument("--trace-finite-boundaries", action="store_true")
    parser.add_argument("--diagnostic-prefill-heads64", action="store_true")
    parser.add_argument("--diagnostic-disable-fp8-prefill", action="store_true")
    parser.add_argument("--torch-gold-mla", action="store_true")
    parser.add_argument("--audit-loaded-linears", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--only-effort",
        choices=["default", "none", "low", "medium", "high", "xhigh", "max"],
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    from vllm import LLM, SamplingParams
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.models.raon1_mini import effort_steering as s

    config = dict(
        model=args.model,
        dtype="auto",
        max_model_len=262144,
        tensor_parallel_size=1,
        data_parallel_size=1,
        max_num_batched_tokens=16384,
        max_num_seqs=128,
        gpu_memory_utilization=0.85,
        kv_cache_dtype="fp8_e4m3",
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        compilation_config={"mode": 0, "cudagraph_mode": "PIECEWISE"},
        attention_config={
            "backend_per_kind": {"mla_attention": args.mla_backend},
            "mla_prefill_backend": "ROCM_AITER_FA",
        },
        additional_config={"kda_prefill_backend": "fused"},
        seed=42,
    )
    if args.draft_model:
        config["speculative_config"] = {
            "method": "dflash",
            "model": args.draft_model,
            "num_speculative_tokens": 7,
        }
    result = {
        "status": "STARTING",
        "scope": __doc__,
        "config": config,
        "accuracy": "NOT_EVALUATED",
        "throughput": "NOT_MEASURED",
    }
    write_json(args.output, result)
    if args.diagnostic_disable_fp8_prefill:
        from vllm.v1.attention.backends.mla import rocm_aiter_mla

        rocm_aiter_mla._fp8_mla_prefill_supported = lambda: False
        result["diagnostic_fp8_prefill"] = "DISABLED_USES_CONFIGURED_AITER_FA"
    if args.diagnostic_prefill_heads64:
        from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper

        original_heads = AiterMLAHelper.get_fp8_prefill_num_heads
        AiterMLAHelper.get_fp8_prefill_num_heads = staticmethod(
            lambda heads: 64 if heads == 40 else original_heads(heads)
        )
        result["diagnostic_prefill_heads"] = 64
    trace = None
    if args.trace_finite_boundaries:
        result["diagnostic_segmentation"] = "EXTRA_BREAKS_AND_SYNCHRONIZATION"
        trace = install_finite_trace(result, args.output)
    llm = LLM(**config)
    effective = llm.llm_engine.vllm_config
    result["effective_config"] = repr(effective)
    assert not effective.model_config.enforce_eager
    assert effective.compilation_config.cudagraph_mode.name == "PIECEWISE", (
        "Graph mode was changed during engine initialization"
    )
    assert s.CONFIG is not None, "Release config did not enable steering"
    if args.audit_loaded_linears:
        from .loaded_linears import check

        result["loaded_linears"] = llm.apply_model(
            lambda model: check(model, args.model)
        )
        write_json(args.output, result)
        checks = [c for group in result["loaded_linears"] for c in group]
        assert checks, "No native dense layers audited"
        if not all(
            c["payload_equal"]
            and c["scale_equal"]
            and all(v["pass"] for v in c["torch_gold"])
            for c in checks
        ):
            result["status"] = "FAIL_LOADED_DENSE_GOLD"
            write_json(args.output, result)
            raise RuntimeError("Loaded dense weights/GEMM do not match checkpoint gold")
    if args.probe_moe_rows:
        result["moe_identical_row_probe"] = llm.apply_model(
            lambda model: probe_moe_rows(model, effective)
        )
        write_json(args.output, result)
    # A decoder can turn NaN logits into repeated text. Reject that at the
    # sampler boundary, while retaining the real graph execution upstream.
    import torch

    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    def checked_sampler(original, stage):
        def checked(module, logits, *values, **keywords):
            result["raw_logit_calls"] = result.get("raw_logit_calls", 0) + 1
            valid = (
                (~torch.isnan(logits)).all()
                & (~torch.isposinf(logits)).all()
                & torch.isfinite(logits).any(-1).all()
            )
            if not bool(valid):
                result.update(status="FAIL_NONFINITE_LOGITS", failure_stage=stage)
                write_json(args.output, result)
                torch.save(logits.detach().cpu(), args.output + ".logits.pt")
                raise RuntimeError("Nonfinite raw logits in " + stage)
            return original(module, logits, *values, **keywords)

        return checked

    Sampler.__call__ = checked_sampler(Sampler.__call__, "sampler")
    RejectionSampler.__call__ = checked_sampler(RejectionSampler.__call__, "rejection")
    if args.torch_gold_mla:
        install_mla_gold(result, args.output)
    tok = llm.get_tokenizer()
    messages = [
        {"role": "user", "content": "한국어로 2 더하기 3의 답을 짧게 말해 주세요."}
    ]
    ids = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        truncate_history_thinking=False,
        return_dict=False,
    )
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    records = []
    original = llm.llm_engine.input_processor.process_inputs

    def observed(*values, **keywords):
        request = original(*values, **keywords)
        records.append(
            {
                "id": request.request_id,
                "alpha": s.request_alpha(request.sampling_params),
                "cache_salt": request.cache_salt,
            }
        )
        return request

    llm.llm_engine.input_processor.process_inputs = observed
    efforts = [None, "low", "medium", "high", "xhigh", "max", "none"]
    if args.only_effort:
        efforts = [None if args.only_effort == "default" else args.only_effort]
    params = []
    for effort in efforts:
        request = ChatCompletionRequest(
            model="release", messages=messages, reasoning_effort=effort
        )
        serving = SimpleNamespace(
            _effective_chat_template_kwargs=lambda r: (
                {}
                if r.reasoning_effort is None
                else {"reasoning_effort": r.reasoning_effort}
            )
        )
        assert s.api_request_error(serving, request) is None
        params.append(
            SamplingParams(
                max_tokens=args.max_tokens,
                temperature=0,
                seed=42,
                extra_args=request.vllm_xargs,
            )
        )
    from vllm.models.raon1_mini.model import _GRAPH_REPLAYS

    replays_before = _GRAPH_REPLAYS["n"]
    if trace is not None:
        trace["active"] = True
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for _ in efforts], params, use_tqdm=False
    )
    expected = [s.alpha_for(e) for e in efforts]
    if args.torch_gold_mla:
        assert result["mla_torch_gold"], "Native MLA gold hook did not run"
    if trace is not None:
        result["finite_boundary_checks"] = trace["checks"]
    assert [r["alpha"] for r in records] == expected, records
    if not args.only_effort:
        assert records[0]["cache_salt"] == records[3]["cache_salt"]
        assert records[1]["cache_salt"] != records[3]["cache_salt"]
        assert records[5]["cache_salt"] == records[6]["cache_salt"]
    assert _GRAPH_REPLAYS["n"] > replays_before, "No generation graph replay observed"
    result.update(
        status="PASS",
        graph_replays_generation=_GRAPH_REPLAYS["n"] - replays_before,
        request_steering=records,
        outputs=[
            {
                "effort": effort,
                "token_ids": o.outputs[0].token_ids,
                "text": o.outputs[0].text,
                "finish_reason": o.outputs[0].finish_reason,
            }
            for effort, o in zip(efforts, outputs)
        ],
    )
    runs = [
        max((sum(1 for _ in g) for _, g in groupby(o.outputs[0].token_ids)), default=0)
        for o in outputs
    ]
    result["repeated_token_run"] = runs
    result["output_screen"] = (
        "FAIL_REPETITION" if max(runs, default=0) >= 8 else "NO_LONG_SINGLE_TOKEN_RUN"
    )
    if result["output_screen"] == "FAIL_REPETITION":
        result["status"] = "FAIL_OUTPUT_REPETITION"
    write_json(args.output, result)
    print(json.dumps({"status": result["status"], "requests": len(outputs)}))
    assert result["status"] == "PASS", "Degenerate repetition in the fixed math smoke"


if __name__ == "__main__":
    main()
