# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Instrument real eager inference and retain the first numerical failure."""

import functools
import inspect
import json
import os
import time
from pathlib import Path

import torch

from .common import error, provenance, write_json


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu(v) for v in value]
    return value


class Guard:
    def __init__(self, args):
        self.args = args
        self.calls = 0
        self.snapshot = {}
        self.records = {}
        self.logits_calls = 0
        self.active = False
        self.logits_trace = {}
        self.inside_reject = False
        self.sampler_calls = 0
        self.acceptance_histogram = {}
        self.hidden_trace = {}

    def capture_hidden(self, tensor, stage):
        if self.args.force_token is None or not self.active:
            return
        points = {0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511}
        positions = self.target_positions[: self.live_tokens].tolist()
        tokens = self.target_ids[: self.live_tokens].tolist()
        for i, (position, token) in enumerate(zip(positions, tokens)):
            if position >= self.prompt_length and token != self.args.force_token:
                break
            point = position - self.prompt_length + 1
            if point in points:
                key = stage + "@" + str(point)
                if key not in self.hidden_trace:
                    self.hidden_trace[key] = tensor[i].detach().float().cpu().clone()

    def capture_logits(self, logits, indices, draft=None, base=None):
        if self.args.force_token is None or not self.active:
            return
        if base is None:
            base = len(next(iter(self.records.values()))["token_ids"])
        points = {
            0,
            1,
            2,
            3,
            7,
            8,
            15,
            16,
            31,
            32,
            63,
            64,
            127,
            128,
            255,
            256,
            511,
            512,
            1023,
            2047,
        }
        for offset, row in enumerate(indices):
            if draft is not None and any(
                t != self.args.force_token for t in draft[:offset]
            ):
                break
            position = base + offset
            if position in points and position not in self.logits_trace:
                self.logits_trace[position] = logits[row].detach().float().cpu().clone()

    def check(self, tensor, stage, logits=False):
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return
        if logits:
            # Vocab padding may be -inf; NaN/+inf or a wholly masked row is invalid.
            valid = (~torch.isnan(tensor)).all() & (~torch.isposinf(tensor)).all()
            valid = valid & torch.isfinite(tensor).any(-1).all()
        else:
            valid = torch.isfinite(tensor).all()
        if not bool(valid):
            bundle = dict(
                self.snapshot,
                stage=stage,
                bad_tensor=tensor,
                accepted_outputs=self.records,
                rda_calls=self.calls,
                logits_calls=self.logits_calls,
            )
            torch.save(cpu(bundle), self.args.output + ".failure.pt")
            raise RuntimeError("First nonfinite value at " + stage)

    def install(self):
        from vllm.forward_context import get_forward_context
        from vllm.models.raon1_mini import kda
        from vllm.models.raon1_mini.model import Raon1MiniForCausalLM, RaonDecoderLayer

        original = kda.RaonKDAAttention._forward
        guard = self
        original_model_forward = Raon1MiniForCausalLM.forward

        @functools.wraps(original_model_forward)
        def target_forward(model, input_ids, positions, *args, **kwargs):
            if guard.active:
                guard.target_ids, guard.target_positions = input_ids, positions
            return original_model_forward(model, input_ids, positions, *args, **kwargs)

        Raon1MiniForCausalLM.forward = target_forward

        @functools.wraps(original)
        def forward(layer, **kwargs):
            if not guard.active or guard.args.light:
                return original(layer, **kwargs)
            guard.calls += 1
            context = get_forward_context()
            if context.attn_metadata is None:
                return original(layer, **kwargs)
            m = context.attn_metadata[layer.prefix]
            conv, state = layer.kv_cache
            index_tensors = [
                getattr(m, name, None)
                for name in [
                    "spec_state_indices_tensor",
                    "non_spec_state_indices_tensor",
                ]
            ]
            indices = [v.flatten().long() for v in index_tensors if v is not None]
            ids = (
                torch.cat(indices).unique()
                if indices
                else torch.empty(0, device=state.device, dtype=torch.long)
            )
            assert bool(((ids >= -1) & (ids < state.shape[0])).all()), (
                "Cache index bounds"
            )
            ids = ids[ids > 0]
            guard.snapshot = {
                "layer": layer.prefix,
                "inputs": {k: v for k, v in kwargs.items() if k != "core_attn_out"},
                "cache_rows": ids,
                "recurrent_before": state[ids].clone(),
                "conv_before": conv[ids].clone(),
                "metadata": {
                    name: getattr(m, name, None)
                    for name in [
                        "num_spec_decodes",
                        "num_decodes",
                        "num_prefills",
                        "num_actual_tokens",
                        "num_accepted_tokens",
                        "spec_state_indices_tensor",
                        "non_spec_state_indices_tensor",
                        "spec_query_start_loc",
                        "non_spec_query_start_loc",
                        "spec_token_indx",
                        "non_spec_token_indx",
                    ]
                },
                "conv_weight": layer.conv1d.weight,
                "A_log": layer.A_log,
                "dt_bias": layer.dt_bias,
            }
            for name, value in kwargs.items():
                if name != "core_attn_out":
                    n = m.num_actual_tokens
                    live = (
                        value[:, :n] if name in ("g1", "beta", "ratio") else value[:n]
                    )
                    guard.check(live, layer.prefix + ".input." + name)
            original(layer, **kwargs)
            guard.check(
                kwargs["core_attn_out"][:, : m.num_actual_tokens],
                layer.prefix + ".output",
            )

        kda.RaonKDAAttention._forward = forward
        original_logits = Raon1MiniForCausalLM.compute_logits

        @functools.wraps(original_logits)
        def compute_logits(model, hidden_states):
            if not guard.active:
                return original_logits(model, hidden_states)
            guard.check(hidden_states, "target.hidden_before_logits")
            out = original_logits(model, hidden_states)
            guard.logits_calls += 1
            guard.check(out, "target.logits", logits=True)
            return out

        Raon1MiniForCausalLM.compute_logits = compute_logits
        original_layer = RaonDecoderLayer.forward
        original_init = RaonDecoderLayer.__init__

        @functools.wraps(original_init)
        def decoder_init(layer, config, vllm_config, prefix, aux_stream):
            original_init(layer, config, vllm_config, prefix, aux_stream)
            layer._raon_probe_prefix = prefix

        RaonDecoderLayer.__init__ = decoder_init

        @functools.wraps(original_layer)
        def decoder(layer, *args, **kwargs):
            if guard.active:
                metadata = get_forward_context().attn_metadata
                name = layer._raon_probe_prefix + ".self_attn"
                matches = [
                    v
                    for k, v in metadata.items()
                    if k == name or k.startswith(name + ".")
                ]
                assert len(matches) == 1, "Ambiguous target-layer metadata: " + name
                m = matches[0]
                n = getattr(m, "num_actual_tokens", args[-1].shape[0])
                guard.current_layer = layer._raon_probe_prefix
                guard.live_tokens = n
            out = original_layer(layer, *args, **kwargs)
            if guard.active:
                guard.check(out[:n], layer._raon_probe_prefix + ".hidden")
                guard.capture_hidden(out[:n], layer._raon_probe_prefix + ".hidden")
            return out

        RaonDecoderLayer.forward = decoder
        from vllm.models.raon1_mini.mla_attention import MultiHeadLatentAttention
        from vllm.models.raon1_mini.mlp import RaonMLP, RaonMoE

        def wrap_branch(cls, label):
            original_branch = cls.forward

            @functools.wraps(original_branch)
            def branch(module, *args, **kwargs):
                if guard.active:
                    guard.snapshot["active_branch"] = guard.current_layer + "." + label
                    for name, value in list(enumerate(args)) + list(kwargs.items()):
                        if isinstance(value, torch.Tensor):
                            if value.ndim == 2 and value.shape[-1] == 2560:
                                guard.capture_hidden(
                                    value[: guard.live_tokens],
                                    guard.current_layer + "." + label + ".input",
                                )
                            guard.check(
                                value[: guard.live_tokens],
                                guard.current_layer
                                + "."
                                + label
                                + ".input."
                                + str(name),
                            )
                out = original_branch(module, *args, **kwargs)
                if guard.active:
                    guard.check(
                        out[: guard.live_tokens],
                        guard.current_layer + "." + label + ".output",
                    )
                    guard.capture_hidden(
                        out[: guard.live_tokens], guard.current_layer + "." + label
                    )
                return out

            cls.forward = branch

        wrap_branch(MultiHeadLatentAttention, "mla")
        wrap_branch(RaonMoE, "moe")
        wrap_branch(RaonMLP, "mlp")
        import aiter.mla as aiter_mla

        original_mla = aiter_mla.mla_decode_fwd
        mla_signature = inspect.signature(original_mla)

        @functools.wraps(original_mla)
        def native_mla(*args, **kwargs):
            if not guard.active:
                return original_mla(*args, **kwargs)
            bound = mla_signature.bind(*args, **kwargs).arguments
            guard.check(bound["q"].float(), guard.current_layer + ".mla.quantized_q")
            out = original_mla(*args, **kwargs)
            value = bound["o"][:, :40]
            if not bool(torch.isfinite(value).all()):
                indices = bound["kv_indices"][: int(bound["kv_indptr"][-1])].long()
                cache = bound["kv_buffer"]
                guard.snapshot["native_mla"] = {
                    "arguments": {k: v for k, v in bound.items() if k != "kv_buffer"},
                    "cache_shape": list(cache.shape),
                    "gathered_kv": cache.reshape(-1, cache.shape[-1])[indices],
                    "indices": indices,
                }
                guard.check(value, guard.current_layer + ".mla.native_output")
            return out

        aiter_mla.mla_decode_fwd = native_mla
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.v1.sample.rejection_sampler import RejectionSampler

        original_processor = LogitsProcessor.forward

        @functools.wraps(original_processor)
        def processor(module, *args, **kwargs):
            out = original_processor(module, *args, **kwargs)
            if guard.active:
                guard.check(out, "target_or_draft.logits_processor", logits=True)
            return out

        LogitsProcessor.forward = processor
        original_reject = RejectionSampler.forward

        @functools.wraps(original_reject)
        def reject(module, metadata, draft_probs, logits, sampling_metadata):
            if guard.active:
                guard.check(draft_probs, "sampler.draft_probs")
                guard.check(logits, "sampler.target_logits", logits=True)
            if guard.args.force_token is not None and guard.active:
                rows = (
                    metadata.target_logits_indices.tolist()
                    + metadata.bonus_logits_indices.tolist()
                )
                guard.capture_logits(logits, rows, metadata.draft_token_ids.tolist())
            guard.inside_reject = True
            try:
                return original_reject(
                    module, metadata, draft_probs, logits, sampling_metadata
                )
            finally:
                guard.inside_reject = False

        RejectionSampler.forward = reject
        from vllm.v1.sample.sampler import Sampler

        original_sample = Sampler.forward

        @functools.wraps(original_sample)
        def sample(module, logits, sampling_metadata, *args, **kwargs):
            if not guard.inside_reject and logits.shape[0] == 1:
                guard.capture_logits(logits, [0])
            return original_sample(module, logits, sampling_metadata, *args, **kwargs)

        Sampler.forward = sample
        from vllm.v1.worker.gpu.sample.sampler import Sampler as V2Sampler
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
            RejectionSampler as V2Reject,
        )

        original_v2_sample = V2Sampler.__call__
        original_v2_reject = V2Reject.__call__
        original_v2_params = V2Sampler.apply_sampling_params

        def v2_sample(module, logits, input_batch):
            if guard.active:
                guard.sampler_calls += 1
                guard.check(logits, "v2.sampler.raw_logits", logits=True)
                if input_batch.num_reqs == 1 and guard.args.force_token is not None:
                    positions = input_batch.positions[input_batch.logits_indices]
                    base = int(positions[0]) - guard.prompt_length + 1
                    guard.capture_logits(logits, [0], base=base)
            return original_v2_sample(module, logits, input_batch)

        def v2_reject(
            module, logits, input_batch, draft_logits=None, invalid_drafts=None
        ):
            if guard.active:
                guard.sampler_calls += 1
                guard.check(logits, "v2.rejection.raw_logits", logits=True)
                if draft_logits is not None:
                    mapping = input_batch.idx_mapping.tolist()
                    lengths = input_batch.cu_num_logits_np
                    for i, state_index in enumerate(mapping):
                        count = int(lengths[i + 1] - lengths[i]) - 1
                        if count > 0:
                            begin = int(lengths[i])
                            token_index = input_batch.logits_indices[begin]
                            base_position = int(input_batch.positions[token_index])
                            prefill_length = int(
                                module.sampler.req_states.prefill_len.gpu[state_index]
                            )
                            # V2 deliberately rejects placeholder drafts while the
                            # first input row is still inside the prefill prefix.
                            if base_position < prefill_length:
                                continue
                            guard.snapshot["sampler_context"] = {
                                "state_index": state_index,
                                "count": count,
                                "base_position": base_position,
                                "prefill_length": prefill_length,
                                "cu_num_logits": lengths.tolist(),
                            }
                            guard.check(
                                draft_logits[state_index, :count],
                                "v2.rejection.draft_logits",
                                logits=True,
                            )
                if input_batch.num_reqs == 1 and guard.args.force_token is not None:
                    proposals = input_batch.input_ids[
                        input_batch.logits_indices
                    ].tolist()[1:]
                    positions = input_batch.positions[input_batch.logits_indices]
                    base = int(positions[0]) - guard.prompt_length + 1
                    guard.capture_logits(
                        logits, list(range(logits.shape[0])), proposals, base=base
                    )
            out = original_v2_reject(
                module, logits, input_batch, draft_logits, invalid_drafts
            )
            if guard.active:
                for count in out.num_sampled.tolist():
                    key = str(count)
                    guard.acceptance_histogram[key] = (
                        guard.acceptance_histogram.get(key, 0) + 1
                    )
            return out

        def v2_params(module, *args, **kwargs):
            out = original_v2_params(module, *args, **kwargs)
            if guard.active:
                guard.check(out, "v2.sampler.processed_logits", logits=True)
                if guard.args.force_token is not None:
                    # Diagnostic teacher forcing after recording raw logits.
                    # Keep the serving API validation and production sampler intact.
                    out.fill_(-torch.inf)
                    out[:, guard.args.force_token] = 0.0
            return out

        V2Sampler.__call__ = v2_sample
        V2Reject.__call__ = v2_reject
        V2Sampler.apply_sampling_params = v2_params
        from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM
        from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

        original_candidates = DFlash2Qwen3ForCausalLM.compute_candidates
        original_scores = DFlash2Speculator._score_candidates

        def candidates(module, hidden_states):
            if guard.active:
                guard.snapshot["draft_hidden"] = hidden_states
                guard.check(hidden_states, "dflash2.hidden_before_candidates")
            ids, logits = original_candidates(module, hidden_states)
            if guard.active:
                guard.snapshot["draft_candidates"] = {"ids": ids, "logits": logits}
                guard.check(logits, "dflash2.candidate_unary_logits", logits=True)
            return ids, logits

        def scores(module, candidate_ids, unary_logits, hidden_states):
            out = original_scores(module, candidate_ids, unary_logits, hidden_states)
            if guard.active:
                guard.snapshot["candidate_scores"] = out
                guard.check(out, "dflash2.candidate_selector_scores", logits=True)
            return out

        DFlash2Qwen3ForCausalLM.compute_candidates = candidates
        DFlash2Speculator._score_candidates = scores
        from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Attention

        original_draft_init = DFlashQwen3Attention.__init__

        def draft_init(module, *args, **kwargs):
            original_draft_init(module, *args, **kwargs)

            def before_attention(attention, inputs):
                if not guard.active:
                    return
                metadata = get_forward_context().attn_metadata[attention.layer_name]
                n = metadata.num_actual_tokens
                for name, value in zip(["q", "k", "v"], inputs):
                    guard.check(value[:n], module.layer_name + "." + name)

            def after_attention(attention, inputs, out):
                if not guard.active:
                    return
                metadata = get_forward_context().attn_metadata[attention.layer_name]
                n = metadata.num_actual_tokens
                if not bool(torch.isfinite(out[:n]).all()):
                    rows = (~torch.isfinite(out[:n]).all(-1)).nonzero().flatten()
                    request = torch.searchsorted(
                        metadata.query_start_loc[1:].long(), rows.long(), right=True
                    ).unique()
                    cache = attention.kv_cache
                    blocks = [
                        metadata.block_table[
                            int(i),
                            : (int(metadata.seq_lens[i]) + cache.shape[2] - 1)
                            // cache.shape[2],
                        ]
                        for i in request.tolist()
                    ]
                    ids = torch.cat(blocks).long().unique()
                    ids = ids[(ids >= 0) & (ids < cache.shape[0])][:256]
                    guard.snapshot["draft_attention"] = {
                        "name": module.layer_name,
                        "qkv": list(inputs),
                        "metadata": vars(metadata),
                        "bad_rows": rows,
                        "cache_shape": list(cache.shape),
                        "cache_block_ids": ids,
                        "cache_blocks": cache[ids].clone(),
                        "k_scale": attention._k_scale,
                        "v_scale": attention._v_scale,
                    }
                    guard.check(out[:n], module.layer_name + ".attention_output")

            module.attn.register_forward_pre_hook(before_attention)
            module.attn.register_forward_hook(after_attention)

        DFlashQwen3Attention.__init__ = draft_init
        if guard.args.accepted_drafts is not None:
            original_draft = DFlash2Speculator._generate_draft

            def forced_draft(module, num_reqs, *args, **kwargs):
                original_draft(module, num_reqs, *args, **kwargs)
                if not guard.active:
                    return
                count = guard.args.accepted_drafts
                token = guard.args.force_token
                assert token is not None and 0 <= count <= guard.args.draft_tokens
                module.draft_tokens[:num_reqs].fill_((token + 1) % 259840)
                module.draft_tokens[:num_reqs, :count].fill_(token)
                if module.draft_logits is not None:
                    module.draft_logits.fill_(-torch.inf)
                    tokens = (
                        module.draft_tokens[:1]
                        .long()
                        .expand(module.draft_logits.shape[:2])
                    )
                    module.draft_logits.scatter_(2, tokens[..., None], 0.0)

            DFlash2Speculator._generate_draft = forced_draft
        return {
            "rda": provenance(original),
            "v2_sampler": provenance(original_v2_sample),
            "v2_rejection": provenance(original_v2_reject),
        }


def run(args, result):
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    assert args.model and args.prompts_json
    if args.mla_heads != 48:
        assert args.mla_heads == 64, "Only the explicit H40->64 candidate is supported"
        from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper

        os.environ["VLLM_ROCM_AITER_MLA_PAD_H40_TO_64"] = "1"
        assert AiterMLAHelper.get_actual_mla_num_heads(40) == 64, (
            "Install the vLLM H40 padding patch before using --mla-heads 64"
        )
        result["candidate_kernel_change"] = (
            "Decode head padding H40->64; prefill unchanged"
        )
    guard = Guard(args)
    result["source"] = guard.install()
    config = dict(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        data_parallel_size=1,
        max_model_len=262144,
        max_num_batched_tokens=16384,
        max_num_seqs=16,
        gpu_memory_utilization=0.80,
        enforce_eager=True,
        enable_prefix_caching=False,
        kv_cache_dtype=args.kv_cache_dtype,
        mamba_cache_mode="none",
        seed=42,
        attention_config={
            "backend_per_kind": {"mla_attention": "ROCM_AITER_MLA"},
            "mla_prefill_backend": "ROCM_AITER_FA",
        },
        additional_config={"kda_prefill_backend": "fused"},
    )
    if args.draft_tokens:
        assert args.draft_model
        config["speculative_config"] = dict(
            method="dflash",
            model=args.draft_model,
            num_speculative_tokens=args.draft_tokens,
            draft_sample_method="probabilistic",
        )
    result["requested_config"] = config
    llm = LLM(**config)
    result["observed_config"] = repr(llm.llm_engine.vllm_config)
    tokenizer = llm.get_tokenizer()
    rows = json.loads(Path(args.prompts_json).read_text())
    assert 0 < len(rows) <= 160
    if args.command == "teacher":
        assert args.force_token is not None and args.temperature == 0 and len(rows) == 1
    prompts = {}
    for i, row in enumerate(rows):
        rid = str(row.get("id", i))
        assert rid not in prompts
        ids = row.get("prompt_token_ids")
        if ids is None:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
                truncate_history_thinking=False,
                return_dict=False,
            )
        assert 0 < len(ids) < 262144
        prompts[rid] = ids
        if args.force_token is not None:
            guard.prompt_length = len(ids)
        guard.records[rid] = {"token_ids": [], "finished": False, "logprobs": []}
        params = SamplingParams(
            temperature=args.temperature,
            top_p=0.95 if args.temperature else 1.0,
            top_k=20 if args.temperature else -1,
            presence_penalty=1.5,
            max_tokens=min(args.max_tokens, 262144 - len(ids)),
            min_tokens=0,
            ignore_eos=False,
            stop_token_ids=[259202, 259200],
            seed=42 + int(rid),
            logprobs=5,
            output_kind=RequestOutputKind.DELTA,
        )
        llm.llm_engine.add_request(rid, {"prompt_token_ids": ids}, params)
    result["prompt_tokens"] = prompts
    guard.active = True
    started = time.monotonic()
    try:
        while llm.llm_engine.has_unfinished_requests():
            if time.monotonic() - started >= args.generation_seconds:
                result["incomplete"] = True
                break
            for out in llm.llm_engine.step():
                record = guard.records[out.request_id]
                if out.outputs:
                    choice = out.outputs[0]
                    record["token_ids"].extend(choice.token_ids)
                    if args.force_token is not None:
                        assert all(t == args.force_token for t in choice.token_ids), (
                            "Teacher forcing failed"
                        )
                    record["logprobs"].extend(
                        [
                            {str(k): float(v.logprob) for k, v in dist.items()}
                            for dist in (choice.logprobs or [])
                        ]
                    )
                    record["finished"] = out.finished
                    record["finish_reason"] = choice.finish_reason
                    if len(record["token_ids"]) >= 256 and all(
                        t == 0 for t in record["token_ids"][-256:]
                    ):
                        raise RuntimeError("256 consecutive token0 outputs")
            if guard.logits_calls % 20 == 0:
                result["generated_tokens"] = {
                    k: len(v["token_ids"]) for k, v in guard.records.items()
                }
                write_json(args.output, result)
    finally:
        result["records"] = guard.records
        for record in result["records"].values():
            record["text"] = tokenizer.decode(record["token_ids"])
        result["generation_seconds"] = time.monotonic() - started
        result["rda_calls"] = guard.calls
        result["logits_calls"] = guard.logits_calls
        result["sampler_calls"] = guard.sampler_calls
        result["acceptance_histogram"] = guard.acceptance_histogram
        if args.force_token is not None:
            torch.save(guard.logits_trace, args.output + ".logits.pt")
            torch.save(guard.hidden_trace, args.output + ".hidden.pt")
            result["raw_logit_positions"] = sorted(guard.logits_trace)
            result["hidden_trace_count"] = len(guard.hidden_trace)
            assert guard.logits_trace, (
                "No raw-logit hook executed; diagnostic is invalid"
            )
    result["cases"].append(
        {
            "finite_outputs": {"pass": True},
            "generation_ran": {"pass": guard.logits_calls > 0},
            "sampler_hook_ran": {"pass": guard.sampler_calls > 0},
        }
    )
    if args.accepted_drafts is not None:
        result["cases"].append(
            {
                "forced_acceptance_exercised": {
                    "pass": str(args.accepted_drafts + 1) in guard.acceptance_histogram
                }
            }
        )
    if args.reference_json:
        reference = json.loads(Path(args.reference_json).read_text())
        assert reference["prompt_tokens"] == prompts, "Unmatched prompts"
        for key in config:
            if key != "speculative_config":
                assert reference["requested_config"][key] == config[key], key
        if args.force_token is not None:
            raw_reference = torch.load(
                args.reference_json + ".logits.pt", weights_only=True
            )
            positions = sorted(set(raw_reference) & set(guard.logits_trace))
            assert positions, "No matched teacher-forced positions"
            for position in positions:
                a, b = guard.logits_trace[position], raw_reference[position]
                result["cases"].append(
                    {
                        "teacher_position": position,
                        "raw_logits": error(a, b),
                        "target_argmax_equal": bool(a.argmax() == b.argmax()),
                    }
                )
            path = Path(args.reference_json + ".hidden.pt")
            if path.exists():
                hidden_reference = torch.load(path, weights_only=True)
                for key in sorted(set(hidden_reference) & set(guard.hidden_trace)):
                    result["cases"].append(
                        {
                            "hidden_key": key,
                            "hidden": error(
                                guard.hidden_trace[key], hidden_reference[key]
                            ),
                        }
                    )
            return
        comparisons = []
        for rid, record in guard.records.items():
            a, b = reference["records"][rid]["token_ids"], record["token_ids"]
            mismatch = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            comparisons.append(
                {
                    "id": rid,
                    "first_mismatch": mismatch,
                    "common_tokens_checked": min(len(a), len(b)),
                    "reference_finished": reference["records"][rid]["finished"],
                    "candidate_finished": record["finished"],
                }
            )
        result["greedy_comparison"] = comparisons
        result["cases"].append(
            {
                "matched_prefix": {
                    "pass": all(c["first_mismatch"] is None for c in comparisons)
                }
            }
        )
