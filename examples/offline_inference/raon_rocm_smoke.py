# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run four natural-EOS Raon tasks with TP1/DP1, optionally using DFlash2.

Source docker/raon/runtime-env.sh before starting the interpreter.
"""

import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="KRAFTON/Raon-1-mini-preview")
    parser.add_argument(
        "--revision", default="1b4f494d71dff50318288309f64c72aa8c11340e"
    )
    parser.add_argument("--draft", default=None)
    parser.add_argument(
        "--draft-revision", default="164b3dda020f145a4650b3c050841e15250c2f77"
    )
    args = parser.parse_args()
    config = dict(
        model=args.model,
        tensor_parallel_size=1,
        data_parallel_size=1,
        max_model_len=262144,
        max_num_batched_tokens=16384,
        max_num_seqs=16,
        gpu_memory_utilization=0.80,
        enforce_eager=True,
        enable_prefix_caching=False,
        kv_cache_dtype="fp8_e4m3",
        mamba_cache_mode="none",
        attention_config={
            "backend_per_kind": {"mla_attention": "ROCM_AITER_MLA"},
            "mla_prefill_backend": "ROCM_AITER_FA",
        },
        additional_config={"kda_prefill_backend": "fused"},
        seed=42,
    )
    if not Path(args.model).is_dir():
        config["revision"] = args.revision
    if args.draft:
        spec = dict(
            method="dflash",
            model=args.draft,
            num_speculative_tokens=7,
            draft_sample_method="probabilistic",
        )
        if not Path(args.draft).is_dir():
            spec["revision"] = args.draft_revision
        config["speculative_config"] = spec
    llm = LLM(**config)
    tokenizer = llm.get_tokenizer()
    questions = [
        "Calculate 17 multiplied by 23. Give the integer result.",
        "List the first eight prime numbers and explain how you checked them.",
        "Translate into Korean: A clear experiment records its inputs and results.",
        "Read this sequence carefully: "
        + ", ".join(str(i * i) for i in range(1, 100))
        + ". What rule generated it, and what are the next two values?",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            truncate_history_thinking=False,
            return_dict=False,
        )
        for question in questions
    ]
    params = [
        SamplingParams(
            temperature=0,
            max_tokens=262144 - len(ids),
            min_tokens=0,
            ignore_eos=False,
            stop_token_ids=[259202, 259200],
            skip_special_tokens=False,
            seed=772,
        )
        for ids in prompts
    ]
    results = llm.generate([{"prompt_token_ids": ids} for ids in prompts], params)
    records = []
    for question, result in zip(questions, results):
        choice = result.outputs[0]
        text = tokenizer.decode(choice.token_ids, skip_special_tokens=False)
        records.append(
            {
                "question": question,
                "output_tokens": len(choice.token_ids),
                "finish_reason": choice.finish_reason,
                "text": text,
            }
        )
    print(json.dumps(records, ensure_ascii=False, indent=2))
    assert len(records) == 4
    assert all(
        r["finish_reason"] == "stop" and "</think>" in r["text"] for r in records
    )


if __name__ == "__main__":
    main()
